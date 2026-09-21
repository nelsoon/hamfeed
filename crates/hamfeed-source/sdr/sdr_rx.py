#!/usr/bin/env python3
"""sdr_rx: B210 NBFM capture shim for hamfeed (SdrSource child process).

Opens the Ettus B210 with the station parameters measured on
10.0.2.102 (gain 20, 250 ksps, analog BW <= 200 kHz, RX2), NBFM
demodulates one channel, SNR-gates it, and streams fixed 20 ms
16 kHz mono S16LE records on stdout. Squelch-closed time emits
zeros so the downstream sample clock never stalls.

DSP mirrors the working probes (repeater_watch.py fmdemod + PSD
gate, demod.py scaling): polar discriminator, ~3 kHz moving
average, linear-interp resample 250k -> 16k, one-pole DC block,
fixed scaling (2.5 kHz deviation -> 0.8 FS, clipped). No per-block
AGC: gated zeros handle silence, fixed gain handles speech.

TX chain is parked at gain 0 and never streamed (internal leakage).

Usage (stream):
  sdr_rx.py --freq 145.11e6 --gain 20 --rate 250e3 --bw 200e3 \\
      --squelch-db 13 --hang-s 1.5 [--chan 0] [--antenna RX2]

Probe (one JSON line on stdout, then exit):
  sdr_rx.py --freq 145.11e6 --probe [--probe-secs 2]

Retune = restart the process with a new --freq (1-2 s gap).
Fatal errors go to stderr with a nonzero exit; the parent (Rust
SdrSource) logs them and reopens the source.
"""
import argparse
import json
import sys
import time

import numpy as np
import uhd

OUT_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = OUT_RATE * FRAME_MS // 1000  # 320
BLOCK_S = 0.05  # DSP block: 50 ms of RF
SEG = 4096  # PSD FFT size (probe formula)
DEVIATION_HZ = 2500.0  # NBFM peak deviation: full scale reference


def open_usrp(args, freq, rate, gain, bw, chan, antenna):
    usrp = uhd.usrp.MultiUSRP(args)
    for ch in range(usrp.get_tx_num_channels()):
        usrp.set_tx_gain(0, ch)
    usrp.set_rx_rate(rate)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq), chan)
    usrp.set_rx_antenna(antenna, chan)
    usrp.set_rx_gain(gain, chan)
    usrp.set_rx_bandwidth(min(rate, bw), chan)
    time.sleep(0.5)  # device + AGC settle (probe practice)
    return usrp


def start_stream(usrp, chan):
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [chan]
    streamer = usrp.get_rx_stream(st_args)
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)
    return streamer


def recv_block(streamer, n):
    buf = np.zeros((1, n), dtype=np.complex64)
    md = uhd.types.RXMetadata()
    got = 0
    while got < n:
        m = streamer.recv(buf, md, timeout=5.0)
        if md.error_code != uhd.types.RXMetadataErrorCode.none:
            raise RuntimeError(f"rx error: {md.error_code} ({md.strerror()})")
        got += m
    return buf[0]


def psd_snr(x):
    """Probe PSD gate: Hann-averaged FFT, floor = median, snr = peak - floor."""
    m = (len(x) // SEG) * SEG
    if m == 0:
        return None
    segs = x[:m].reshape(-1, SEG) * np.hanning(SEG)
    p = (np.abs(np.fft.fft(segs, axis=1)) ** 2).mean(axis=0)
    pdb = 10 * np.log10(p + 1e-18)
    floor = float(np.median(pdb))
    peak = float(pdb.max())
    return floor, peak, peak - floor


class Demod:
    """Streaming NBFM demod: discriminator -> smooth -> resample -> DC
    block -> fixed scale. State (hang counter, DC estimate) persists
    across blocks; no per-block normalization (no AGC pumping)."""

    def __init__(self, rate, squelch_db, hang_s):
        self.rate = rate
        self.squelch_db = squelch_db
        self.hang_blocks = max(1, int(round(hang_s / BLOCK_S)))
        self.hang_left = 0
        self.k = max(int(rate / 3000), 1)
        self.kernel = np.ones(self.k) / self.k
        self.out_per_block = int(round(OUT_RATE * BLOCK_S))
        self.dc = 0.0
        self.scale = 0.8 * 32767 / DEVIATION_HZ

    def process(self, x):
        """Demodulate one RF block -> int16 audio at OUT_RATE. Returns
        (audio, gate_open, snr_db). Closed gate returns zeros."""
        metrics = psd_snr(x)
        snr = metrics[2] if metrics else -1e9
        if snr >= self.squelch_db:
            self.hang_left = self.hang_blocks
        elif self.hang_left > 0:
            self.hang_left -= 1
        if self.hang_left == 0:
            return np.zeros(self.out_per_block, dtype=np.int16), False, snr
        # Polar discriminator (probe formula).
        d = np.angle(x[1:] * np.conj(x[:-1])) * self.rate / (2 * np.pi)
        d = np.convolve(d, self.kernel, mode="same")
        # Linear-interp resample to OUT_RATE.
        t_in = np.arange(len(d), dtype=np.float64)
        t_out = np.linspace(0, len(d) - 1, self.out_per_block)
        a = np.interp(t_out, t_in, d)
        # One-pole DC blocker (streaming median-substitute).
        y = np.empty_like(a)
        for i, v in enumerate(a):
            self.dc += 0.005 * (v - self.dc)
            y[i] = v - self.dc
        pcm = np.clip(y * self.scale, -32768, 32767).astype(np.int16)
        return pcm, True, snr


def run_stream(a):
    usrp = open_usrp(a.args, a.freq, a.rate, a.gain, a.bw, a.chan, a.antenna)
    streamer = start_stream(usrp, a.chan)
    n = int(a.rate * BLOCK_S)
    demod = Demod(a.rate, a.squelch_db, a.hang_s)
    out = sys.stdout.buffer
    try:
        while True:
            x = recv_block(streamer, n)
            pcm, _, _ = demod.process(x)
            # Fixed-size records: pad/trim to exactly FRAME multiples.
            for i in range(0, len(pcm), FRAME_SAMPLES):
                rec = pcm[i:i + FRAME_SAMPLES]
                if len(rec) < FRAME_SAMPLES:
                    rec = np.pad(rec, (0, FRAME_SAMPLES - len(rec)))
                out.write(rec.tobytes())
            out.flush()
    except BrokenPipeError:
        pass


def run_probe(a):
    usrp = open_usrp(a.args, a.freq, a.rate, a.gain, a.bw, a.chan, a.antenna)
    streamer = start_stream(usrp, a.chan)
    n = int(a.rate * BLOCK_S)
    floors, peaks, snrs = [], [], []
    t_end = time.time() + a.probe_secs
    while time.time() < t_end:
        x = recv_block(streamer, n)
        r = psd_snr(x)
        if r:
            floors.append(r[0])
            peaks.append(r[1])
            snrs.append(r[2])
    row = {
        "freq_hz": a.freq,
        "floor_dbfs": round(float(np.median(floors)), 2) if floors else None,
        "peak_dbfs": round(float(max(peaks)), 2) if peaks else None,
        "snr_db": round(float(max(snrs)), 2) if snrs else None,
        "gate_open": bool(snrs and max(snrs) >= a.squelch_db),
        "squelch_db": a.squelch_db,
    }
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, required=True)
    ap.add_argument("--rate", type=float, default=250e3)
    ap.add_argument("--gain", type=float, default=20)
    ap.add_argument("--bw", type=float, default=200e3)
    ap.add_argument("--squelch-db", type=float, default=13)
    ap.add_argument("--hang-s", type=float, default=1.5)
    ap.add_argument("--chan", type=int, default=0)
    ap.add_argument("--antenna", default="RX2")
    ap.add_argument("--args", default="")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-secs", type=float, default=2.0)
    a = ap.parse_args()
    try:
        if a.probe:
            run_probe(a)
        else:
            run_stream(a)
    except RuntimeError as e:
        print(f"sdr_rx: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
