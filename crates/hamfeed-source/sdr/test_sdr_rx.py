#!/usr/bin/env python3
"""Regression tests for the sdr_rx PSD gate (DC removal + threshold +
concentration). Stdlib unittest only; run with ./../.venv/bin/python
(or any python3 with numpy) as:  python3 test_sdr_rx.py

The uhd import is stubbed: these tests exercise the pure DSP
(psd_snr / gate_open / Demod gating) on synthetic complex samples,
no radio hardware needed.
"""
import sys
import types
import unittest

import numpy as np

sys.modules.setdefault("uhd", types.ModuleType("uhd"))
import sdr_rx  # noqa: E402  (uhd stubbed above)


def rng_blocks(seed, n_blocks=4, block=12500):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_blocks, block))
            + 1j * rng.standard_normal((n_blocks, block))).astype(np.complex64)


class TestPsdSnr(unittest.TestCase):
    def test_short_input_returns_none(self):
        self.assertIsNone(sdr_rx.psd_snr(np.zeros(100, dtype=np.complex64)))

    def test_dc_spike_does_not_open_gate(self):
        # The B210 failure mode: a large DC offset with nothing else
        # on air. Pre-fix this read ~17 dB in bin 0 on every
        # frequency; post-fix the gate must stay closed.
        noise = rng_blocks(7)
        dc = (0.5 + 0.3j) * np.ones_like(noise)
        floor, peak, snr, conc, level = sdr_rx.psd_snr((noise + dc).reshape(-1))
        self.assertFalse(
            sdr_rx.gate_open(snr, conc, -40.0, sdr_rx.effective_squelch(13.0)),
            f"DC spike opened gate: snr={snr:.1f} conc={conc:.1f}",
        )

    def test_noise_stays_closed(self):
        floor, peak, snr, conc, level = sdr_rx.psd_snr(rng_blocks(11).reshape(-1))
        self.assertLess(snr, sdr_rx.MIN_SQUELCH_DB,
                        f"noise snr {snr:.1f} above gate floor")
        self.assertFalse(sdr_rx.gate_open(snr, conc, -40.0, 14.5))

    def test_carrier_opens_gate(self):
        # Complex tone (concentrated carrier) over noise.
        n = 4 * 12500
        t = np.arange(n)
        tone = 0.6 * np.exp(2j * np.pi * 0.05 * t).astype(np.complex64)
        rng = np.random.default_rng(3)
        noise = (0.05 * (rng.standard_normal(n)
                         + 1j * rng.standard_normal(n))).astype(np.complex64)
        floor, peak, snr, conc, level = sdr_rx.psd_snr(tone + noise)
        self.assertGreaterEqual(conc, sdr_rx.MIN_CONCENTRATION_DB,
                                f"carrier conc {conc:.1f} below floor")
        self.assertTrue(sdr_rx.gate_open(snr, conc, -40.0, 14.5),
                        f"carrier failed gate: snr={snr:.1f} conc={conc:.1f}")

    def test_carrier_more_concentrated_than_noise(self):
        n = 4 * 12500
        t = np.arange(n)
        tone = 0.6 * np.exp(2j * np.pi * 0.05 * t).astype(np.complex64)
        rng = np.random.default_rng(5)
        noise = (0.05 * (rng.standard_normal(n)
                         + 1j * rng.standard_normal(n))).astype(np.complex64)
        _, _, _, conc_sig, _ = sdr_rx.psd_snr(tone + noise)
        _, _, _, conc_noise, _ = sdr_rx.psd_snr(rng_blocks(5).reshape(-1))
        self.assertGreater(conc_sig, conc_noise + 3.0,
                           f"sig {conc_sig:.1f} vs noise {conc_noise:.1f}")

    def test_near_dc_hump_stays_closed(self):
        # Residual LO/wander hump near DC (measured B210 bins 3-7):
        # strong and concentrated, but NOT rf. The peak search must
        # ignore the DC zone, so this stays closed.
        n = 4 * 12500
        t = np.arange(n)
        hump = 0.3 * np.exp(2j * np.pi * 5 / 4096 * t).astype(np.complex64)
        rng = np.random.default_rng(9)
        noise = (0.05 * (rng.standard_normal(n)
                         + 1j * rng.standard_normal(n))).astype(np.complex64)
        floor, peak, snr, conc, level = sdr_rx.psd_snr(hump + noise)
        self.assertFalse(sdr_rx.gate_open(snr, conc, -40.0, 14.5),
                         f"DC hump opened gate: snr={snr:.1f} conc={conc:.1f}")

    def test_level_reported(self):
        n = 12500
        x = (0.01 * np.ones(n, dtype=np.complex64)
             + 0j * np.ones(n, dtype=np.complex64))
        _, _, _, _, level = sdr_rx.psd_snr(x)
        self.assertAlmostEqual(level, -40.0, places=1)

    def test_digital_residue_stays_closed(self):
        # The no-RF failure mode (measured B210: samples ~1e-4,
        # level ~-79 dBFS): the residue SHAPE can mimic a signal
        # (hump + crest), so the absolute level guard must hold
        # the gate closed regardless of snr/concentration.
        n = 4 * 12500
        rng = np.random.default_rng(13)
        residue = (1e-4 * (rng.standard_normal(n)
                           + 1j * rng.standard_normal(n)))
        t = np.arange(n)
        hump = 4e-4 * np.exp(2j * np.pi * 5 / 4096 * t)
        x = (residue + hump).astype(np.complex64)
        floor, peak, snr, conc, level = sdr_rx.psd_snr(x)
        self.assertLess(level, sdr_rx.MIN_LEVEL_DBFS,
                        f"residue level {level:.1f} above guard")
        self.assertFalse(sdr_rx.gate_open(30.0, 30.0, level, 14.5),
                         "level guard bypassed by shape metrics")

    def test_effective_squelch_floor(self):
        self.assertEqual(sdr_rx.effective_squelch(13.0), sdr_rx.MIN_SQUELCH_DB)
        self.assertEqual(sdr_rx.effective_squelch(20.0), 20.0)

    def test_demod_gates_noise_to_zeros(self):
        d = sdr_rx.Demod(rate=250000.0, squelch_db=13.0, hang_s=1.5)
        pcm, opened, _snr = d.process(rng_blocks(21, n_blocks=1).reshape(-1))
        self.assertFalse(opened)
        self.assertTrue((pcm == 0).all())

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            sdr_rx.Demod(rate=250000.0, squelch_db=14.5, hang_s=1.5,
                         mode="ssb")

    def test_nbfm_recovers_fm_tone(self):
        # Carrier FM-modulated at 1 kHz, 2.5 kHz deviation: the
        # discriminator output must track the modulating tone.
        n = 12500
        t = np.arange(n) / 250000.0
        phase = 2 * np.pi * (3000 * t
                             + 2500 / (2 * np.pi * 1000)
                             * np.sin(2 * np.pi * 1000 * t))
        rng = np.random.default_rng(23)
        x = (2.0 * np.exp(1j * phase)
             + 0.02 * (rng.standard_normal(n)
                       + 1j * rng.standard_normal(n))).astype(np.complex64)
        d = sdr_rx.Demod(rate=250000.0, squelch_db=1.0, hang_s=1.5,
                         mode="nbfm")
        pcm, opened, _snr = d.process(x)
        self.assertTrue(opened)
        # Discriminator recovers frequency: the cosine, not the sine.
        ref = np.cos(2 * np.pi * 1000 * np.arange(800) / 16000.0)
        corr = float(np.corrcoef(pcm.astype(float), ref)[0, 1])
        self.assertGreater(abs(corr), 0.8,
                           f"nbfm lost the tone: corr={corr:.2f}")

    def test_nbfm_silent_on_dead_carrier(self):
        # Unmodulated carrier = constant frequency offset = DC:
        # blocked, near silence out.
        n = 12500
        t = np.arange(n) / 250000.0
        rng = np.random.default_rng(24)
        x = (2.0 * np.exp(2j * np.pi * 3000 * t)
             + 0.02 * (rng.standard_normal(n)
                       + 1j * rng.standard_normal(n))).astype(np.complex64)
        d = sdr_rx.Demod(rate=250000.0, squelch_db=1.0, hang_s=1.5,
                         mode="nbfm")
        pcm, opened, _snr = d.process(x)
        self.assertTrue(opened)  # carrier holds the gate...
        # ...but carries no audio: second block, DC blocker settled.
        pcm2, _, _ = d.process(x)
        self.assertLess(np.abs(pcm2.astype(float)).mean(), 500,
                        "...but carries no audio")

    def test_wfm_recovers_tone_without_clipping(self):
        # Broadcast FM: 1 kHz tone at full 75 kHz deviation. NBFM
        # scaling would clip this ~30x over; the WFM path must track
        # it near full scale but unclipped.
        n = 12500
        t = np.arange(n) / 250000.0
        phase = 2 * np.pi * (3000 * t
                             + 75000 / (2 * np.pi * 1000)
                             * np.sin(2 * np.pi * 1000 * t))
        rng = np.random.default_rng(31)
        x = (2.0 * np.exp(1j * phase)
             + 0.02 * (rng.standard_normal(n)
                       + 1j * rng.standard_normal(n))).astype(np.complex64)
        d = sdr_rx.Demod(rate=250000.0, squelch_db=1.0, hang_s=1.5,
                         mode="wfm")
        pcm, opened, _snr = d.process(x)
        self.assertTrue(opened)
        ref = np.cos(2 * np.pi * 1000 * np.arange(800) / 16000.0)
        corr = float(np.corrcoef(pcm.astype(float), ref)[0, 1])
        self.assertGreater(abs(corr), 0.8,
                           f"wfm lost the tone: corr={corr:.2f}")
        self.assertLess(np.abs(pcm.astype(float)).max(), 32767,
                        "wfm clipped at full deviation")

    def test_wfm_deemphasis_rolls_off_highs(self):
        # 75 us de-emphasis (-3 dB at ~2.1 kHz): a 5 kHz tone must
        # come out weaker than a 1 kHz tone at equal deviation.
        def level(f_hz):
            n = 12500
            t = np.arange(n) / 250000.0
            phase = 2 * np.pi * (3000 * t
                                 + 20000 / (2 * np.pi * f_hz)
                                 * np.sin(2 * np.pi * f_hz * t))
            x = (2.0 * np.exp(1j * phase)).astype(np.complex64)
            d = sdr_rx.Demod(rate=250000.0, squelch_db=1.0, hang_s=1.5,
                             mode="wfm")
            d.process(x)  # settle DC blocker
            pcm, _, _ = d.process(x)
            return np.abs(pcm.astype(float)).mean()
        self.assertGreater(level(1000), level(5000) * 1.5,
                           "no de-emphasis rolloff measured")

    def test_am_recovers_envelope(self):
        # Carrier AM-modulated at 100 Hz, 50% depth: envelope out
        # must track the modulating tone.
        n = 12500
        t = np.arange(n) / 250000.0
        env = 1.0 + 0.5 * np.sin(2 * np.pi * 100 * t)
        x = (2.0 * env * np.exp(2j * np.pi * 3000 * t)).astype(np.complex64)
        d = sdr_rx.Demod(rate=250000.0, squelch_db=1.0, hang_s=1.5,
                         mode="am")
        pcm, opened, _snr = d.process(x)
        self.assertTrue(opened)
        ref = np.sin(2 * np.pi * 100 * np.arange(800) / 16000.0)
        corr = float(np.corrcoef(pcm.astype(float), ref)[0, 1])
        self.assertGreater(abs(corr), 0.8,
                           f"am lost the envelope: corr={corr:.2f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
