//! hamfeed-pipeline binary: live capture → transcribe → store.

use std::path::PathBuf;

use hamfeed_pipeline::Pipeline;

fn usage() -> ! {
    eprintln!(
        "usage: hamfeed-pipeline [--config PATH] [--once] [--fake SECS] [--simplex] [--list-devices] [--sdr-probe]"
    );
    std::process::exit(2);
}

fn main() {
    let mut config = PathBuf::from("hamfeed.toml");
    let mut once = false;
    let mut fake_secs: Option<u64> = None;
    let mut profile = "repeater";
    let mut sdr_probe = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config = PathBuf::from(args.next().unwrap_or_else(|| usage())),
            "--sdr-probe" => sdr_probe = true,
            "--list-devices" => {
                #[cfg(feature = "capture")]
                {
                    for name in hamfeed_source::list_input_devices() {
                        println!("{name}");
                    }
                    return;
                }
                #[cfg(not(feature = "capture"))]
                {
                    eprintln!("hamfeed-pipeline: rebuild with --features capture");
                    std::process::exit(2);
                }
            }
            "--once" => once = true,
            "--fake" => {
                fake_secs = Some(
                    args.next()
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            "--simplex" => profile = "simplex",
            _ => usage(),
        }
    }

    let pipe = Pipeline::open(&config).unwrap_or_else(|e| {
        eprintln!("hamfeed-pipeline: cannot start\n{e:?}");
        std::process::exit(1);
    });
    // Live monitor relay (Slice 3 fix): the web UI runs in its own process
    // and cannot see our in-memory tap, so publish tap PCM on a socket the
    // web `/api/live` handler subscribes to. No library change: the tap
    // handle is already shareable; this thread only reads it.
    std::thread::spawn({
        let tap = pipe.live_tap();
        let sock = std::path::PathBuf::from(&pipe.config().storage.dir).join("live.sock");
        move || serve_live_tap(tap, sock)
    });
    let recovered = pipe.startup_recovery().unwrap_or_else(|e| {
        eprintln!("hamfeed-pipeline: spill replay failed: {e:?}");
        std::process::exit(1);
    });
    if recovered > 0 {
        eprintln!("hamfeed-pipeline: replayed {recovered} spilled clip(s)");
    }

    if let Some(secs) = fake_secs {
        // Headless simulation: scripted speech-like audio, no hardware.
        let mut pcm = hamfeed_ingest::fixture::speech_like((secs * 4) as usize);
        pcm.extend(hamfeed_ingest::fixture::silence_ms(1200));
        let frames = hamfeed_ingest::fixture::chunks(&pcm, 1600);
        let mut src = hamfeed_source::FakeSource::once(
            frames
                .into_iter()
                .map(|c| hamfeed_source::PcmFrame { samples: c })
                .collect(),
        );
        let cfg = load_cfg(&config);
        let n = pipe
            .run_source(
                &mut src,
                cfg.hang_ms(profile),
                cfg.segment.max_s,
                cfg.vad.beep_split,
                cfg.vad.beep_min_ms,
            )
            .unwrap_or_else(|e| {
                eprintln!("hamfeed-pipeline: simulation failed: {e:?}");
                std::process::exit(1);
            });
        println!("simulated {secs}s: {n} segment(s) stored");
        return;
    }

    if once {
        let n = pipe.drain().unwrap_or_else(|e| {
            eprintln!("hamfeed-pipeline: drain failed: {e:?}");
            std::process::exit(1);
        });
        println!("drained {n} queued clip(s)");
        return;
    }

    if sdr_probe {
        run_sdr_probe(&config);
        return;
    }

    #[cfg(feature = "sdr")]
    {
        let cfg = load_cfg(&config);
        if cfg.source.kind == "sdr" {
            run_sdr_loop(&pipe, &config, profile);
            return;
        }
    }
    #[cfg(not(feature = "sdr"))]
    {
        if load_cfg(&config).source.kind == "sdr" {
            eprintln!(
                "hamfeed-pipeline: sdr source needs the sdr feature \
                 (rebuild with --features sdr); --fake SECS works without it"
            );
            std::process::exit(2);
        }
    }

    #[cfg(feature = "capture")]
    {
        let cfg = load_cfg(&config);
        let mut mic = hamfeed_source::MicSource::open(&cfg.audio.device).unwrap_or_else(|e| {
            eprintln!("hamfeed-pipeline: mic failed: {e:?}");
            std::process::exit(1);
        });
        eprintln!(
            "hamfeed-pipeline: capturing from {} (profile {profile})",
            mic.device_name()
        );
        let n = pipe
            .run_source(
                &mut mic,
                cfg.hang_ms(profile),
                cfg.segment.max_s,
                cfg.vad.beep_split,
                cfg.vad.beep_min_ms,
            )
            .unwrap_or_else(|e| {
                eprintln!("hamfeed-pipeline: capture failed: {e:?}");
                std::process::exit(1);
            });
        println!("capture ended: {n} segment(s) stored");
    }
    #[cfg(not(feature = "capture"))]
    {
        eprintln!(
            "hamfeed-pipeline: live mic needs the capture feature \
             (rebuild with --features capture); --fake SECS works without it"
        );
        std::process::exit(2);
    }
}

fn load_cfg(path: &std::path::Path) -> hamfeed_config::Config {
    hamfeed_config::load(path).unwrap_or_else(|e| {
        eprintln!("hamfeed-pipeline: bad config: {e:?}");
        std::process::exit(1);
    })
}

/// Floor/SNR probe across the configured channels, mirroring
/// `--list-devices`: one JSON row per channel from the shim.
#[cfg(feature = "sdr")]
fn run_sdr_probe(config: &std::path::Path) {
    let cfg = load_cfg(config);
    if cfg.sdr.channels.is_empty() {
        eprintln!("hamfeed-pipeline: no [[sdr.channels]] configured");
        std::process::exit(1);
    }
    for ch in &cfg.sdr.channels {
        let args = hamfeed_source::ProbeArgs {
            python: cfg.sdr.python.clone(),
            script: cfg.sdr.script.clone(),
            freq_hz: ch.freq_hz,
            gain: cfg.sdr.gain,
            rate_hz: cfg.sdr.rate_hz,
            bandwidth_hz: cfg.sdr.bandwidth_hz,
            squelch_db: cfg.sdr.squelch_db,
            antenna: cfg.sdr.antenna.clone(),
        };
        match hamfeed_source::sdr_probe(&args) {
            Ok(row) => println!(
                "{} {:.3} MHz floor={} peak={} snr={} gate={}",
                ch.name,
                row.freq_hz / 1e6,
                fmt_opt(row.floor_dbfs),
                fmt_opt(row.peak_dbfs),
                fmt_opt(row.snr_db),
                if row.gate_open { "OPEN" } else { "closed" }
            ),
            Err(e) => {
                eprintln!("hamfeed-pipeline: probe {} failed: {e:?}", ch.name);
                std::process::exit(1);
            }
        }
    }
}

#[cfg(not(feature = "sdr"))]
fn run_sdr_probe(_config: &std::path::Path) {
    eprintln!("hamfeed-pipeline: --sdr-probe needs the sdr feature (rebuild with --features sdr)");
    std::process::exit(2);
}

#[cfg(feature = "sdr")]
fn fmt_opt(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "-".into())
}

/// Live SDR capture: open the wanted channel, run it until the stream
/// ends (UI switch or child exit), reopen on the new channel. A dead
/// child costs a 1 s breath, never a hot loop; retune gaps are the
/// documented price of single-tuner hopping.
#[cfg(feature = "sdr")]
fn run_sdr_loop(pipe: &Pipeline, config: &std::path::Path, profile: &str) {
    loop {
        let cfg = load_cfg(config);
        let wanted = pipe
            .store()
            .sdr_channel()
            .unwrap_or(None)
            .filter(|n| cfg.sdr.channel(n).is_some())
            .or_else(|| cfg.sdr.active_channel().map(|c| c.name.clone()));
        let Some(name) = wanted else {
            eprintln!("hamfeed-pipeline: no [[sdr.channels]] configured");
            std::process::exit(1);
        };
        let ch = cfg.sdr.channel(&name).expect("channel validated above");
        // Manual UI tune wins over the preset when set and in range;
        // the watch below retunes on any later change either way.
        let freq_override = pipe
            .store()
            .sdr_freq()
            .unwrap_or(None)
            .filter(|f| hamfeed_source::FREQ_RANGE.contains(f));
        let mode_override = freq_override.and_then(|_| {
            pipe.store()
                .sdr_mode()
                .ok()
                .filter(|m| hamfeed_source::SUPPORTED_MODES.contains(&m.as_str()))
        });
        let gain_override = pipe
            .store()
            .sdr_gain()
            .unwrap_or(None)
            .filter(|g| hamfeed_source::GAIN_RANGE.contains(g));
        let (freq_hz, mode) = match (freq_override, mode_override.clone()) {
            (Some(f), Some(m)) => (f, m),
            (Some(f), None) => (f, "nbfm".into()),
            (None, _) => (ch.freq_hz, ch.mode.clone()),
        };
        let gain = gain_override.unwrap_or(cfg.sdr.gain);
        let params = hamfeed_source::SdrParams {
            python: cfg.sdr.python.clone(),
            script: cfg.sdr.script.clone(),
            db_path: cfg.storage.db_path.clone(),
            channel: name.clone(),
            freq_hz,
            mode: mode.clone(),
            freq_override,
            mode_override: mode_override.clone(),
            gain_override,
            gain,
            rate_hz: cfg.sdr.rate_hz,
            bandwidth_hz: cfg.sdr.bandwidth_hz,
            squelch_db: cfg.sdr.squelch_db,
            hang_s: cfg.sdr.hang_s,
            antenna: cfg.sdr.antenna.clone(),
        };
        let mut src = hamfeed_source::SdrSource::open(&params).unwrap_or_else(|e| {
            eprintln!("hamfeed-pipeline: sdr open failed: {e:?}");
            std::process::exit(1);
        });
        eprintln!(
            "hamfeed-pipeline: capturing SDR {name} ({:.3} MHz, {mode}, gain {gain}, profile {profile})",
            freq_hz / 1e6
        );
        match pipe.run_source(
            &mut src,
            cfg.hang_ms(profile),
            cfg.segment.max_s,
            cfg.vad.beep_split,
            cfg.vad.beep_min_ms,
        ) {
            Ok(n) => eprintln!("hamfeed-pipeline: sdr source ended ({n} segment(s)); reopening…"),
            Err(e) => eprintln!("hamfeed-pipeline: sdr run failed: {e:?}; reopening…"),
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

/// Publish tap PCM as raw little-endian i16 over a Unix socket (Slice 3
/// fix). One thread per listener; each starts at the current tail and
/// follows live. A dead or slow listener ends only its own thread —
/// capture never blocks on listeners.
/// Max backlog a listener replays after a stall. The frame loop blocks
/// during transcription (drain runs STT synchronously), so after a long
/// segment a follower can sit on seconds of stale audio: beyond this it
/// jumps to the live tail instead of replaying the past.
const LIVE_CATCHUP_MAX: u64 = 16_000 * 5;

/// Where a stalled reader resumes: its position when fresh, the live tail
/// when it fell further than [`LIVE_CATCHUP_MAX`] behind.
fn catch_up(current: u64, since: u64) -> u64 {
    if current.saturating_sub(since) > LIVE_CATCHUP_MAX {
        current
    } else {
        since
    }
}

fn serve_live_tap(tap: std::sync::Arc<hamfeed_pipeline::LiveTap>, sock: std::path::PathBuf) {
    use std::io::Write as _;
    use std::os::unix::net::UnixListener;
    let _ = std::fs::remove_file(&sock);
    let listener = match UnixListener::bind(&sock) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("hamfeed-pipeline: live relay unavailable ({e:?})");
            return;
        }
    };
    for conn in listener.incoming() {
        let Ok(mut stream) = conn else { continue };
        let tap = std::sync::Arc::clone(&tap);
        std::thread::spawn(move || {
            let mut since = tap.current_seq();
            loop {
                // Jump to live when a transcription stall left this
                // follower on stale audio; otherwise follow contiguously.
                since = catch_up(tap.current_seq(), since);
                let (now, pcm) = tap.read_since(since);
                since = now;
                if pcm.is_empty() {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                let mut raw = Vec::with_capacity(pcm.len() * 2);
                for s in &pcm {
                    raw.extend_from_slice(&s.to_le_bytes());
                }
                if stream.write_all(&raw).is_err() {
                    break;
                }
            }
        });
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_up_jumps_only_when_stale() {
        // Fresh follower: stays put, replays contiguously.
        assert_eq!(catch_up(100_000, 100_000), 100_000);
        assert_eq!(catch_up(100_000, 99_999), 99_999);
        // Exactly at the bound: still replays.
        assert_eq!(catch_up(80_000, 0), 0);
        // Past it (a transcription stall's worth): jumps to live.
        assert_eq!(catch_up(80_001, 0), 80_001);
        assert_eq!(catch_up(480_000, 0), 480_000);
    }
}
