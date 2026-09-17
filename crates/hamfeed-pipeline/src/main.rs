//! hamfeed-pipeline binary: live capture → transcribe → store.

use std::path::PathBuf;

use hamfeed_pipeline::Pipeline;

fn usage() -> ! {
    eprintln!("usage: hamfeed-pipeline [--config PATH] [--once] [--fake SECS] [--simplex] [--list-devices]");
    std::process::exit(2);
}

fn main() {
    let mut config = PathBuf::from("hamfeed.toml");
    let mut once = false;
    let mut fake_secs: Option<u64> = None;
    let mut profile = "repeater";
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config = PathBuf::from(args.next().unwrap_or_else(|| usage())),
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

    let mut pipe = Pipeline::open(&config).unwrap_or_else(|e| {
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

/// Publish tap PCM as raw little-endian i16 over a Unix socket (Slice 3
/// fix). One thread per listener; each starts at the current tail and
/// follows live. A dead or slow listener ends only its own thread —
/// capture never blocks on listeners.
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
