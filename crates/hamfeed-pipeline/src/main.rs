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
