//! hamfeed-source: audio input trait + impls (T4).
//!
//! Shape (plan): `trait AudioSource` yields 16 kHz mono S16 frames. One v1
//! impl (`MicSource` via cpal, behind the `capture` feature) so a future
//! `SdrSource` can slot in without changing ingest (ADR-3). Tests always use
//! `FakeSource`: no mic hardware in tests.

/// Canonical internal rate: 16 kHz mono S16 (R1).
pub const SAMPLE_RATE: u32 = 16_000;

/// One chunk of mono S16 audio at [`SAMPLE_RATE`] (varying length).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmFrame {
    pub samples: Vec<i16>,
}

impl PcmFrame {
    pub fn silence(frames: usize) -> Self {
        Self {
            samples: vec![0; frames],
        }
    }

    /// Full-scale-ish sine at `freq_hz` (test/simulation helper).
    pub fn tone(freq_hz: f32, frames: usize, amplitude: i16) -> Self {
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / SAMPLE_RATE as f32;
            let v = (2.0 * std::f32::consts::PI * freq_hz * t).sin();
            samples.push((v * amplitude as f32) as i16);
        }
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn duration_ms(&self) -> u64 {
        self.samples.len() as u64 * 1000 / SAMPLE_RATE as u64
    }
}

/// Anything ingest can consume: an iterator of 16 kHz mono S16 frames.
pub trait AudioSource {
    /// Native rate of this source; v1 sources always return [`SAMPLE_RATE`].
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Blocking frame iterator. Ends when the source ends (fake) or fails.
    fn stream(&mut self) -> Box<dyn Iterator<Item = PcmFrame> + '_>;
}

/// Deterministic scripted source for tests and simulation.
#[derive(Debug, Clone)]
pub struct FakeSource {
    frames: Vec<PcmFrame>,
    repeat: bool,
    pos: usize,
}

impl FakeSource {
    /// Play `frames` once, then end the stream.
    pub fn once(frames: Vec<PcmFrame>) -> Self {
        Self {
            frames,
            repeat: false,
            pos: 0,
        }
    }

    /// Loop `frames` forever.
    pub fn repeat(frames: Vec<PcmFrame>) -> Self {
        Self {
            frames,
            repeat: true,
            pos: 0,
        }
    }

    /// `secs` seconds of silence in `frame_len`-sample frames.
    pub fn silence(secs: u64, frame_len: usize) -> Self {
        let n = (secs as usize * SAMPLE_RATE as usize).div_ceil(frame_len);
        Self::once(vec![PcmFrame::silence(frame_len); n])
    }
}

impl AudioSource for FakeSource {
    fn stream(&mut self) -> Box<dyn Iterator<Item = PcmFrame> + '_> {
        let it = std::iter::from_fn(move || {
            if self.pos >= self.frames.len() {
                if self.repeat && !self.frames.is_empty() {
                    self.pos = 0;
                } else {
                    return None;
                }
            }
            let f = self.frames[self.pos].clone();
            self.pos += 1;
            Some(f)
        });
        Box::new(it)
    }
}

#[cfg(feature = "capture")]
mod mic {
    use super::*;
    use anyhow::{Context, Result};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::mpsc;

    /// Names of all host input devices (for `--list-devices` and errors).
    pub fn list_input_devices() -> Vec<String> {
        let host = cpal::default_host();
        host.input_devices()
            .map(|devs| devs.filter_map(|d| d.name().ok()).collect())
            .unwrap_or_default()
    }

    /// cpal mic input at 16 kHz mono S16.
    pub struct MicSource {
        rx: mpsc::Receiver<PcmFrame>,
        // Held so the stream keeps flowing while `self` lives.
        _stream: cpal::Stream,
        device_name: String,
    }

    impl MicSource {
        /// Open `wanted` by name, or the default input when `wanted` is
        /// `"default"`. Requests 16 kHz mono I16; errors list what the
        /// device actually supports.
        pub fn open(wanted: &str) -> Result<Self> {
            let host = cpal::default_host();
            let device = if wanted == "default" {
                host.default_input_device()
                    .context("no default input device on this host")?
            } else {
                host.input_devices()
                    .context("cannot list input devices")?
                    .find(|d| d.name().map(|n| n == wanted).unwrap_or(false))
                    .with_context(|| {
                        format!(
                            "input device {wanted:?} not found (available: {})",
                            list_names(&host)
                        )
                    })?
            };
            let device_name = device.name().unwrap_or_else(|_| wanted.into());
            let supported = device.supported_input_configs().with_context(|| {
                format!(
                    "cannot query supported input configs on {device_name:?} \
                         (available: {})",
                    list_names(&host)
                )
            })?;
            let mut supported_note = Vec::new();
            let mut chosen: Option<cpal::SupportedStreamConfig> = None;
            for cfg in supported {
                supported_note.push(format!(
                    "{}ch {}Hz {:?}..{:?}",
                    cfg.channels(),
                    cfg.min_sample_rate().0,
                    cfg.sample_format(),
                    cfg.max_sample_rate().0
                ));
                if cfg.channels() == 1
                    && cfg.sample_format() == cpal::SampleFormat::I16
                    && cfg.min_sample_rate().0 <= SAMPLE_RATE
                    && SAMPLE_RATE <= cfg.max_sample_rate().0
                {
                    chosen = Some(cfg.with_sample_rate(cpal::SampleRate(SAMPLE_RATE)));
                }
            }
            let chosen = chosen.with_context(|| {
                format!(
                    "device {device_name:?} offers no 16kHz mono I16 mode (offers: {})",
                    supported_note.join(" | ")
                )
            })?;
            let (tx, rx) = mpsc::channel();
            let stream = device
                .build_input_stream(
                    &chosen.config(),
                    move |data: &[i16], _| {
                        let _ = tx.send(PcmFrame {
                            samples: data.to_vec(),
                        });
                    },
                    |err| tracing::warn!(%err, "mic stream error"),
                    None,
                )
                .context("cannot build mic input stream")?;
            stream.play().context("cannot start mic input stream")?;
            Ok(Self {
                rx,
                _stream: stream,
                device_name,
            })
        }

        pub fn device_name(&self) -> &str {
            &self.device_name
        }
    }

    fn list_names(host: &cpal::Host) -> String {
        match host.input_devices() {
            Ok(devs) => {
                let names: Vec<String> = devs.filter_map(|d| d.name().ok()).collect();
                if names.is_empty() {
                    "(none)".into()
                } else {
                    names.join(", ")
                }
            }
            Err(e) => format!("(list failed: {e})"),
        }
    }

    impl AudioSource for MicSource {
        fn stream(&mut self) -> Box<dyn Iterator<Item = PcmFrame> + '_> {
            Box::new(std::iter::from_fn(move || self.rx.recv().ok()))
        }
    }
}

#[cfg(feature = "capture")]
pub use mic::{list_input_devices, MicSource};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_frames_rate() {
        // 16 kHz mono S16 shape: every frame carries i16 samples whose
        // total duration matches the scripted length.
        let mut src = FakeSource::once(vec![
            PcmFrame::tone(440.0, 1600, 10_000),
            PcmFrame::silence(1600),
        ]);
        assert_eq!(src.sample_rate(), 16_000);
        let got: Vec<PcmFrame> = src.stream().collect();
        assert_eq!(got.len(), 2);
        for f in &got {
            assert_eq!(f.len(), 1600);
            assert_eq!(f.duration_ms(), 100);
        }
        assert!(got[0].samples.iter().any(|&s| s != 0));
        assert!(got[1].samples.iter().all(|&s| s == 0));
    }
}
