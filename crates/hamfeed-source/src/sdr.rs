//! B210 capture source: a UHD shim child process (`sdr/sdr_rx.py`) streams
//! fixed 20 ms 16 kHz mono S16LE records on stdout; this module frames
//! them into [`PcmFrame`]s. Retune = reopen (the UI writes the wanted
//! channel to the settings table; the stream ends itself on change and
//! the pipeline loop reopens on the new channel).
//!
//! Behind the `sdr` feature: no radio, no UHD, no numpy in CI — tests
//! exercise framing and channel watching with in-memory readers and a
//! temp settings DB, plus a best-effort spawn test that skips honestly
//! when `python3` is absent.

use super::{AudioSource, PcmFrame, SAMPLE_RATE};
use anyhow::{Context, Result};
use std::io::Read;
use std::process::{Child, Command, Stdio};

/// 20 ms records: 320 samples of S16LE = 640 bytes. Fixed size keeps
/// framing trivial and restarts clean (no partial-record state).
pub const SDR_FRAME_SAMPLES: usize = 320;
pub const SDR_FRAME_BYTES: usize = SDR_FRAME_SAMPLES * 2;

/// How often the stream re-reads the wanted channel (frames).
const WATCH_EVERY_FRAMES: u64 = 25;

/// Settings-table key the UI writes and the stream watches.
pub const SDR_CHANNEL_KEY: &str = "sdr_channel";
/// Manual tune override (Hz, text) the UI frequency entry writes.
pub const SDR_FREQ_KEY: &str = "sdr_freq_hz";
/// Demod mode for a manual tune (`nbfm`/`am`).
pub const SDR_MODE_KEY: &str = "sdr_mode";
/// Demod modes the shim implements (mirrors `Demod.SUPPORTED_MODES`).
pub const SUPPORTED_MODES: &[&str] = &["nbfm", "am"];
/// Manual-tune range (Hz); mirrors the config channel validation
/// (B210/AD9361 tunes 70 MHz..=6 GHz — AM broadcast band is out).
pub const FREQ_RANGE: std::ops::RangeInclusive<f64> = 70e6..=6e9;

/// Open parameters, built from `[sdr]` config + the active channel.
#[derive(Debug, Clone)]
pub struct SdrParams {
    pub python: String,
    pub script: String,
    pub db_path: String,
    pub channel: String,
    pub freq_hz: f64,
    pub mode: String,
    /// Raw UI override keys (watch init only; the effective freq/mode
    /// above drive the shim). Needed so clearing an override retunes.
    pub freq_override: Option<f64>,
    pub mode_override: Option<String>,
    pub gain: f64,
    pub rate_hz: f64,
    pub bandwidth_hz: f64,
    pub squelch_db: f64,
    pub hang_s: f64,
    pub antenna: String,
}

impl SdrParams {
    fn argv(&self) -> (String, Vec<String>) {
        (
            self.python.clone(),
            vec![
                self.script.clone(),
                "--freq".into(),
                self.freq_hz.to_string(),
                "--mode".into(),
                self.mode.clone(),
                "--gain".into(),
                self.gain.to_string(),
                "--rate".into(),
                self.rate_hz.to_string(),
                "--bw".into(),
                self.bandwidth_hz.to_string(),
                "--squelch-db".into(),
                self.squelch_db.to_string(),
                "--hang-s".into(),
                self.hang_s.to_string(),
                "--antenna".into(),
                self.antenna.clone(),
            ],
        )
    }
}

/// One `--probe` run: floor/peak/SNR at a frequency, then exit.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeRow {
    pub freq_hz: f64,
    pub floor_dbfs: Option<f32>,
    pub peak_dbfs: Option<f32>,
    pub snr_db: Option<f32>,
    pub gate_open: bool,
}

/// Probe arguments: station constants plus the frequency to measure.
#[derive(Debug, Clone)]
pub struct ProbeArgs {
    pub python: String,
    pub script: String,
    pub freq_hz: f64,
    pub gain: f64,
    pub rate_hz: f64,
    pub bandwidth_hz: f64,
    pub squelch_db: f64,
    pub antenna: String,
}

/// Run the shim in probe mode and parse its single JSON line.
/// Mirrors `--list-devices`: one row per configured frequency.
pub fn sdr_probe(args: &ProbeArgs) -> Result<ProbeRow> {
    let out = Command::new(&args.python)
        .args([
            args.script.as_str(),
            "--freq",
            &args.freq_hz.to_string(),
            "--gain",
            &args.gain.to_string(),
            "--rate",
            &args.rate_hz.to_string(),
            "--bw",
            &args.bandwidth_hz.to_string(),
            "--squelch-db",
            &args.squelch_db.to_string(),
            "--antenna",
            args.antenna.as_str(),
            "--probe",
        ])
        .output()
        .with_context(|| format!("cannot run sdr probe {} {}", args.python, args.script))?;
    if !out.status.success() {
        anyhow::bail!(
            "sdr probe failed at {}: {}",
            args.freq_hz,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value =
        serde_json::from_str(line.trim()).context("sdr probe printed no JSON row")?;
    Ok(ProbeRow {
        freq_hz: v
            .get("freq_hz")
            .and_then(|x| x.as_f64())
            .unwrap_or(args.freq_hz),
        floor_dbfs: v
            .get("floor_dbfs")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32),
        peak_dbfs: v
            .get("peak_dbfs")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32),
        snr_db: v.get("snr_db").and_then(|x| x.as_f64()).map(|x| x as f32),
        gate_open: v
            .get("gate_open")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
    })
}

/// Watches the wanted channel: the opened name until the settings
/// table says otherwise. Pure + unit-tested; the stream consults it.
/// Wanted tune as raw settings: preset channel plus the manual
/// frequency/mode override keys (the UI frequency entry), each
/// `None` when unset or garbage. Equality against the opened triple
/// drives retune — crucially, clearing an override (back to preset)
/// differs from the opened triple, so it retunes instead of
/// sticking on the old frequency. The pipeline loop resolves these
/// into effective freq/mode; the watch only compares.
#[derive(Debug, Clone, PartialEq)]
pub struct Tune {
    pub channel: String,
    pub freq_override: Option<f64>,
    pub mode_override: Option<String>,
}

pub struct ChannelWatch {
    db_path: String,
    opened: Tune,
}

impl ChannelWatch {
    pub fn new(
        db_path: &str,
        channel: &str,
        freq_override: Option<f64>,
        mode_override: Option<String>,
    ) -> Self {
        Self {
            db_path: db_path.into(),
            opened: Tune {
                channel: channel.into(),
                freq_override,
                mode_override,
            },
        }
    }

    /// Current wanted tune from the settings keys. A missing DB
    /// degrades to "stay" — capture never dies because the UI store
    /// is briefly locked.
    pub fn wanted(&self) -> Tune {
        let channel = setting_from_db(&self.db_path, SDR_CHANNEL_KEY)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| self.opened.channel.clone());
        // Same range rule as the pipeline loop and the API: an
        // out-of-range override (e.g. below the B210's 70 MHz floor)
        // reads as unset everywhere, so all three agree and the
        // stream never flaps on a stale key.
        let freq_override = setting_from_db(&self.db_path, SDR_FREQ_KEY)
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|f| f.is_finite() && FREQ_RANGE.contains(f));
        // Mode override only counts with a frequency override (the
        // API writes them together; a stray mode key alone retunes
        // nothing, which also avoids a reopen loop).
        let mode_override = freq_override.and(
            setting_from_db(&self.db_path, SDR_MODE_KEY)
                .filter(|s| SUPPORTED_MODES.contains(&s.as_str())),
        );
        Tune {
            channel,
            freq_override,
            mode_override,
        }
    }

    /// True when the operator retuned away from the opened signal.
    pub fn changed(&self) -> bool {
        self.wanted() != self.opened
    }

    /// Channel this instance was opened on (retune target bookkeeping).
    pub fn channel(&self) -> &str {
        &self.opened.channel
    }
}

/// Read one settings value straight from the table (same
/// `settings(key, value)` shape the store owns; this crate stays
/// independent of hamfeed-store).
fn setting_from_db(db_path: &str, key: &str) -> Option<String> {
    let conn =
        rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .ok()?;
    let mut stmt = conn
        .prepare("SELECT value FROM settings WHERE key = ?")
        .ok()?;
    let mut rows = stmt.query([key]).ok()?;
    rows.next().ok()??.get(0).ok()
}

/// Read fixed-size S16LE records into frames. Short reads and EOF end
/// the stream (child exited or retuning); I/O errors end it too — the
/// pipeline loop reopens and logs formative. Test-only seam: the live
/// [`AudioSource::stream`] below inlines the same logic with the
/// channel watch (closures can't share the helper's stop flag).
#[cfg(test)]
fn read_frames<R: Read>(mut r: R, mut stop: impl FnMut() -> bool) -> Vec<PcmFrame> {
    let mut frames = Vec::new();
    let mut buf = [0u8; SDR_FRAME_BYTES];
    loop {
        if stop() {
            break;
        }
        match r.read_exact(&mut buf) {
            Ok(()) => {
                let (chunks, _) = buf.as_chunks::<2>();
                let samples: Vec<i16> = chunks.iter().map(|c| i16::from_le_bytes(*c)).collect();
                frames.push(PcmFrame { samples });
            }
            Err(_) => break,
        }
    }
    frames
}

/// Decode one fixed record (shared shape with the live stream above).
fn decode_record(buf: &[u8; SDR_FRAME_BYTES]) -> PcmFrame {
    let (chunks, _) = buf.as_chunks::<2>();
    let samples: Vec<i16> = chunks.iter().map(|c| i16::from_le_bytes(*c)).collect();
    PcmFrame { samples }
}

/// B210 capture source: child shim on stdout, self-ending stream on
/// channel switch or child exit.
pub struct SdrSource {
    child: Child,
    out: std::process::ChildStdout,
    watch: ChannelWatch,
    frames_since_poll: u64,
}

impl SdrSource {
    /// Spawn the shim for `params`. The child inherits no stdin; its
    /// stderr is drained to the log so a chatty UHD never blocks it.
    pub fn open(params: &SdrParams) -> Result<Self> {
        let (prog, args) = params.argv();
        let mut child = Command::new(&prog)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("cannot spawn sdr shim {prog} {}", params.script))?;
        if let Some(err) = child.stderr.take() {
            std::thread::spawn(move || {
                use std::io::BufRead;
                for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
                    eprintln!("sdr shim: {line}");
                }
            });
        }
        let out = child.stdout.take().context("sdr shim stdout not piped")?;
        Ok(Self {
            child,
            out,
            watch: ChannelWatch::new(
                &params.db_path,
                &params.channel,
                params.freq_override,
                params.mode_override.clone(),
            ),
            frames_since_poll: 0,
        })
    }

    /// Channel this instance was opened on (retune target bookkeeping).
    pub fn channel(&self) -> &str {
        self.watch.channel()
    }
}

impl Drop for SdrSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl AudioSource for SdrSource {
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Blocking frame iterator. Ends on child exit/error or when the
    /// settings table names a different channel (the pipeline loop
    /// reopens on the new one). Polls the table every
    /// [`WATCH_EVERY_FRAMES`] frames (~0.5 s at 20 ms records).
    fn stream(&mut self) -> Box<dyn Iterator<Item = PcmFrame> + '_> {
        let mut buf = [0u8; SDR_FRAME_BYTES];
        let it = std::iter::from_fn(move || {
            if self.frames_since_poll >= WATCH_EVERY_FRAMES {
                self.frames_since_poll = 0;
                if self.watch.changed() {
                    return None;
                }
            }
            match self.out.read_exact(&mut buf) {
                Ok(()) => {
                    self.frames_since_poll += 1;
                    Some(decode_record(&buf))
                }
                Err(_) => None,
            }
        });
        Box::new(it)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Temp settings DB with wanted channel/freq/mode preset.
    fn temp_db(wanted: Option<&str>) -> String {
        temp_db_full(wanted, None, None)
    }

    fn temp_db_full(wanted: Option<&str>, freq: Option<&str>, mode: Option<&str>) -> String {
        let path = std::env::temp_dir().join(format!(
            "hamfeed-sdr-test-{}-{}.db",
            std::process::id(),
            rand_suffix()
        ));
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        for (k, v) in [
            ("sdr_channel", wanted),
            ("sdr_freq_hz", freq),
            ("sdr_mode", mode),
        ] {
            if let Some(val) = v {
                conn.execute("INSERT INTO settings(key, value) VALUES (?, ?)", [k, val])
                    .unwrap();
            }
        }
        path.to_string_lossy().into_owned()
    }

    fn rand_suffix() -> String {
        // No rand crate here: nanos are unique enough per test process.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
            .to_string()
    }

    #[test]
    fn frames_split_exact_records() {
        // Two 20 ms records of nonzero S16LE survive the round trip;
        // a trailing partial record ends the stream (retune-clean).
        let mut raw = Vec::new();
        for i in 0..(SDR_FRAME_SAMPLES * 2) {
            raw.extend_from_slice(&(i as i16).to_le_bytes());
        }
        raw.extend_from_slice(&[9u8; 100]); // runt tail
        let got = read_frames(Cursor::new(raw), || false);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].samples.len(), SDR_FRAME_SAMPLES);
        assert_eq!(got[0].samples[0], 0);
        assert_eq!(got[1].samples[0], SDR_FRAME_SAMPLES as i16);
    }

    #[test]
    fn stop_ends_stream_early() {
        let raw = vec![0u8; SDR_FRAME_BYTES * 4];
        let mut calls = 0;
        let got = read_frames(Cursor::new(raw), || {
            calls += 1;
            calls > 2
        });
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn watch_follows_settings_or_stays() {
        // Settings naming another channel: switched.
        let db = temp_db(Some("marine"));
        let w = ChannelWatch::new(&db, "2m VE2", None, None);
        assert_eq!(w.wanted().channel, "marine");
        assert!(w.changed());
        // Empty value: stay on the opened channel.
        let db = temp_db(Some(""));
        let w = ChannelWatch::new(&db, "2m VE2", None, None);
        assert_eq!(w.wanted().channel, "2m VE2");
        assert!(!w.changed());
        // Missing key: stay.
        let db = temp_db(None);
        let w = ChannelWatch::new(&db, "2m VE2", None, None);
        assert!(!w.changed());
        // Missing DB file: stay (capture never dies on a locked store).
        let w = ChannelWatch::new("/nonexistent/hamfeed-test.db", "2m VE2", None, None);
        assert!(!w.changed());
    }

    #[test]
    fn watch_follows_freq_and_mode_tune() {
        // Manual UI tune (freq + mode): retune fires; same values stay.
        let db = temp_db_full(None, Some("161775000"), Some("am"));
        let w = ChannelWatch::new(&db, "marine", None, None);
        let t = w.wanted();
        assert_eq!(t.freq_override, Some(161775000.0));
        assert_eq!(t.mode_override.as_deref(), Some("am"));
        assert!(w.changed());
        let w2 = ChannelWatch::new(&db, "marine", Some(161775000.0), Some("am".into()));
        assert!(!w2.changed());
        // Garbage freq/mode reads as unset (stay).
        let db = temp_db_full(None, Some("junk"), Some("ssb"));
        let w3 = ChannelWatch::new(&db, "marine", None, None);
        assert_eq!(w3.wanted().freq_override, None);
        assert_eq!(w3.wanted().mode_override, None);
        assert!(!w3.changed());
        // Out-of-range freq (below the B210 floor) also reads as
        // unset, matching the loop/API rule — no flap on stale keys.
        let db = temp_db_full(None, Some("1670000"), Some("am"));
        let w4 = ChannelWatch::new(&db, "marine", None, None);
        assert_eq!(w4.wanted().freq_override, None);
        assert!(!w4.changed());
    }

    #[test]
    fn watch_clearing_override_retunes() {
        // Opened on a manual tune, operator picks a preset (override
        // keys deleted): wanted differs from opened, so the loop
        // reopens instead of sticking on the old frequency.
        let db = temp_db(Some("marine"));
        let w = ChannelWatch::new(&db, "marine", Some(161775000.0), Some("nbfm".into()));
        assert_eq!(w.wanted().freq_override, None);
        assert!(w.changed());
    }

    #[test]
    fn probe_parses_json_row() {
        // Probe JSON contract without radio hardware: a stub script
        // printing one row must parse (spawned only when python3 exists).
        let py = match which_python() {
            Some(p) => p,
            None => {
                eprintln!("SKIP probe_parses_json_row: no python3");
                return;
            }
        };
        let dir = std::env::temp_dir();
        let stub = dir.join(format!("hamfeed-probe-stub-{}.py", std::process::id()));
        std::fs::write(
            &stub,
            "import json,sys\nprint(json.dumps({'freq_hz':145110000.0,'floor_dbfs':-75.0,'peak_dbfs':-60.0,'snr_db':15.0,'gate_open':True}))\n",
        )
        .unwrap();
        let row = sdr_probe(&ProbeArgs {
            python: py,
            script: stub.to_string_lossy().into_owned(),
            freq_hz: 145110000.0,
            gain: 20.0,
            rate_hz: 250000.0,
            bandwidth_hz: 200000.0,
            squelch_db: 13.0,
            antenna: "RX2".into(),
        })
        .expect("stub probe must parse");
        assert_eq!(row.freq_hz, 145110000.0);
        assert_eq!(row.snr_db, Some(15.0));
        assert!(row.gate_open);
        let _ = std::fs::remove_file(&stub);
    }

    #[test]
    fn spawn_streams_child_records() {
        // Full open() → stream() → drop() path against a stdlib-only
        // child (no numpy/UHD): 5 sine records then EOF must arrive
        // intact with audio values, proving framing over a real pipe.
        let py = match which_python() {
            Some(p) => p,
            None => {
                eprintln!("SKIP spawn_streams_child_records: no python3");
                return;
            }
        };
        let dir = std::env::temp_dir();
        let script = dir.join(format!("hamfeed-stream-stub-{}.py", std::process::id()));
        std::fs::write(
            &script,
            "import math,sys\nout=sys.stdout.buffer\nfor i in range(5*320):\n out.write(int(10000*math.sin(i/10)).to_bytes(2,'little',signed=True))\n",
        )
        .unwrap();
        let db = temp_db(None);
        // The stub ignores SDR flags (reads none) — open() passes them
        // positionally, python ignores the extras.
        let params = SdrParams {
            python: py,
            script: script.to_string_lossy().into_owned(),
            db_path: db,
            channel: "stub".into(),
            freq_hz: 145110000.0,
            mode: "nbfm".into(),
            freq_override: None,
            mode_override: None,
            gain: 20.0,
            rate_hz: 250000.0,
            bandwidth_hz: 200000.0,
            squelch_db: 13.0,
            hang_s: 1.5,
            antenna: "RX2".into(),
        };
        // The stub ignores SDR flags — but SdrSource::open passes them
        // positionally, so wrap: python -u <stub> extra args ignored.
        let mut src = SdrSource::open(&params).expect("open stub");
        let got: Vec<PcmFrame> = src.stream().collect();
        assert_eq!(got.len(), 5);
        assert!(got.iter().all(|f| f.samples.len() == SDR_FRAME_SAMPLES));
        assert!(got.iter().flat_map(|f| f.samples.iter()).any(|&s| s != 0));
        let _ = std::fs::remove_file(&script);
    }

    fn which_python() -> Option<String> {
        for p in ["python3", "/usr/bin/python3"] {
            if Command::new(p)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
            {
                return Some(p.into());
            }
        }
        None
    }
}
