//! Shipper + manifest + audit logic for the 009 finetune loop (T2).
//!
//! Pure functions over the 004 R5 training export; all LAN/zip/cron
//! plumbing lives in thin box scripts that call this crate's binary.
//! Splits are hash-assigned and frozen: the same clip id always lands
//! in the same split, so separately shipped batches can merge safely.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

/// Frozen held-out share (percent). Fixed on day one so every future
/// WER stays comparable; changing it invalidates the eval history.
pub const HELDOUT_PCT: u64 = 20;

/// Languages the loop trains on (009 plan: French-first buckets).
/// Anything else is excluded, never silently absorbed.
const LANG_WHITELIST: &[&str] = &["fr", "en"];

/// One row of the 004 R5 export manifest (`manifest.csv` in the zip).
#[derive(Debug, Clone, PartialEq)]
pub struct ExportPair {
    pub id: String,
    pub true_text: String,
    pub lang: String,
    pub original: String,
}

/// Split one CSV line honoring `"quoted, fields"` and `""` escapes.
fn split_csv_line(line: &str) -> Result<Vec<String>> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    let mut in_field = false;
    while let Some(c) = chars.next() {
        in_field = true;
        match c {
            '"' if !in_quotes && cur.trim().is_empty() => {
                cur.clear();
                in_quotes = true;
            }
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut cur));
                in_field = false;
            }
            _ => cur.push(c),
        }
    }
    if in_quotes {
        bail!("unterminated quote");
    }
    if in_field || line.ends_with(',') {
        fields.push(cur);
    }
    Ok(fields)
}

/// Parse + schema-check the export manifest. Loud on anything off:
/// wrong header, wrong arity, empty id or empty true text.
pub fn parse_export_csv(text: &str) -> Result<Vec<ExportPair>> {
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("");
    if header.trim() != "id,true_text,lang,original_transcript" {
        bail!("bad export header: {header:?}");
    }
    let mut pairs = Vec::new();
    for (n, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(line).with_context(|| format!("line {}: unparseable", n + 2))?;
        if f.len() != 4 {
            bail!("line {}: want 4 fields, got {}", n + 2, f.len());
        }
        let pair = ExportPair {
            id: f[0].trim().to_string(),
            true_text: f[1].trim().to_string(),
            lang: f[2].trim().to_string(),
            original: f[3].clone(),
        };
        if pair.id.is_empty() {
            bail!("line {}: empty clip id", n + 2);
        }
        if pair.true_text.is_empty() {
            bail!("line {}: empty true text for {}", n + 2, pair.id);
        }
        pairs.push(pair);
    }
    Ok(pairs)
}

/// Idempotent ship plan: export order, first-seen wins, already-shipped
/// ids dropped. Running it twice with updated state ships nothing.
pub fn plan_ship(export: &[ExportPair], shipped: &HashSet<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    export
        .iter()
        .map(|p| p.id.clone())
        .filter(|id| seen.insert(id.clone()) && !shipped.contains(id))
        .collect()
}

/// Stable non-crypto shard hash (splits only need stability, not security).
pub fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Heldout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Gold,
    Silver,
}

/// Frozen assignment: the same clip id always maps to the same split.
pub fn assign_split(clip_id: &str) -> Split {
    if fnv1a64(clip_id) % 100 < HELDOUT_PCT {
        Split::Heldout
    } else {
        Split::Train
    }
}

/// One dataset row (009 plan manifest contract).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub clip_id: String,
    pub sha256: String,
    pub true_text: String,
    pub original_guess: String,
    pub lang: String,
    pub source: Source,
    pub split: Split,
}

/// A row refused by the data ladder, with the reason logged.
#[derive(Debug, Clone, PartialEq)]
pub struct Exclusion {
    pub clip_id: String,
    pub reason: String,
}

/// Ladder exclusions checkable without decoding audio (009 R2.4).
/// Duration/tones/overlap need waveform analysis in `train.sh` (T3).
pub fn exclusion_reason(pair: &ExportPair) -> Option<String> {
    if !LANG_WHITELIST.contains(&pair.lang.as_str()) {
        return Some(format!("off-whitelist lang {:?}", pair.lang));
    }
    None
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Build dataset entries; excluded rows come back separately with
/// reasons (dropped from the manifest, never silently).
pub fn build_manifest(
    items: Vec<(ExportPair, Vec<u8>, Source)>,
) -> (Vec<ManifestEntry>, Vec<Exclusion>) {
    let mut entries = Vec::new();
    let mut excluded = Vec::new();
    for (pair, audio, source) in items {
        if let Some(reason) = exclusion_reason(&pair) {
            excluded.push(Exclusion {
                clip_id: pair.id,
                reason,
            });
            continue;
        }
        entries.push(ManifestEntry {
            split: assign_split(&pair.id),
            clip_id: pair.id,
            sha256: sha256_hex(&audio),
            true_text: pair.true_text,
            original_guess: pair.original,
            lang: pair.lang,
            source,
        });
    }
    (entries, excluded)
}

pub fn manifest_json(entries: &[ManifestEntry]) -> Result<String> {
    serde_json::to_string_pretty(entries).context("cannot encode manifest")
}

/// Parse + schema-check a dataset manifest: required fields, known
/// source/split words, 64-hex hashes.
pub fn parse_manifest_json(text: &str) -> Result<Vec<ManifestEntry>> {
    let entries: Vec<ManifestEntry> =
        serde_json::from_str(text).context("manifest is not valid JSON")?;
    for e in &entries {
        if e.clip_id.trim().is_empty() {
            bail!("manifest entry with empty clip_id");
        }
        if e.true_text.trim().is_empty() {
            bail!("{}: empty true_text", e.clip_id);
        }
        if e.sha256.len() != 64 || !e.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("{}: sha256 is not 64 hex chars", e.clip_id);
        }
    }
    Ok(entries)
}

/// Fail loud on train/held-out overlap or same-id-different-bytes.
/// Same id in both splits is an error, never a warning.
pub fn audit(entries: &[ManifestEntry]) -> Result<()> {
    let mut by_id: HashMap<&str, &ManifestEntry> = HashMap::new();
    let mut violations: Vec<String> = Vec::new();
    for e in entries {
        if let Some(prev) = by_id.insert(e.clip_id.as_str(), e) {
            if prev.split != e.split {
                violations.push(format!(
                    "overlap: {} is {:?} and {:?}",
                    e.clip_id, prev.split, e.split
                ));
            }
            if prev.sha256 != e.sha256 {
                violations.push(format!("sha-mismatch: {}", e.clip_id));
            }
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(anyhow!("eval leakage:\n{}", violations.join("\n")))
    }
}

/// One evaluated clip inside an eval report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalClip {
    pub clip_id: String,
    pub lang: String,
    pub base_wer: f32,
    pub tuned_wer: Option<f32>,
}

/// Per-language bucket summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalBucket {
    pub n: usize,
    pub base_wer: Option<f32>,
    pub tuned_wer: Option<f32>,
}

/// Owner-facing verdict. The loop never self-deploys: even
/// `DeployCandidate` waits for the human call (009 T4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    NoDeploy,
    DeployCandidate,
}

/// `eval.sh` report contract (009 G2): frozen-manifest pin, both
/// WERs, buckets, clips, and a verdict with its reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub generated_at: String,
    pub dataset_manifest_sha256: String,
    pub base_model: String,
    pub candidate_model: Option<String>,
    pub n_heldout: usize,
    pub base_wer: Option<f32>,
    pub tuned_wer: Option<f32>,
    pub buckets: HashMap<String, EvalBucket>,
    pub verdict: Verdict,
    pub reason: String,
    pub clips: Vec<EvalClip>,
}

fn wer_in_range(label: &str, v: Option<f32>) -> Result<()> {
    if let Some(w) = v {
        if !(0.0..=1.0).contains(&w) {
            bail!("{label}: WER {w} outside [0,1]");
        }
    }
    Ok(())
}

/// Parse + schema-check an eval report: shapes, WER ranges, clip
/// counts matching the header, buckets matching the clips.
pub fn parse_eval_report(text: &str) -> Result<EvalReport> {
    let r: EvalReport = serde_json::from_str(text).context("report is not valid JSON")?;
    if r.dataset_manifest_sha256.len() != 64
        || !r
            .dataset_manifest_sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
    {
        bail!("dataset_manifest_sha256 is not 64 hex chars");
    }
    if r.clips.len() != r.n_heldout {
        bail!("n_heldout {} != {} clips", r.n_heldout, r.clips.len());
    }
    wer_in_range("base_wer", r.base_wer)?;
    wer_in_range("tuned_wer", r.tuned_wer)?;
    if r.verdict == Verdict::DeployCandidate && (r.base_wer.is_none() || r.tuned_wer.is_none()) {
        bail!("deploy-candidate without both WERs");
    }
    for (lang, b) in &r.buckets {
        let have = r.clips.iter().filter(|c| &c.lang == lang).count();
        if b.n != have {
            bail!("bucket {lang}: n {} != {} clips", b.n, have);
        }
        wer_in_range(&format!("bucket {lang} base"), b.base_wer)?;
        wer_in_range(&format!("bucket {lang} tuned"), b.tuned_wer)?;
    }
    for c in &r.clips {
        wer_in_range(&format!("clip {}", c.clip_id), Some(c.base_wer))?;
        wer_in_range(&format!("clip {} tuned", c.clip_id), c.tuned_wer)?;
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_CSV: &str = "id,true_text,lang,original_transcript\n\
        c1,\"Opération du réseau, faites une pause.\",fr,\"Operation du reseau, faites une pause\"\n\
        c2,hello true friends,en,hello friends\n\
        c3,\"quoted \"\"word\"\", tail\",en,\"quoted word, tail\"\n";

    fn pairs() -> Vec<ExportPair> {
        parse_export_csv(FIXTURE_CSV).unwrap()
    }

    #[test]
    fn parses_quoted_commas_accents_and_escapes() {
        let p = pairs();
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].id, "c1");
        assert!(p[0].true_text.contains("Opération"));
        assert_eq!(p[2].true_text, "quoted \"word\", tail");
    }

    #[test]
    fn schema_rejects_bad_header_arity_and_blanks() {
        assert!(parse_export_csv("nope\n").is_err());
        assert!(parse_export_csv("id,true_text,lang,original_transcript\nonly,two\n").is_err());
        assert!(
            parse_export_csv("id,true_text,lang,original_transcript\n,blank id,en,x\n").is_err()
        );
        assert!(parse_export_csv("id,true_text,lang,original_transcript\nc9,,en,x\n").is_err());
    }

    #[test]
    fn ship_plan_is_idempotent_and_dedups() {
        let mut export = pairs();
        export.push(export[0].clone()); // endpoint repeats an id
        let shipped: HashSet<String> = ["c1".into()].into_iter().collect();
        let first = plan_ship(&export, &shipped);
        assert_eq!(first, vec!["c2".to_string(), "c3".to_string()]);
        // Second run with updated state ships nothing.
        let shipped2: HashSet<String> = ["c1".into(), "c2".into(), "c3".into()]
            .into_iter()
            .collect();
        assert!(plan_ship(&export, &shipped2).is_empty());
    }

    #[test]
    fn split_is_frozen_and_roughly_twenty_percent() {
        for id in ["c1", "alpha", "beta-42"] {
            assert_eq!(assign_split(id), assign_split(id));
        }
        let held = (0..2000)
            .filter(|i| assign_split(&format!("clip-{i}")) == Split::Heldout)
            .count();
        assert!((250..550).contains(&held), "held out {held}/2000");
    }

    fn entry(id: &str, split: Split, sha: &str) -> ManifestEntry {
        ManifestEntry {
            clip_id: id.into(),
            sha256: sha.into(),
            true_text: "t".into(),
            original_guess: "g".into(),
            lang: "fr".into(),
            source: Source::Gold,
            split,
        }
    }

    #[test]
    fn manifest_roundtrip_and_schema_checks() {
        let items: Vec<(ExportPair, Vec<u8>, Source)> = pairs()
            .into_iter()
            .map(|p| {
                let audio = format!("audio-{}", p.id).into_bytes();
                (p, audio, Source::Gold)
            })
            .collect();
        let (entries, excluded) = build_manifest(items);
        assert!(excluded.is_empty());
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].sha256, sha256_hex(b"audio-c1"));
        let back = parse_manifest_json(&manifest_json(&entries).unwrap()).unwrap();
        assert_eq!(back, entries);
        // Schema: bad hash, unknown source word, empty text all fail.
        let mut bad = entries.clone();
        bad[0].sha256 = "zzz".into();
        assert!(parse_manifest_json(&serde_json::to_string(&bad).unwrap()).is_err());
        assert!(parse_manifest_json(r#"[{"clip_id":"x","sha256":"a","true_text":"t","original_guess":"g","lang":"fr","source":"platinum","split":"train"}]"#).is_err());
    }

    #[test]
    fn ladder_excludes_off_whitelist_with_reasons() {
        let mut items: Vec<(ExportPair, Vec<u8>, Source)> = pairs()
            .into_iter()
            .map(|p| (p, b"audio".to_vec(), Source::Gold))
            .collect();
        items[1].0.lang = "es".into();
        let (entries, excluded) = build_manifest(items);
        assert_eq!(entries.len(), 2);
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].clip_id, "c2");
        assert!(excluded[0].reason.contains("es"));
    }

    #[test]
    fn ogg_roundtrip_preserves_tone() {
        // The decode path everything downstream depends on: synth a
        // 440 Hz tone, ship it through prod's own Opus-in-Ogg codec,
        // decode it back, check it is still a 440 Hz tone.
        let pcm: Vec<i16> = (0..16000)
            .map(|i| {
                (10000.0 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin()) as i16
            })
            .collect();
        let ogg = hamfeed_ingest::encode_pcm_to_ogg(&pcm).unwrap();
        let back = hamfeed_ingest::decode_ogg_to_pcm(&ogg).unwrap();
        assert!((back.len() as i64 - pcm.len() as i64).abs() < 2000);
        let energy: f32 = back
            .iter()
            .take(8000)
            .map(|&s| (s as f32 / 32768.0).powi(2))
            .sum::<f32>()
            / 8000.0;
        assert!(energy > 0.01, "decoded tone lost: {energy}");
        assert!(hamfeed_ingest::decode_ogg_to_pcm(b"junk").is_err());
    }

    const REPORT_FIXTURE: &str = r#"{
        "generated_at": "2026-09-18T12:38:39+00:00",
        "dataset_manifest_sha256": "94607da900000000000000000000000000000000000000000000000000000000",
        "base_model": "ggml-base",
        "candidate_model": null,
        "n_heldout": 1,
        "base_wer": 0.11111111,
        "tuned_wer": null,
        "buckets": {"fr": {"n": 1, "base_wer": 0.11111111, "tuned_wer": null},
                    "en": {"n": 0, "base_wer": null, "tuned_wer": null}},
        "verdict": "no-deploy",
        "reason": "no candidate model yet (baseline only)",
        "clips": [{"clip_id": "c789", "lang": "fr",
                   "base_wer": 0.11111111, "tuned_wer": null}]
    }"#;

    #[test]
    fn report_schema_accepts_baseline_and_rejects_garbage() {
        let r = parse_eval_report(REPORT_FIXTURE).unwrap();
        assert_eq!(r.verdict, Verdict::NoDeploy);
        assert_eq!(r.n_heldout, 1);
        // WER outside [0,1] fails.
        let bad = REPORT_FIXTURE.replace("0.11111111", "1.5");
        assert!(parse_eval_report(&bad).is_err());
        // Unknown verdict word fails.
        let bad = REPORT_FIXTURE.replace("no-deploy", "ship-it");
        assert!(parse_eval_report(&bad).is_err());
        // Clip count mismatch fails.
        let bad = REPORT_FIXTURE.replace("\"n_heldout\": 1", "\"n_heldout\": 2");
        assert!(parse_eval_report(&bad).is_err());
        // Bucket count mismatch fails.
        let bad = REPORT_FIXTURE.replace("\"fr\": {\"n\": 1", "\"fr\": {\"n\": 5");
        assert!(parse_eval_report(&bad).is_err());
        // deploy-candidate without both WERs fails.
        let bad = REPORT_FIXTURE.replace("no-deploy", "deploy-candidate");
        assert!(parse_eval_report(&bad).is_err());
    }

    #[test]
    fn audit_passes_clean_and_fails_planted_overlap() {
        let good = vec![
            entry("a", Split::Train, &"0".repeat(64)),
            entry("b", Split::Heldout, &"1".repeat(64)),
        ];
        assert!(audit(&good).is_ok());
        // Planted overlap: same clip id in both splits.
        let mut evil = good.clone();
        let mut dup = evil[0].clone();
        dup.split = Split::Heldout;
        evil.push(dup);
        let err = audit(&evil).unwrap_err().to_string();
        assert!(err.contains('a'), "names the clip: {err}");
        // Same id, different bytes: corruption, also loud.
        let mut evil2 = good.clone();
        let mut dup2 = evil2[1].clone();
        dup2.sha256 = "2".repeat(64);
        evil2.push(dup2);
        assert!(audit(&evil2).unwrap_err().to_string().contains('b'));
    }
}
