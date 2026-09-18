//! Box-side glue for the 009 shipper loop (T2). All decisions live in
//! the lib (tested); this binary only moves files and exits loud.
//!
//! ```text
//! hamfeed-train ship-plan --csv manifest.csv --shipped shipped.txt
//! hamfeed-train manifest --csv manifest.csv --audio-dir pairs/ \
//!     --source gold --out manifest.json
//! hamfeed-train audit --manifest dataset/manifest.json [...]
//! ```

use anyhow::{anyhow, bail, Context, Result};
use hamfeed_train as ht;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

fn arg(args: &[String], name: &str) -> Result<String> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
        .ok_or_else(|| anyhow!("missing {name}"))
}

fn read_state(path: &Path) -> HashSet<String> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn cmd_ship_plan(args: &[String]) -> Result<()> {
    let csv = fs::read_to_string(arg(args, "--csv")?).context("read csv")?;
    let shipped = read_state(Path::new(&arg(args, "--shipped")?));
    let export = ht::parse_export_csv(&csv)?;
    for id in ht::plan_ship(&export, &shipped) {
        println!("{id}");
    }
    Ok(())
}

fn audio_for(dir: &Path, id: &str) -> Result<Vec<u8>> {
    let mut hits: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_stem().and_then(|s| s.to_str()) == Some(id) && p.is_file())
        .collect();
    hits.sort();
    hits.into_iter()
        .next()
        .map(|p| fs::read(&p).with_context(|| format!("read {}", p.display())))
        .ok_or_else(|| anyhow!("no audio for {id} in {}", dir.display()))?
}

fn cmd_manifest(args: &[String]) -> Result<()> {
    let csv = fs::read_to_string(arg(args, "--csv")?).context("read csv")?;
    let dir = PathBuf::from(arg(args, "--audio-dir")?);
    let source = match arg(args, "--source")?.as_str() {
        "gold" => ht::Source::Gold,
        "silver" => ht::Source::Silver,
        other => bail!("source must be gold|silver, got {other:?}"),
    };
    let export = ht::parse_export_csv(&csv)?;
    let mut items = Vec::new();
    for pair in export {
        match audio_for(&dir, &pair.id) {
            Ok(bytes) => items.push((pair, bytes, source)),
            Err(e) => bail!("{e:#}"),
        }
    }
    let (entries, excluded) = ht::build_manifest(items);
    fs::write(arg(args, "--out")?, ht::manifest_json(&entries)?).context("write manifest")?;
    eprintln!(
        "manifest: {} kept, {} excluded",
        entries.len(),
        excluded.len()
    );
    for x in &excluded {
        eprintln!("  excluded {}: {}", x.clip_id, x.reason);
    }
    Ok(())
}

fn cmd_audit(args: &[String]) -> Result<()> {
    let paths: Vec<&String> = args
        .windows(2)
        .filter(|w| w[0] == "--manifest")
        .map(|w| &w[1])
        .collect();
    if paths.is_empty() {
        bail!("pass at least one --manifest");
    }
    let mut all = Vec::new();
    for p in paths {
        let text = fs::read_to_string(p).with_context(|| format!("read {p}"))?;
        all.extend(ht::parse_manifest_json(&text)?);
    }
    ht::audit(&all)?;
    println!("audit ok: {} entries, no overlap", all.len());
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("ship-plan") => cmd_ship_plan(&args),
        Some("manifest") => cmd_manifest(&args),
        Some("audit") => cmd_audit(&args),
        _ => {
            eprintln!("usage: hamfeed-train (ship-plan|manifest|audit) ...");
            std::process::exit(2);
        }
    }
}
