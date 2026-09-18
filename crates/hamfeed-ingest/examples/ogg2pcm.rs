//! ogg2pcm: decode one hamfeed Opus-in-Ogg clip to raw s16le on stdout.
//! Dataset-prep helper for the offline ML harness (the strict decoders in
//! Python land reject our page layout; our own decoder eats it daily).
//! Usage: ogg2pcm <clip.ogg> > clip.s16   (16 kHz mono, little-endian i16)

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: ogg2pcm <clip.ogg> > clip.s16");
        std::process::exit(2);
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let pcm = hamfeed_ingest::decode_ogg_to_pcm(&bytes).unwrap_or_else(|e| {
        eprintln!("cannot decode {path}: {e}");
        std::process::exit(1);
    });
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in &pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    use std::io::Write as _;
    std::io::stdout().write_all(&out).unwrap();
}
