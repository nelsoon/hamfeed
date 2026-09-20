//! hamfeed-callsign: self-ID extraction from transcripts (003 T1).
//!
//! Pure string logic, no I/O: plain callsigns (`VE2DEM`) plus spelled NATO
//! runs (`"Victor Echo 2 Delta Echo Mike"`, FR or EN, accents tolerated).
//! Ranking is (confidence, byte position) — length is NEVER a tie-break.
//! Group contract: pipeline calls in, nothing calls back.

use std::sync::OnceLock;

use regex::Regex;

/// How a hit was heard: written plain or spelled via phonetics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitKind {
    Plain,
    Spelled,
}

/// One candidate callsign mention in a transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct CallsignHit {
    /// Surface text as heard (`"VE2DEM"` or `"Victor Echo 2 …"`).
    pub raw: String,
    /// Compact uppercase form (`"VE2DEM"`).
    pub normalized: String,
    pub kind: HitKind,
    /// 0.9 plain, 0.7 spelled.
    pub confidence: f32,
    /// Byte offset in the input text (the rank key after confidence).
    pub pos: usize,
}

/// Minimum spelled-run length admitted as a candidate.
pub const MIN_SPELLED_RUN: usize = 4;

/// Uppercase and strip to alphanumerics (`"ve-2dem"` → `"VE2DEM"`).
pub fn normalize(raw: &str) -> String {
    raw.to_ascii_uppercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

fn plain_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Z]{1,2}\d[A-Z]{1,3}\b").unwrap())
}

/// True when the compact form has callsign shape.
pub fn is_valid(normalized: &str) -> bool {
    plain_re().is_match(normalized)
}

/// Canadian amateur prefix blocks (operator-supplied allocation table):
/// CF–CK, CY–CZ, VA–VG (covers VA2/VE2 Québec), VO, VX–VY, XJ–XO.
/// True when the leading letters fall inside one of them; anything else
/// (including a bare single letter) is not a plannable Canadian prefix.
pub fn canadian_prefix_ok(normalized: &str) -> bool {
    let letters: String = normalized
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    let mut it = letters.chars();
    match (it.next(), it.next(), it.next()) {
        (Some(a), Some(b), None) => matches!(
            (a, b),
            ('C', 'F'..='K')
                | ('C', 'Y'..='Z')
                | ('V', 'A'..='G')
                | ('V', 'O')
                | ('V', 'X'..='Y')
                | ('X', 'J'..='O')
        ),
        _ => false,
    }
}

/// Q brevity codes are procedure words, never callsigns — but a spelled
/// run ("quebec sierra lima") assembles them letter-perfect. Anything
/// in this set is rejected wherever a candidate is admitted, both
/// passes. Shape alone cannot tell QSL from a callsign; the list can.
pub fn is_qcode(normalized: &str) -> bool {
    matches!(
        normalized,
        "QRA"
            | "QRG"
            | "QRH"
            | "QRI"
            | "QRK"
            | "QRL"
            | "QRM"
            | "QRN"
            | "QRO"
            | "QRP"
            | "QRQ"
            | "QRS"
            | "QRT"
            | "QRU"
            | "QRV"
            | "QRW"
            | "QRX"
            | "QRZ"
            | "QSA"
            | "QSB"
            | "QSD"
            | "QSG"
            | "QSK"
            | "QSL"
            | "QSM"
            | "QSN"
            | "QSO"
            | "QSP"
            | "QST"
            | "QSU"
            | "QSV"
            | "QSW"
            | "QSX"
            | "QSY"
            | "QSZ"
            | "QTC"
            | "QTH"
            | "QTR"
            | "QTU"
    )
}

/// US amateur prefix blocks (operator-supplied): single K/N/W, or
/// AA–AL, KA–KZ, NA–NZ, WA–WZ pairs — each followed by a digit.
/// True only when the shape matches; the digit itself is not range
/// checked (district 0–9 all exist).
pub fn us_prefix_ok(normalized: &str) -> bool {
    let mut chars = normalized.chars();
    match (chars.next(), chars.next(), chars.next()) {
        (Some(a), Some(d), _) if d.is_ascii_digit() => {
            matches!(a, 'K' | 'N' | 'W')
        }
        (Some(a), Some(b), Some(d)) if d.is_ascii_digit() => {
            matches!(
                (a, b),
                ('A', 'A'..='L') | ('K', 'A'..='Z') | ('N', 'A'..='Z') | ('W', 'A'..='Z')
            )
        }
        _ => false,
    }
}

/// Nationally plannable: Canadian or US prefix. Shape-only `is_valid`
/// admits junk like `V2CSQ` (no such block); learning and suggestion
/// must gate on this, never on shape alone.
pub fn national_ok(normalized: &str) -> bool {
    canadian_prefix_ok(normalized) || us_prefix_ok(normalized)
}

/// Repair candidates for a bare-V shape ("V2CRS"): on a Québec repeater
/// the lone Victor is a dropped middle word — Echo or Alfa. Returns the
/// VE- and VA-prefixed forms in that order; empty for anything else.
/// The caller picks via the callbook and never invents: no book hit
/// means the heard form stands.
pub fn bare_v_repair_candidates(normalized: &str) -> Vec<String> {
    let mut chars = normalized.chars();
    match (chars.next(), chars.next()) {
        (Some('V'), Some(d)) if d.is_ascii_digit() => {
            let rest = &normalized[1..];
            vec![format!("VE{rest}"), format!("VA{rest}")]
        }
        _ => vec![],
    }
}

/// Lowercase + strip accents so Écho/echo/ECHO all match.
pub fn fold(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| match c {
            'à' | 'á' | 'â' | 'ä' => 'a',
            'ç' => 'c',
            'è' | 'é' | 'ê' | 'ë' => 'e',
            'ì' | 'í' | 'î' | 'ï' => 'i',
            'ñ' => 'n',
            'ò' | 'ó' | 'ô' | 'ö' => 'o',
            'ù' | 'ú' | 'û' | 'ü' => 'u',
            'ÿ' => 'y',
            _ => c,
        })
        .collect()
}

/// NATO word (either language's digits) → letter/digit, if any.
fn nato_word(w: &str) -> Option<char> {
    Some(match w {
        "alpha" => 'A',
        "bravo" => 'B',
        "charlie" => 'C',
        "delta" => 'D',
        "echo" => 'E',
        "foxtrot" => 'F',
        "golf" => 'G',
        "hotel" => 'H',
        "india" => 'I',
        "juliett" | "juliet" => 'J',
        "kilo" => 'K',
        "lima" => 'L',
        "mike" => 'M',
        "november" => 'N',
        "oscar" => 'O',
        "papa" => 'P',
        "quebec" => 'Q',
        "romeo" => 'R',
        "sierra" => 'S',
        "tango" => 'T',
        "uniform" => 'U',
        "victor" => 'V',
        "whiskey" => 'W',
        "xray" | "x-ray" => 'X',
        "yankee" => 'Y',
        "zulu" => 'Z',
        "zero" | "oh" => '0',
        "one" => '1',
        "two" => '2',
        "three" => '3',
        "four" => '4',
        "five" => '5',
        "six" => '6',
        "seven" => '7',
        "eight" => '8',
        "nine" => '9',
        // French digits (letters use the same NATO table above).
        "un" => '1',
        "deux" => '2',
        "trois" => '3',
        "quatre" => '4',
        "cinq" => '5',
        "sept" => '7',
        "huit" => '8',
        "neuf" => '9',
        _ => return None,
    })
}

/// Extract candidate callsigns from a transcript, best first.
///
/// `lang` is accepted for forward compatibility and currently ignored:
/// both languages' digit words always apply (S2: Québec French speech is
/// routinely FR/EN mixed mid-sentence).
pub fn extract(text: &str, _lang: &str) -> Vec<CallsignHit> {
    let mut hits: Vec<CallsignHit> = vec![];

    // Plain pass on uppercased text. Uppercasing is ASCII
    // length-preserving, so byte offsets match the original text.
    let upper = text.to_ascii_uppercase();
    for m in plain_re().find_iter(&upper) {
        let norm = normalize(m.as_str());
        // The shape regex needs a digit, so most Q codes never reach
        // here; the ban stays explicit so no pass can admit one.
        if is_qcode(&norm) {
            continue;
        }
        hits.push(CallsignHit {
            raw: m.as_str().to_string(),
            normalized: norm,
            kind: HitKind::Plain,
            confidence: 0.9,
            pos: m.start(),
        });
    }

    // Spelled pass on folded words: maximal runs of NATO words/digits.
    let mut run: Vec<(String, usize)> = vec![]; // (word, byte_offset)
    let flush = |run: &mut Vec<(String, usize)>, hits: &mut Vec<CallsignHit>| {
        if run.len() >= MIN_SPELLED_RUN {
            let raw: String = run
                .iter()
                .map(|(w, _)| w.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let mut norm = String::new();
            for (w, _) in run.iter() {
                if let Some(c) = nato_word(&fold(w)) {
                    norm.push(c);
                } else {
                    // Digit-only tokens admitted by the run gate have no
                    // NATO mapping: keep their literal digits (never unwrap).
                    for d in w.chars() {
                        if d.is_ascii_digit() {
                            norm.push(d);
                        }
                    }
                }
            }
            // A spelled run assembles Q codes letter-perfect ("quebec
            // sierra lima" → QSL): shape is not enough, ban them here.
            if is_valid(&norm) && !is_qcode(&norm) {
                // No dedupe against plain hits: spelled raw contains spaces
                // so byte spans never overlap; ranking decides the sender.
                hits.push(CallsignHit {
                    raw,
                    normalized: norm,
                    kind: HitKind::Spelled,
                    confidence: 0.7,
                    pos: run[0].1,
                });
            }
        }
        run.clear();
    };

    // Word tokens with exact byte offsets (char_indices), so multi-byte
    // delimiters (accents, em-dash) never shift `pos`.
    let mut tokens: Vec<(String, usize)> = vec![];
    let mut buf = String::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() || c == '-' {
            if start.is_none() {
                start = Some(i);
            }
            buf.push(c);
        } else if let Some(s) = start.take() {
            tokens.push((std::mem::take(&mut buf), s));
        }
    }
    if let Some(s) = start.take() {
        tokens.push((std::mem::take(&mut buf), s));
    }
    for (w, off) in &tokens {
        let f = fold(w);
        if nato_word(&f).is_some() || w.chars().all(|c| c.is_ascii_digit()) {
            run.push((w.to_string(), *off));
        } else {
            flush(&mut run, &mut hits);
        }
    }
    flush(&mut run, &mut hits);

    // Rank: confidence desc, byte-offset asc. `"VE3MA ... W1AW"` (both
    // plain 0.9) picks VE3MA — length is NEVER a tie-break.
    hits.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap()
            .then_with(|| a.pos.cmp(&b.pos))
    });
    hits
}

/// Optimal String Alignment distance on chars: insert/delete/substitute
/// cost 1 + adjacent transposition cost 1; empty-vs-n → n.
/// Inputs are normalized defensively via [`normalize`].
pub fn osa_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = normalize(a).chars().collect();
    let b: Vec<char> = normalize(b).chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut d = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[m][n]
}

/// Fuzzy candidates for a heard callsign: every entry with
/// `osa_distance <= max_dist` (exact match excluded), sorted by
/// (distance, callsign). Returns NORMALIZED strings.
pub fn fuzzy_candidates(heard: &str, candidates: &[String], max_dist: usize) -> Vec<(String, u8)> {
    let heard = normalize(heard);
    let mut out: Vec<(String, u8)> = Vec::new();
    for c in candidates {
        let nc = normalize(c);
        if nc == heard {
            continue;
        }
        let dist = osa_distance(&heard, &nc);
        if dist <= max_dist {
            debug_assert!(dist <= 255);
            out.push((nc, dist as u8));
        }
    }
    out.sort_by(|x, y| x.1.cmp(&y.1).then_with(|| x.0.cmp(&y.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_hit() {
        let hits = extract("ici VE2DEM vous m'entendez", "fr");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].normalized, "VE2DEM");
        assert_eq!(hits[0].kind, HitKind::Plain);
        assert_eq!(hits[0].raw, "VE2DEM");
    }

    #[test]
    fn canadian_blocks_and_bare_v_repair() {
        // Operator allocation table: CF–CK, CY–CZ, VA–VG, VO, VX–VY, XJ–XO.
        for ok in [
            "VE2DEM", "VA2ABC", "VE2CRS", "VO1XYZ", "CY2ABC", "CF2ABC", "VX2ABC", "XJ2ABC",
            "VG2ABC", "CK2ABC", "CZ2ABC", "VY2ABC", "XO2ABC",
        ] {
            assert!(canadian_prefix_ok(ok), "{ok} must pass");
        }
        for bad in ["V2CRS", "W1AW", "F5ABC", "G2ABC", "2CRS"] {
            assert!(!canadian_prefix_ok(bad), "{bad} must fail");
        }
        // Bare-V repair: dropped Echo or Alfa, VE tried first.
        assert_eq!(
            bare_v_repair_candidates("V2CRS"),
            vec!["VE2CRS".to_string(), "VA2CRS".to_string()]
        );
        assert!(bare_v_repair_candidates("VE2DEM").is_empty());
        assert!(bare_v_repair_candidates("W1AW").is_empty());
    }

    #[test]
    fn spelled_en() {
        let hits = extract("this is victor echo two delta echo mike over", "en");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].normalized, "VE2DEM");
        assert_eq!(hits[0].kind, HitKind::Spelled);
    }

    #[test]
    fn spelled_fr_accents() {
        // Québec French: French digit words + accented NATO ("VÉ2DEM").
        let hits = extract("ici victor écho deux delta écho mike", "fr");
        assert_eq!(hits.len(), 1, "accents + FR digits must parse");
        assert_eq!(hits[0].normalized, "VE2DEM");
    }

    #[test]
    fn run_gate_rejects_short() {
        // Three NATO words: below the run gate, and no plain hit either.
        assert!(extract("bravo delta mike over", "en").is_empty());
    }

    #[test]
    fn qcodes_never_admit() {
        // Procedure words assemble letter-perfect through the spelled
        // pass; the ban kills them. "merci QSL" names an addressee, not
        // a sender — and QSL is not a callsign either way.
        assert!(extract("quebec sierra lima over", "en").is_empty());
        assert!(extract("merci QSL et à la prochaine", "fr").is_empty());
        assert!(extract("QTH Montréal, QSB ce soir", "fr").is_empty());
        assert!(is_qcode("QSL") && is_qcode("QTH") && is_qcode("QRZ"));
        assert!(!is_qcode("VE2DEM"));
        // Real callsigns still pass through both passes.
        assert_eq!(extract("ici VE2DEM", "fr")[0].normalized, "VE2DEM");
    }

    #[test]
    fn us_blocks_and_national_gate() {
        for ok in [
            "W1AW", "K2ABC", "N0CALL", "AA2XYZ", "KA1ABC", "VE2DEM", "VA2LHA",
        ] {
            assert!(national_ok(ok), "{ok} must pass");
        }
        for bad in ["V2CSQ", "F5ABC", "G2ABC", "QSL", "2CRS", "ZZ9ZZZ"] {
            assert!(!national_ok(bad), "{bad} must fail");
        }
        // Single-letter US prefixes need the digit right after;
        // two-letter blocks (KA–KZ…) take it third.
        assert!(us_prefix_ok("K2A"));
        assert!(us_prefix_ok("KA2A"));
        assert!(!us_prefix_ok("AZ2A"));
    }

    #[test]
    fn rank_confidence_then_position() {
        // Spelled first in text, plain later: plain (0.9) still wins.
        let hits = extract("victor echo two delta echo mike calling VE3MA", "en");
        assert!(hits.len() >= 2);
        assert_eq!(hits[0].normalized, "VE3MA");
        assert_eq!(hits[0].kind, HitKind::Plain);
    }

    #[test]
    fn length_never_tiebreaks() {
        // Both plain 0.9: earlier position wins even though W1AW is shorter
        // than the (invalid-shape, spelled-only) alternative — and two
        // plains pick the first regardless of length either way.
        let hits = extract("VE3MA then W1AW", "en");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].normalized, "VE3MA");
        assert_eq!(hits[1].normalized, "W1AW");
    }

    #[test]
    fn fuzzy_orders_by_dist() {
        let cands = vec![
            "VE2DEM".to_string(),
            "VE2DEN".to_string(),
            "VE2DXM".to_string(),
            "VE2DXY".to_string(),
            "W1AW".to_string(),
        ];
        let out = fuzzy_candidates("VE2DEM", &cands, 2);
        // Exact match excluded; distance 1 first (ties alphabetical),
        // then distance 2; W1AW is farther than max_dist.
        assert_eq!(
            out,
            vec![
                ("VE2DEN".to_string(), 1),
                ("VE2DXM".to_string(), 1),
                ("VE2DXY".to_string(), 2),
            ]
        );
    }

    #[test]
    fn osa_transposition_costs_one() {
        assert_eq!(osa_distance("VE2DEM", "VE2DME"), 1);
        assert_eq!(osa_distance("", "VE2"), 3);
        assert_eq!(osa_distance("ve2dem", "VE2DEM"), 0);
    }

    #[test]
    fn french_six_maps_to_digit() {
        let hits = extract("victor echo six delta echo mike", "fr");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].normalized, "VE6DEM");
    }
}
