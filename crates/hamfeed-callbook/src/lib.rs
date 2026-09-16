//! hamfeed-callbook: local CA/US operator lookup (003 T5).
//!
//! Owns the contacts database file (the only non-store rusqlite user — it
//! never touches the message DB). Lookups are infallible by contract: ANY
//! storage failure — missing file, missing table, corrupt DB — degrades to
//! `Ok(None)`, so enrichment never sees an `Err` from here. Imports are
//! offline admin actions over real dump formats (ISED `;`-delimited, FCC
//! ULS `HD.dat`+`EN.dat`).

use rusqlite::{params, Connection, OpenFlags};

/// One known operator.
#[derive(Debug, Clone, PartialEq)]
pub struct Contact {
    pub callsign: String,
    pub name: String,
    pub country: String,
}

/// Handle for a callbook database file. A missing file is a valid
/// degraded handle: every lookup returns `Ok(None)` until an import
/// creates the DB. Read-only per-call connections — no lock, no held
/// guard across I/O.
#[derive(Debug, Clone)]
pub struct Callbook {
    path: String,
}

/// Open a callbook handle. Never fails on a missing file or directory:
/// degradation is the normal pre-import state, not an error.
pub fn open(db_path: &str) -> Callbook {
    Callbook {
        path: db_path.to_string(),
    }
}

impl Callbook {
    /// Look up one callsign (case-insensitive). Degraded contract: ANY
    /// storage-layer failure maps to `Ok(None)`.
    pub fn lookup(&self, callsign: &str) -> anyhow::Result<Option<Contact>> {
        let key = callsign.to_ascii_uppercase();
        let conn = match Connection::open_with_flags(&self.path, OpenFlags::SQLITE_OPEN_READ_ONLY) {
            Ok(c) => c,
            Err(_) => return Ok(None),
        };
        let mut hit =
            match conn.prepare("SELECT callsign, name, country FROM contacts WHERE callsign=?") {
                Ok(s) => s,
                Err(_) => return Ok(None),
            };
        let mut rows = match hit.query(params![key]) {
            Ok(r) => r,
            Err(_) => return Ok(None),
        };
        let row: Option<Contact> = match rows.next() {
            Ok(Some(r)) => Some(Contact {
                callsign: r.get(0).unwrap_or_default(),
                name: r.get(1).unwrap_or_default(),
                country: r.get(2).unwrap_or_default(),
            }),
            _ => None,
        };
        Ok(row)
    }

    fn ensure_db(path: &str) -> anyhow::Result<Connection> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS contacts(callsign TEXT PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL)",
        )?;
        Ok(conn)
    }

    /// Import a real ISED amateur file (`;`-delimited, no header row):
    /// `callsign;first;surname;…;club_name;…`. Club stations (repeaters,
    /// club calls) take the club name — the meaningful badge identity —
    /// falling back to the holder's personal name. UTF-8 preserved.
    pub fn import_ised(&self, path: &str) -> anyhow::Result<usize> {
        use std::io::BufRead;
        let f = std::fs::File::open(path)?;
        let mut conn = Self::ensure_db(&self.path)?;
        let tx = conn.transaction()?;
        let mut n = 0usize;
        let mut first = true;
        // A single undecodable line must not truncate the import: skip it
        // loudly and carry on (dumps are hundreds of megabytes; one bad
        // line is data dirt, not a fatal error).
        for line in std::io::BufReader::new(f).lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("callbook import: skipping undecodable line ({e})");
                    continue;
                }
            };
            if first {
                first = false;
                if line.to_lowercase().starts_with("callsign;") {
                    continue;
                }
            }
            let cols: Vec<&str> = line.split(';').collect();
            if cols.len() < 3 {
                continue;
            }
            let cs = cols[0].trim().to_ascii_uppercase();
            let club = if cols.len() > 13 {
                let full = cols[13].trim();
                if full.is_empty() {
                    cols[12].trim().to_string()
                } else {
                    full.to_string()
                }
            } else if cols.len() > 12 {
                cols[12].trim().to_string()
            } else {
                String::new()
            };
            let name = if club.is_empty() {
                format!("{} {}", cols[1].trim(), cols[2].trim())
                    .trim()
                    .to_string()
            } else {
                club
            };
            if cs.is_empty() || name.is_empty() {
                continue;
            }
            if tx
                .execute(
                    "INSERT OR REPLACE INTO contacts VALUES(?,?,?)",
                    params![cs, name, "CA"],
                )
                .is_ok()
            {
                n += 1;
            }
        }
        tx.commit()?;
        Ok(n)
    }

    /// Import a real FCC ULS dump directory (`HD.dat` + `EN.dat`,
    /// pipe-delimited, ~200 MB each — both streamed, never loaded).
    /// Keeps EN licensee rows whose unique-id is Active in HD.
    pub fn import_uls(&self, dir: &str) -> anyhow::Result<usize> {
        use std::collections::HashSet;
        use std::io::BufRead;
        let hd = std::fs::File::open(format!("{dir}/HD.dat"))?;
        let mut active: HashSet<u32> = HashSet::new();
        for line in std::io::BufReader::new(hd).lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("callbook import: skipping undecodable HD line ({e})");
                    continue;
                }
            };
            let cols: Vec<&str> = line.split('|').collect();
            if cols.len() < 7 || cols[0] != "HD" {
                continue;
            }
            if cols[5] != "A" {
                continue;
            }
            if let Ok(id) = cols[1].trim().parse::<u32>() {
                active.insert(id);
            }
        }
        let en = std::fs::File::open(format!("{dir}/EN.dat"))?;
        let mut conn = Self::ensure_db(&self.path)?;
        let tx = conn.transaction()?;
        let mut n = 0usize;
        for line in std::io::BufReader::new(en).lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("callbook import: skipping undecodable EN line ({e})");
                    continue;
                }
            };
            let cols: Vec<&str> = line.split('|').collect();
            if cols.len() < 8 || cols[0] != "EN" || cols[5] != "L" {
                continue;
            }
            let Ok(id) = cols[1].trim().parse::<u32>() else {
                continue;
            };
            if !active.contains(&id) {
                continue;
            }
            let cs = cols[4].trim().to_ascii_uppercase();
            let name = cols[7].trim().to_string();
            if cs.is_empty() || name.is_empty() {
                continue;
            }
            if tx
                .execute(
                    "INSERT OR REPLACE INTO contacts VALUES(?,?,?)",
                    params![cs, name, "US"],
                )
                .is_ok()
            {
                n += 1;
            }
        }
        tx.commit()?;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> std::path::PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("hamfeed-cb-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn ised_import() {
        let dir = test_dir("ised");
        // Tiny but shape-honest ISED sample: personal, club (full name
        // wins), malformed row skipped.
        std::fs::write(
            dir.join("amateur.txt"),
            "VE2DEM;Jean;Tremblay;;;;Qc;;;;;;;;\n\
             VE2RGC;;;Club;Montreal;QC;;;;;;;;Club Radio Montreal;;;\n\
             garbage-without-delimiters\n",
        )
        .unwrap();
        let cb = open(&dir.join("cb.db").to_string_lossy());
        assert_eq!(
            cb.import_ised(&dir.join("amateur.txt").to_string_lossy())
                .unwrap(),
            2
        );
        let hit = cb.lookup("ve2dem").unwrap().expect("personal hit");
        assert_eq!(hit.name, "Jean Tremblay");
        assert_eq!(hit.country, "CA");
        let club = cb.lookup("VE2RGC").unwrap().expect("club hit");
        assert_eq!(club.name, "Club Radio Montreal");
    }

    #[test]
    fn bad_line_does_not_truncate_import() {
        // Invalid UTF-8 mid-file: the bad line is skipped, later rows
        // still land (map_while(Result::ok) used to stop the whole import).
        let dir = test_dir("badline");
        let mut bytes = b"VE2AAA;Alain;Aubert\n".to_vec();
        bytes.extend_from_slice(b"\xff\xfe not utf-8\n");
        bytes.extend_from_slice(b"VE2BBB;Berthe;Blais\n");
        std::fs::write(dir.join("amateur.txt"), &bytes).unwrap();
        let cb = open(&dir.join("cb.db").to_string_lossy());
        assert_eq!(
            cb.import_ised(&dir.join("amateur.txt").to_string_lossy())
                .unwrap(),
            2
        );
        assert!(cb.lookup("VE2AAA").unwrap().is_some());
        assert!(cb.lookup("VE2BBB").unwrap().is_some());
    }

    #[test]
    fn uls_import_active_only() {
        let dir = test_dir("uls");
        std::fs::write(
            dir.join("HD.dat"),
            "HD|101|x|x|x|A|x\nHD|102|x|x|x|C|x\nHD|bad|x\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("EN.dat"),
            "EN|101|x|x|W1AW|L|x|Hiram Maxim\n\
             EN|102|x|x|K1ZZ|L|x|Cancelled Op\n\
             EN|103|x|x|N0CALL|L|x|Never Active\n",
        )
        .unwrap();
        let cb = open(&dir.join("cb.db").to_string_lossy());
        assert_eq!(cb.import_uls(&dir.to_string_lossy()).unwrap(), 1);
        let hit = cb.lookup("w1aw").unwrap().expect("active licensee");
        assert_eq!(hit.name, "Hiram Maxim");
        assert_eq!(hit.country, "US");
        assert!(cb.lookup("K1ZZ").unwrap().is_none(), "cancelled excluded");
        assert!(cb.lookup("N0CALL").unwrap().is_none(), "inactive excluded");
    }

    #[test]
    fn lookup_hit() {
        let dir = test_dir("hit");
        let cb = open(&dir.join("cb.db").to_string_lossy());
        std::fs::write(dir.join("amateur.txt"), "VE2DEM;Jean;Tremblay\n").unwrap();
        cb.import_ised(&dir.join("amateur.txt").to_string_lossy())
            .unwrap();
        // Case-insensitive exact match.
        assert_eq!(
            cb.lookup("ve2dem").unwrap().unwrap(),
            Contact {
                callsign: "VE2DEM".into(),
                name: "Jean Tremblay".into(),
                country: "CA".into(),
            }
        );
        assert!(cb.lookup("VE2XXX").unwrap().is_none());
    }

    #[test]
    fn missing_db_degrades_to_none() {
        // No import ever ran: lookups degrade, never error.
        let dir = test_dir("missing");
        let cb = open(&dir.join("never-created.db").to_string_lossy());
        assert_eq!(cb.lookup("VE2DEM").unwrap(), None);
    }

    #[test]
    fn corrupt_db_degrades() {
        // Garbage bytes at the DB path: still Ok(None), never Err.
        let dir = test_dir("corrupt");
        let path = dir.join("cb.db");
        std::fs::write(&path, b"not a database at all").unwrap();
        let cb = open(&path.to_string_lossy());
        assert_eq!(cb.lookup("VE2DEM").unwrap(), None);
    }
}
