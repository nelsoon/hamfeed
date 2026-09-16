//! callbook-import: offline CA/US callbook loader (003 T12).
//!
//! Usage:
//!   callbook-import --db <contacts.db> [--ised <amateur.txt>] [--uls <dir>]
//!
//! At least one of `--ised` / `--uls` is required. Imports stream the
//! dumps (never loaded whole) inside one transaction each. Offline only.

use std::process::ExitCode;

fn usage() -> ! {
    eprintln!("usage: callbook-import --db <contacts.db> [--ised <amateur.txt>] [--uls <dir>]");
    std::process::exit(2);
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut db: Option<String> = None;
    let mut ised: Option<String> = None;
    let mut uls: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--db" => db = args.next(),
            "--ised" => ised = args.next(),
            "--uls" => uls = args.next(),
            _ => usage(),
        }
    }
    let Some(db) = db else { usage() };
    if ised.is_none() && uls.is_none() {
        eprintln!("callbook-import: nothing to do (pass --ised and/or --uls)");
        usage();
    }
    let cb = hamfeed_callbook::open(&db);
    let mut total = 0usize;
    if let Some(path) = ised {
        match cb.import_ised(&path) {
            Ok(n) => {
                println!("ised: {n} contacts from {path}");
                total += n;
            }
            Err(e) => {
                eprintln!("ised import failed: {e:?}");
                return ExitCode::from(1);
            }
        }
    }
    if let Some(dir) = uls {
        match cb.import_uls(&dir) {
            Ok(n) => {
                println!("uls: {n} contacts from {dir}");
                total += n;
            }
            Err(e) => {
                eprintln!("uls import failed: {e:?}");
                return ExitCode::from(1);
            }
        }
    }
    println!("done: {total} contacts in {db}");
    ExitCode::SUCCESS
}
