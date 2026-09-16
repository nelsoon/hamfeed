//! hamfeed-web binary: serve the live feed.

use std::path::PathBuf;

use hamfeed_web::{create_app, AppState};

fn usage() -> ! {
    eprintln!("usage: hamfeed-web [--config PATH] [--port N] [--static-dir DIR]");
    std::process::exit(2);
}

#[tokio::main]
async fn main() {
    let mut config = PathBuf::from("hamfeed.toml");
    let mut port: u16 = 8080;
    let mut static_dir: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config = PathBuf::from(args.next().unwrap_or_else(|| usage())),
            "--port" => {
                port = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--static-dir" => {
                static_dir = Some(args.next().unwrap_or_else(|| usage()));
            }
            _ => usage(),
        }
    }

    let pipe = hamfeed_pipeline::Pipeline::open(&config).unwrap_or_else(|e| {
        eprintln!("hamfeed-web: cannot start\n{e:?}");
        std::process::exit(1);
    });
    let dir = static_dir.unwrap_or_else(|| "crates/hamfeed-web/static".into());
    let state = AppState::new(pipe, dir);
    let app = create_app(state);
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("hamfeed-web: cannot bind {addr}: {e}");
            std::process::exit(1);
        });
    println!("hamfeed-web: serving on http://{addr}/");
    axum::serve(listener, app).await.unwrap_or_else(|e| {
        eprintln!("hamfeed-web: server error: {e}");
        std::process::exit(1);
    });
}
