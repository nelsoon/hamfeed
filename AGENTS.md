# AGENTS.md (public, safe subset)

How to build, test, and contribute. No private context here.

- Build: `cargo build --workspace` (stable LLVM).
  Fast local loop (nightly only): `RUSTFLAGS="-Zcodegen-backend=cranelift"
  cargo +nightly test --workspace`. Never ship Cranelift builds; CI uses
  stable LLVM. (Do NOT put the flag in `.cargo/config.toml` — stable
  refuses nightly-only options.)
- Test: `cargo test --workspace --all-features` (stable). All tests are
  fixture-backed; no audio hardware or network required. STT fixture tests
  need `models/ggml-tiny.bin` (`sh scripts/download-model.sh tiny`; CI
  caches it) and fail loudly when it is absent — never skip them quietly.
- Lint: `cargo fmt --all -- --check`, `cargo clippy --all-targets
  --all-features -- -D warnings`.
- PRs: small, one concern each, include test notes. CI must be green;
  human review checks behavior, not just green CI.
- Group contract: store owns schema + SQL; pipeline owns status/triage
  transitions; web is a thin reader over store + pipeline helpers.
  `group_id`/`seq` are assigned only by ingest.
