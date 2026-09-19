// audiopus links the system libopus, which the GPU box lacks (no sudo
// there). The matching libopus.so.0 ships next to the binary in
// `finetune/lib/` (vendored from prod); $ORIGIN keeps it working with
// no env vars in cron. System paths remain as fallback.
fn main() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/x86_64-linux-gnu");
}
