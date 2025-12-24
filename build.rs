use std::env;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    // WASM: Set stack size via linker argument
    if target.contains("wasm32") {
        println!("cargo:rustc-link-arg=-zstack-size=8388608");
    }

    // Native tests: Increase thread stack size to 8MB for Searcher's large arrays
    // The Searcher struct has ~270KB of arrays (history + capture_history + countermoves)
    // Default stack is often 1MB which isn't enough when tests run in parallel
    if env::var("PROFILE").unwrap_or_default() == "debug" {
        println!("cargo:rustc-env=RUST_MIN_STACK=8388608");
    }
}
