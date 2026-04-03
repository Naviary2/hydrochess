use std::env;
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    if target.contains("wasm32") {
        println!("cargo:rustc-link-arg=-zstack-size=8388608");
    }

    // Only embed git commit info when building the sprt feature.
    if env::var("CARGO_FEATURE_SPRT").is_ok() {
        // Embed git commit info so every binary self-reports which snapshot it was built from.
        // The values are empty strings when git is unavailable or the repo has no commits.
        let commit = Command::new("git")
            .args(["rev-parse", "--short=7", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();

        let date = Command::new("git")
            .args(["log", "-1", "--format=%cd", "--date=format-local:%Y-%m-%d", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        
        // Check if the worktree is dirty (has uncommitted .rs files).
        let is_dirty = Command::new("git")
            .args(["status", "--porcelain", "--", "*.rs"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty())
            .unwrap_or(false);
        
        println!("cargo:rustc-env=SPRT_GIT_COMMIT={}", commit);
        println!("cargo:rustc-env=SPRT_GIT_DATE={}", date);
        println!("cargo:rustc-env=SPRT_GIT_DIRTY={}", if is_dirty { "1" } else { "0" });
        
        // Rebuild when the checked-out commit changes, whether refs are stored
        // as loose files under .git/refs or packed into .git/packed-refs.
        println!("cargo:rerun-if-changed=.git/HEAD");
        println!("cargo:rerun-if-changed=.git/refs");
        println!("cargo:rerun-if-changed=.git/packed-refs");
    }
}
