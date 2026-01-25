const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const configPath = path.join(__dirname, '.cargo', 'config.toml');
const backupPath = configPath + '.bak';

// 1. Backup and modify config.toml
// wasm-pack doesn't pass -Z flags well, so we put them in the config temporarily.
let configChanged = false;
try {
    if (fs.existsSync(configPath)) {
        fs.copyFileSync(configPath, backupPath);
        let content = fs.readFileSync(configPath, 'utf8');
        if (!content.includes('build-std')) {
            content += '\n\n[unstable]\nbuild-std = ["panic_abort", "std"]\n';
            fs.writeFileSync(configPath, content);
            configChanged = true;
            console.log("Temporarily enabled build-std in .cargo/config.toml");
        }
    } else {
        const dir = path.dirname(configPath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(configPath, '[unstable]\nbuild-std = ["panic_abort", "std"]\n');
        configChanged = true;
        console.log("Temporarily created .cargo/config.toml with build-std");
    }
} catch (err) {
    console.error("Failed to modify .cargo/config.toml:", err.message);
}

function restoreConfig() {
    if (configChanged) {
        configChanged = false;
        try {
            if (fs.existsSync(backupPath)) {
                fs.copyFileSync(backupPath, configPath);
                fs.unlinkSync(backupPath);
                console.log("Restored original .cargo/config.toml content");
            } else {
                // Only delete if we were the ones who created the file from scratch
                fs.unlinkSync(configPath);
                console.log("Removed temporary .cargo/config.toml");
            }
        } catch (err) {
            console.error("Failed to restore .cargo/config.toml:", err.message);
        }
    }
}

// Ensure restoration on exit
process.on('SIGINT', () => { restoreConfig(); process.exit(); });
process.on('exit', restoreConfig);

// 2. Set environment variables
const env = { ...process.env };
env.RUSTFLAGS = [
    "-C", "target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128",
    "-C", "link-arg=--shared-memory",
    "-C", "link-arg=--max-memory=1073741824",
    "-C", "link-arg=--import-memory",
    "-C", "link-arg=--export=__wasm_init_tls",
    "-C", "link-arg=--export=__tls_size",
    "-C", "link-arg=--export=__tls_align",
    "-C", "link-arg=--export=__tls_base"
].join(" ");

env.RUSTUP_TOOLCHAIN = "nightly";

console.log("Building Multi-threaded (Lazy SMP) with wasm-pack...");

const args = [
    "build",
    "--target", "web",
    "--release",
    "--features", "multithreading"
];

const build = spawn("wasm-pack", args, {
    env: env,
    shell: true,
    stdio: 'inherit'
});

build.on('close', (code) => {
    restoreConfig();
    if (code !== 0) {
        console.error(`Build failed with code ${code}`);
        process.exit(code);
    } else {
        console.log("Build completed successfully!");
    }
});
