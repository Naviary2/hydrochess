#!/usr/bin/env node
/**
 * HydroChess Web SPRT Helper
 *
 * - Treats root/pkg as the OLD engine snapshot
 * - Builds a NEW web WASM into root/pkg-new
 * - Copies both into sprt/web/pkg-old and sprt/web/pkg-new
 * - Starts `npx serve .` in sprt/web so browser UI can import both
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

const SPRT_DIR = __dirname;
const PROJECT_ROOT = path.join(SPRT_DIR, '..');

// Root-level packages
const ROOT_PKG_OLD = path.join(PROJECT_ROOT, 'pkg-old');  // existing old snapshot

// Web UI directories
const WEB_DIR = path.join(SPRT_DIR, 'web');
const WEB_PKG_OLD_DIR = path.join(WEB_DIR, 'pkg-old');
const WEB_PKG_NEW_DIR = path.join(WEB_DIR, 'pkg-new');

function copyDirectory(src, dest) {
    if (!fs.existsSync(src)) {
        throw new Error(`Source directory does not exist: ${src}`);
    }
    if (!fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
    }
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);
        if (entry.isDirectory()) {
            copyDirectory(srcPath, destPath);
        } else {
            fs.copyFileSync(srcPath, destPath);
        }
    }
}

function rmDirIfExists(dir) {
    if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

function snapshotOldFromRoot() {
    if (!fs.existsSync(ROOT_PKG_OLD)) {
        console.error('[web-sprt] Error: expected OLD pkg-old directory at', ROOT_PKG_OLD);
        console.error('[web-sprt] Build your old reference version with: wasm-pack build --target web --out-dir pkg-old');
        process.exit(1);
    }

    if (!fs.existsSync(WEB_DIR)) {
        fs.mkdirSync(WEB_DIR, { recursive: true });
    }

    console.log('[web-sprt] Copying OLD pkg (root/pkg-old) to sprt/web/pkg-old');
    rmDirIfExists(WEB_PKG_OLD_DIR);
    copyDirectory(ROOT_PKG_OLD, WEB_PKG_OLD_DIR);
}

function buildNewWebPkg() {
    console.log('\n[web-sprt] Building NEW WASM (target=web -> sprt/web/pkg-new)...');
    try {
        rmDirIfExists(WEB_PKG_NEW_DIR);
        const extraFeatures = (process.env.EVAL_TUNING === '1' || process.env.EVAL_TUNING === 'true')
            ? ' --features eval_tuning'
            : '';
        execSync('wasm-pack build --target web --out-dir sprt/web/pkg-new' + extraFeatures, {
            cwd: PROJECT_ROOT,
            stdio: 'inherit',
        });
    } catch (e) {
        console.error('[web-sprt] wasm-pack build failed:', e.message);
        process.exit(1);
    }
    if (!fs.existsSync(WEB_PKG_NEW_DIR)) {
        console.error('[web-sprt] Build finished but pkg-new is missing:', WEB_PKG_NEW_DIR);
        process.exit(1);
    }
}

function startServer() {
    console.log('[web-sprt] Starting dev server in sprt/web (npx serve .)');
    console.log('[web-sprt] Open this URL in your browser (default): http://localhost:3000');

    const child = spawn('npx', ['serve', '.'], {
        cwd: WEB_DIR,
        // Silence verbose HTTP 2xx/304 access logs on stdout while keeping
        // error output visible on stderr.
        stdio: ['ignore', 'ignore', 'inherit'],
        shell: true, // avoid EINVAL on Windows by using shell resolution for npx
    });

    child.on('exit', (code) => {
        process.exit(code ?? 0);
    });
}

(function main() {
    try {
        snapshotOldFromRoot();
        buildNewWebPkg();
        startServer();
    } catch (e) {
        console.error('[web-sprt] Fatal error:', e.message);
        process.exit(1);
    }
})();
