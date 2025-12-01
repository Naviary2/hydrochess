#!/usr/bin/env node
/**
 * Puppeteer-based dataset generator for Texel-style tuning.
 *
 * Workflow:
 * - Spawns the existing sprt.js helper, which builds web WASM and starts
 *   `npx serve .` in sprt/web on http://localhost:3000/.
 * - Launches a headless Chromium via Puppeteer.
 * - Loads the web SPRT UI, waits for WASM readiness, configures a fast
 *   time control, and starts an SPRT run.
 * - Waits until SPRT finishes, then pulls Texel samples exported from the
 *   page (via window.__sprt_export_samples) and writes them to
 *   sprt/data/texel_samples.jsonl.
 *
 * This script does NOT depend on Node-target WASM; all engine use happens
 * inside the browser.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');
const puppeteer = require('puppeteer');

const SPRT_DIR = __dirname;
const DEFAULT_WEB_URL = process.env.SPRT_URL || 'http://localhost:3000/';
const DATA_DIR = path.join(SPRT_DIR, 'data');
const DATA_FILE = path.join(DATA_DIR, 'texel_features.jsonl');

async function waitForServer(url, timeoutMs = 120000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    function check() {
      const req = http.get(url, (res) => {
        res.resume();
        resolve();
      });
      req.on('error', () => {
        if (Date.now() - start > timeoutMs) {
          reject(new Error('Timed out waiting for server at ' + url));
        } else {
          setTimeout(check, 500);
        }
      });
    }
    check();
  });
}

function waitForWebUrlFromSprt(sprtProc, timeoutMs = 180000) {
  return new Promise((resolve, reject) => {
    let resolved = false;

    function finish(err, url) {
      if (resolved) return;
      resolved = true;
      clearTimeout(timer);
      if (err) {
        reject(err);
      } else {
        resolve(url || DEFAULT_WEB_URL);
      }
    }

    const timer = setTimeout(() => {
      finish(new Error('Timed out waiting for server URL from sprt.js'));
    }, timeoutMs);

    if (!sprtProc.stdout) {
      finish(null, DEFAULT_WEB_URL);
      return;
    }

    sprtProc.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      // Mirror sprt.js output to our own stdout so the user still sees logs.
      process.stdout.write(text);

      const lines = text.split(/\r?\n/);
      for (const line of lines) {
        if (!line) continue;
        let m = line.match(/- Local:\s+(https?:\/\/[^\s]+)/);
        if (!m) {
          m = line.match(/Open this URL[^:]*:\s*(https?:\/\/[^\s]+)/);
        }
        if (m) {
          let url = m[1];
          if (!url.endsWith('/')) {
            url += '/';
          }
          finish(null, url);
          return;
        }
      }
    });

    sprtProc.on('exit', (code) => {
      finish(new Error('sprt.js exited before reporting server URL (code ' + code + ')'));
    });
  });
}

async function runPuppeteer(webUrl, existingCount, stream) {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  console.log('[gen-dataset] Opening SPRT page at', webUrl);
  await page.goto(webUrl, { waitUntil: 'networkidle2', timeout: 120000 });

  // Wait for WASM ready hook
  console.log('[gen-dataset] Waiting for SPRT/WASM to become ready...');
  await page.waitForFunction(
    'window.__sprt_is_ready && window.__sprt_is_ready()',
    { timeout: 120000 }
  );
  console.log('[gen-dataset] SPRT page is ready, configuring run...');

  // Configure a fast test suitable for dataset generation
  await page.evaluate(() => {
    const tc = document.getElementById('sprtTimeControl');
    const conc = document.getElementById('sprtConcurrency');
    const minGames = document.getElementById('sprtMinGames');
    const maxGames = document.getElementById('sprtMaxGames');
    const maxMoves = document.getElementById('sprtMaxMoves');
    const mat = document.getElementById('sprtMaterialThreshold');

    if (tc) tc.value = '1+0.08';
    if (conc) conc.value = '8';
    if (minGames) minGames.value = '2000';
    if (maxGames) maxGames.value = '4000';
    if (maxMoves) maxMoves.value = '120';
    if (mat) mat.value = '1500';
  });

  // Start SPRT run
  await page.click('#runSprt');
  console.log('[gen-dataset] SPRT run started, polling status and flushing samples...');

  // Poll SPRT status periodically so the user sees progress even when
  // the browser UI is busy and HTTP logs are silenced.
  const start = Date.now();
  // Maximum time to wait for a full SPRT run (1 hour)
  const maxSeconds = 3600;
  // Poll every 10 seconds
  const pollIntervalMs = 10000;

  // Track how many raw samples we've already exported from the browser
  let sampleOffset = 0;
  let totalNew = 0;
  let lastFlush = Date.now();

  async function flushNewSamples() {
    const now = Date.now();
    const elapsedSinceFlush = (now - lastFlush) / 1000;
    if (elapsedSinceFlush < 60 && totalNew > 0) {
      // We already flushed within the last minute; skip until next interval.
      return;
    }

    const { featureRows, newOffset } = await page
      .evaluate(async (offset) => {
        if (
          typeof window.__sprt_export_samples !== 'function' ||
          typeof window.__sprt_compute_features !== 'function'
        ) {
          return { featureRows: [], newOffset: offset };
        }
        const rawNew = window.__sprt_export_samples(offset);
        if (!Array.isArray(rawNew) || rawNew.length === 0) {
          return { featureRows: [], newOffset: offset };
        }
        const featureRows = await window.__sprt_compute_features(rawNew);
        return { featureRows: featureRows || [], newOffset: offset + rawNew.length };
      }, sampleOffset)
      .catch(() => ({ featureRows: [], newOffset: sampleOffset }));

    if (!featureRows || !featureRows.length) {
      sampleOffset = newOffset;
      return;
    }

    // Write new feature rows incrementally to the dataset file.
    for (const row of featureRows) {
      // Ensure we add a newline before every row if the file was not empty
      // or we have already written previous rows in this run.
      if (existingCount > 0 || totalNew > 0) {
        stream.write('\n');
      }
      stream.write(JSON.stringify(row));
      totalNew++;
    }

    lastFlush = now;
    sampleOffset = newOffset;
    console.log('[gen-dataset] Incremental flush: wrote', featureRows.length, 'new rows (total this run =', totalNew + ')');
  }

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { status } = await page
      .evaluate(() => {
        const now = Date.now();
        const s =
          typeof window.__sprt_status === 'function'
            ? window.__sprt_status()
            : null;
        return { status: s };
      })
      .catch(() => ({ status: null }));

    const elapsed = Math.floor((Date.now() - start) / 1000);

    if (status) {
      const summary = JSON.stringify(status).slice(0, 200);
      console.log(
        '[gen-dataset] SPRT status:',
        'elapsed=', elapsed + 's,',
        'running=', !!status.running + ',',
        'summary=', summary
      );
      // Periodically flush completed samples while the run is active.
      await flushNewSamples();

      if (!status.running) {
        break;
      }
    } else {
      console.log('[gen-dataset] SPRT status: no window.__sprt_status() available, stopping poll');
      break;
    }

    if (elapsed > maxSeconds) {
      throw new Error('SPRT run timed out after ' + maxSeconds + ' seconds');
    }

    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
  }

  // Final flush for any remaining samples after SPRT stops.
  await flushNewSamples();

  await browser.close();

  return { totalNew };
}

async function main() {
  console.log('[gen-dataset] Starting SPRT helper (node sprt.js)...');
  const sprtProc = spawn('node', ['sprt.js'], {
    cwd: SPRT_DIR,
    stdio: ['inherit', 'pipe', 'inherit'],
    shell: true,
    env: { ...process.env, EVAL_TUNING: '1' },
  });

  try {
    console.log('[gen-dataset] Waiting for sprt.js to report server URL...');
    let webUrl;
    try {
      webUrl = await waitForWebUrlFromSprt(sprtProc);
    } catch (e) {
      console.error('[gen-dataset] Failed to auto-detect server URL from sprt output, falling back to default:', DEFAULT_WEB_URL);
      webUrl = DEFAULT_WEB_URL;
    }

    console.log('[gen-dataset] Waiting for web server at', webUrl);
    await waitForServer(webUrl, 180000);
    console.log('[gen-dataset] Server is up, launching Puppeteer...');

    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }

    // Compute how many rows already exist so we can append instead of overwrite.
    let existingCount = 0;
    if (fs.existsSync(DATA_FILE)) {
      try {
        const existing = fs.readFileSync(DATA_FILE, 'utf8');
        if (existing.trim().length > 0) {
          existingCount = existing.split(/\r?\n/).filter(Boolean).length;
        }
      } catch (e) {
        console.warn('[gen-dataset] Warning: failed to read existing dataset, will append anyway:', e.message);
      }
    }

    console.log('[gen-dataset] Existing rows in', DATA_FILE, ':', existingCount);

    const stream = fs.createWriteStream(DATA_FILE, {
      flags: existingCount > 0 ? 'a' : 'w',
    });

    const { totalNew } = await runPuppeteer(webUrl, existingCount, stream);

    stream.end();
    console.log(
      '[gen-dataset] Finished writing dataset; total rows now ~',
      existingCount + totalNew,
      'at',
      DATA_FILE
    );
  } catch (err) {
    console.error('[gen-dataset] Error:', err && err.message ? err.message : err);
  } finally {
    console.log('[gen-dataset] Stopping SPRT helper...');
    try {
      sprtProc.kill('SIGTERM');
    } catch (e) {
      // ignore
    }
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error('[gen-dataset] Fatal:', err);
    process.exit(1);
  });
}
