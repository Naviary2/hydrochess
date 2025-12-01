#!/usr/bin/env node

// Apply tuned Texel parameters from sprt/data/eval_params_tuned.json
// into src/evaluation.rs by regex-replacing the matching const values.

const fs = require('fs');
const path = require('path');

const SPRT_DIR = __dirname;
const ROOT_DIR = path.join(SPRT_DIR, '..');
const DATA_FILE = path.join(SPRT_DIR, 'data', 'eval_params_tuned.json');
const EVAL_FILE = path.join(ROOT_DIR, 'src', 'evaluation.rs');

function toConstName(paramName) {
  // king_ring_pawn_bonus -> KING_RING_PAWN_BONUS
  return paramName.toUpperCase().replace(/[^A-Z0-9]+/g, '_');
}

function loadTunedParams() {
  if (!fs.existsSync(DATA_FILE)) {
    throw new Error('Tuned params file not found: ' + DATA_FILE);
  }
  const raw = fs.readFileSync(DATA_FILE, 'utf8');
  const obj = JSON.parse(raw);
  // File has shape { params: { ... }, negLogLikelihood, samples, timestamp }
  if (obj && typeof obj.params === 'object' && obj.params !== null) {
    return obj.params;
  }
  return obj;
}

function applyParamsToSource(src, params) {
  let changedCount = 0;

  for (const [name, value] of Object.entries(params)) {
    const constName = toConstName(name);
    // Match lines like: const KING_RING_PAWN_BONUS: i32 = 30;
    const re = new RegExp(`(const\\s+${constName}\\s*:\\s*i32\\s*=\\s*)(-?\\d+)(;)`);

    if (!re.test(src)) {
      // Constant not present in evaluation.rs; skip silently.
      continue;
    }

    src = src.replace(re, (match, prefix, oldVal, suffix) => {
      const newVal = String(value | 0); // ensure integer
      if (oldVal === newVal) {
        return match; // no change
      }
      changedCount++;
      console.log(`[apply-tuned] ${constName}: ${oldVal} -> ${newVal}`);
      return `${prefix}${newVal}${suffix}`;
    });
  }

  return { src, changedCount };
}

function main() {
  console.log('[apply-tuned] Loading tuned parameters from', DATA_FILE);
  const params = loadTunedParams();

  console.log('[apply-tuned] Reading evaluation constants from', EVAL_FILE);
  const original = fs.readFileSync(EVAL_FILE, 'utf8');

  const { src: updated, changedCount } = applyParamsToSource(original, params);

  if (changedCount === 0) {
    console.log('[apply-tuned] No matching consts were updated.');
  } else {
    fs.writeFileSync(EVAL_FILE, updated, 'utf8');
    console.log(`[apply-tuned] Updated ${changedCount} constants in ${EVAL_FILE}`);
  }
}

if (require.main === module) {
  try {
    main();
  } catch (err) {
    console.error('[apply-tuned] Fatal:', err);
    process.exit(1);
  }
}
