#!/usr/bin/env node
/**
 * HydroChess Automated Parameter Tuner
 * 
 * Uses coordinate descent optimization to tune evaluation parameters.
 * Each parameter is adjusted one at a time, tested with SPRT, and kept if improved.
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

// ==================== CONFIGURATION ====================

const CONFIG = {
    // File paths
    evaluationFile: path.join(__dirname, '..', 'src', 'evaluation.rs'),
    baselineDir: path.join(__dirname, 'baseline'),
    testDir: path.join(__dirname, 'test'),
    pkgDir: path.join(__dirname, '..', 'pkg'),
    tunerStateFile: path.join(__dirname, 'tuner-state.json'),
    
    // SPRT settings for tuning (faster, less confident)
    sprt: {
        elo0: 0,
        elo1: 5,      // Smaller bound for faster tests
        alpha: 0.10,  // Higher alpha = faster but less confident
        beta: 0.10,
        maxGames: 2000,  // Limit games per test
    },
    
    // Tuning settings
    tuning: {
        stepSize: 10,         // Initial step size
        minStepSize: 5,       // Minimum step size
        stepDecay: 0.8,       // Decay step size after no improvement
        maxIterations: 100,   // Max iterations per parameter
        rounds: 3,            // Full rounds through all parameters
    },
    
    // Parameters to tune (name, min, max, current regex pattern)
    parameters: [
        // Piece values
        { name: 'PAWN_VALUE', regex: /PieceType::Pawn => (\d+)/, min: 80, max: 120, step: 5, file: 'evaluation.rs' },
        { name: 'KNIGHT_VALUE', regex: /PieceType::Knight => (\d+)/, min: 280, max: 350, step: 10, file: 'evaluation.rs' },
        { name: 'BISHOP_VALUE', regex: /PieceType::Bishop => (\d+)/, min: 300, max: 400, step: 10, file: 'evaluation.rs' },
        { name: 'ROOK_VALUE', regex: /PieceType::Rook => (\d+)/, min: 450, max: 550, step: 10, file: 'evaluation.rs' },
        { name: 'QUEEN_VALUE', regex: /PieceType::Queen => (\d+)/, min: 850, max: 1000, step: 25, file: 'evaluation.rs' },
        
        // Bonuses/Penalties
        { name: 'ROOK_BEHIND_ENEMY_BONUS', regex: /ROOK_BEHIND_ENEMY_BONUS: i32 = (\d+)/, min: 10, max: 50, step: 5, file: 'evaluation.rs' },
        { name: 'QUEEN_BEHIND_ENEMY_BONUS', regex: /QUEEN_BEHIND_ENEMY_BONUS: i32 = (\d+)/, min: 10, max: 50, step: 5, file: 'evaluation.rs' },
        { name: 'PAWN_SHIELD_BONUS', regex: /PAWN_SHIELD_BONUS: i32 = (\d+)/, min: 5, max: 30, step: 5, file: 'evaluation.rs' },
        { name: 'KNIGHT_CENTRALITY_BONUS', regex: /KNIGHT_CENTRALITY_BONUS: i32 = (\d+)/, min: 5, max: 25, step: 5, file: 'evaluation.rs' },
        { name: 'BISHOP_PAIR_BONUS', regex: /BISHOP_PAIR_BONUS: i32 = (\d+)/, min: 20, max: 60, step: 5, file: 'evaluation.rs' },
        { name: 'ROOK_OPEN_FILE_BONUS', regex: /ROOK_OPEN_FILE_BONUS: i32 = (\d+)/, min: 15, max: 40, step: 5, file: 'evaluation.rs' },
        { name: 'ROOK_SEMI_OPEN_BONUS', regex: /ROOK_SEMI_OPEN_BONUS: i32 = (\d+)/, min: 5, max: 25, step: 5, file: 'evaluation.rs' },
        { name: 'PASSED_PAWN_BONUS', regex: /PASSED_PAWN_BONUS: i32 = (\d+)/, min: 10, max: 40, step: 5, file: 'evaluation.rs' },
        { name: 'DOUBLED_PAWN_PENALTY', regex: /DOUBLED_PAWN_PENALTY: i32 = (\d+)/, min: 5, max: 30, step: 5, file: 'evaluation.rs' },
        { name: 'ISOLATED_PAWN_PENALTY', regex: /ISOLATED_PAWN_PENALTY: i32 = (\d+)/, min: 5, max: 25, step: 5, file: 'evaluation.rs' },
        { name: 'KING_TROPISM_BONUS', regex: /KING_TROPISM_BONUS: i32 = (\d+)/, min: 1, max: 10, step: 1, file: 'evaluation.rs' },
    ],
};

// ==================== TERMINAL OUTPUT ====================

const COLORS = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    dim: '\x1b[2m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    white: '\x1b[37m',
    bgGreen: '\x1b[42m',
    bgRed: '\x1b[41m',
    bgYellow: '\x1b[43m',
};

function printHeader() {
    console.log(`
${COLORS.magenta}╔══════════════════════════════════════════════════════════════════╗
║${COLORS.bright}${COLORS.white}     🎯 HydroChess Automated Parameter Tuner                      ${COLORS.reset}${COLORS.magenta}║
║${COLORS.dim}     Coordinate Descent Optimization with SPRT Testing            ${COLORS.reset}${COLORS.magenta}║
╚══════════════════════════════════════════════════════════════════╝${COLORS.reset}
`);
}

function printParameterStatus(param, oldValue, newValue, result) {
    const arrow = result === 'improved' ? `${COLORS.green}↑` : 
                  result === 'regressed' ? `${COLORS.red}↓` : `${COLORS.yellow}=`;
    const status = result === 'improved' ? `${COLORS.bgGreen}${COLORS.white} IMPROVED ${COLORS.reset}` :
                   result === 'regressed' ? `${COLORS.bgRed}${COLORS.white} REGRESSED ${COLORS.reset}` :
                   `${COLORS.bgYellow}${COLORS.white} NO CHANGE ${COLORS.reset}`;
    
    console.log(`  ${param.name}: ${oldValue} ${arrow} ${newValue}${COLORS.reset} ${status}`);
}

// ==================== FILE MANIPULATION ====================

function readEvaluationFile() {
    return fs.readFileSync(CONFIG.evaluationFile, 'utf8');
}

function writeEvaluationFile(content) {
    fs.writeFileSync(CONFIG.evaluationFile, content);
}

function getCurrentValue(param) {
    const content = readEvaluationFile();
    const match = content.match(param.regex);
    if (match) {
        return parseInt(match[1]);
    }
    throw new Error(`Could not find parameter ${param.name}`);
}

function setParameterValue(param, value) {
    let content = readEvaluationFile();
    const match = content.match(param.regex);
    if (!match) {
        throw new Error(`Could not find parameter ${param.name}`);
    }
    
    // Replace the value in the match
    const oldStr = match[0];
    const newStr = oldStr.replace(match[1], String(value));
    content = content.replace(oldStr, newStr);
    
    writeEvaluationFile(content);
}

// ==================== BUILD MANAGEMENT ====================

function buildEngine() {
    console.log(`${COLORS.dim}Building engine...${COLORS.reset}`);
    try {
        execSync('wasm-pack build --target web', {
            cwd: path.join(__dirname, '..'),
            stdio: 'pipe',
        });
        return true;
    } catch (e) {
        console.error(`${COLORS.red}Build failed!${COLORS.reset}`);
        return false;
    }
}

function copyDirectory(src, dest) {
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

function saveBaseline() {
    if (!fs.existsSync(CONFIG.pkgDir)) {
        console.error(`${COLORS.red}No pkg directory. Build first.${COLORS.reset}`);
        return false;
    }
    copyDirectory(CONFIG.pkgDir, CONFIG.baselineDir);
    return true;
}

function saveTestBuild() {
    if (!fs.existsSync(CONFIG.pkgDir)) {
        return false;
    }
    copyDirectory(CONFIG.pkgDir, CONFIG.testDir);
    return true;
}

// ==================== SPRT TESTING ====================

/**
 * Run a quick SPRT test and return result
 * Returns: 'passed' | 'failed' | 'inconclusive'
 */
function runQuickSPRT() {
    // Run the SPRT script with tuning configuration
    const sprtScript = path.join(__dirname, 'sprt-quick.js');
    
    try {
        const result = execSync(`node "${sprtScript}"`, {
            cwd: __dirname,
            encoding: 'utf8',
            timeout: 120000, // 2 minute timeout
        });
        
        if (result.includes('PASSED')) return 'passed';
        if (result.includes('FAILED')) return 'failed';
        return 'inconclusive';
    } catch (e) {
        return 'inconclusive';
    }
}

// ==================== TUNING STATE ====================

function loadTunerState() {
    if (fs.existsSync(CONFIG.tunerStateFile)) {
        return JSON.parse(fs.readFileSync(CONFIG.tunerStateFile, 'utf8'));
    }
    return {
        currentRound: 0,
        currentParamIndex: 0,
        bestValues: {},
        history: [],
    };
}

function saveTunerState(state) {
    fs.writeFileSync(CONFIG.tunerStateFile, JSON.stringify(state, null, 2));
}

// ==================== COORDINATE DESCENT ====================

async function tuneParameter(param, state) {
    const currentValue = getCurrentValue(param);
    let bestValue = currentValue;
    let improved = false;
    
    console.log(`\n${COLORS.cyan}Tuning ${param.name}${COLORS.reset} (current: ${currentValue}, range: [${param.min}, ${param.max}])`);
    
    // Try increasing
    const upValue = Math.min(param.max, currentValue + param.step);
    if (upValue !== currentValue) {
        console.log(`  Testing ${currentValue} → ${upValue}...`);
        setParameterValue(param, upValue);
        
        if (buildEngine() && saveTestBuild()) {
            const result = runQuickSPRT();
            if (result === 'passed') {
                console.log(`  ${COLORS.green}✓ Improvement found!${COLORS.reset}`);
                bestValue = upValue;
                improved = true;
                saveBaseline(); // New baseline
            } else {
                console.log(`  ${COLORS.dim}No improvement (${result})${COLORS.reset}`);
            }
        }
    }
    
    // If no improvement up, try decreasing
    if (!improved) {
        setParameterValue(param, currentValue); // Reset
        const downValue = Math.max(param.min, currentValue - param.step);
        
        if (downValue !== currentValue) {
            console.log(`  Testing ${currentValue} → ${downValue}...`);
            setParameterValue(param, downValue);
            
            if (buildEngine() && saveTestBuild()) {
                const result = runQuickSPRT();
                if (result === 'passed') {
                    console.log(`  ${COLORS.green}✓ Improvement found!${COLORS.reset}`);
                    bestValue = downValue;
                    improved = true;
                    saveBaseline(); // New baseline
                } else {
                    console.log(`  ${COLORS.dim}No improvement (${result})${COLORS.reset}`);
                    setParameterValue(param, currentValue); // Reset to original
                }
            }
        }
    }
    
    state.bestValues[param.name] = bestValue;
    state.history.push({
        param: param.name,
        before: currentValue,
        after: bestValue,
        improved,
        timestamp: new Date().toISOString(),
    });
    
    return improved;
}

async function runTuning() {
    printHeader();
    
    // Ensure baseline exists
    if (!fs.existsSync(path.join(CONFIG.baselineDir, 'hydrochess_wasm_bg.wasm'))) {
        console.log(`${COLORS.yellow}No baseline found. Building and saving baseline...${COLORS.reset}`);
        if (!buildEngine() || !saveBaseline()) {
            console.error(`${COLORS.red}Failed to create baseline.${COLORS.reset}`);
            return;
        }
        console.log(`${COLORS.green}Baseline created!${COLORS.reset}`);
    }
    
    const state = loadTunerState();
    
    console.log(`${COLORS.bright}Starting tuning round ${state.currentRound + 1}/${CONFIG.tuning.rounds}${COLORS.reset}`);
    console.log(`Parameters to tune: ${CONFIG.parameters.length}`);
    
    let totalImprovements = 0;
    
    for (let round = state.currentRound; round < CONFIG.tuning.rounds; round++) {
        console.log(`\n${COLORS.bright}═══ Round ${round + 1}/${CONFIG.tuning.rounds} ═══${COLORS.reset}`);
        
        let roundImprovements = 0;
        
        for (let i = state.currentParamIndex; i < CONFIG.parameters.length; i++) {
            const param = CONFIG.parameters[i];
            
            const improved = await tuneParameter(param, state);
            if (improved) {
                roundImprovements++;
                totalImprovements++;
            }
            
            state.currentParamIndex = i + 1;
            saveTunerState(state);
        }
        
        console.log(`\n${COLORS.cyan}Round ${round + 1} complete: ${roundImprovements} improvements${COLORS.reset}`);
        
        // Reset for next round
        state.currentRound = round + 1;
        state.currentParamIndex = 0;
        saveTunerState(state);
        
        // If no improvements in this round, stop early
        if (roundImprovements === 0 && round > 0) {
            console.log(`${COLORS.yellow}No improvements in this round. Stopping early.${COLORS.reset}`);
            break;
        }
    }
    
    // Print final summary
    console.log(`
${COLORS.magenta}═══════════════════════════════════════════════════════════════════${COLORS.reset}
${COLORS.bright}Tuning Complete!${COLORS.reset}
  Total Improvements: ${totalImprovements}
  
${COLORS.bright}Final Values:${COLORS.reset}`);
    
    for (const param of CONFIG.parameters) {
        const value = getCurrentValue(param);
        console.log(`  ${param.name}: ${value}`);
    }
    
    console.log(`${COLORS.magenta}═══════════════════════════════════════════════════════════════════${COLORS.reset}`);
}

// ==================== CLI ====================

const args = process.argv.slice(2);
const command = args[0];

switch (command) {
    case 'run':
    case undefined:
        runTuning();
        break;
        
    case 'reset':
        if (fs.existsSync(CONFIG.tunerStateFile)) {
            fs.unlinkSync(CONFIG.tunerStateFile);
            console.log('Tuner state reset.');
        }
        break;
        
    case 'status':
        const state = loadTunerState();
        console.log('Current tuner state:');
        console.log(JSON.stringify(state, null, 2));
        break;
        
    case 'help':
    default:
        printHeader();
        console.log(`${COLORS.bright}Usage:${COLORS.reset}
  node tuner.js [command]

${COLORS.bright}Commands:${COLORS.reset}
  run      Start/resume tuning (default)
  reset    Reset tuner state
  status   Show current tuner state
  help     Show this help
`);
        break;
}
