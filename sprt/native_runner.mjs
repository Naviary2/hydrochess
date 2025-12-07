/**
 * Native WASM Game Runner for SPSA Testing
 * 
 * This module runs games using Node.js native WASM execution,
 * bypassing Puppeteer/browser overhead for much faster testing.
 * 
 * Falls back to browser-based execution if native doesn't work.
 */

import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readFileSync, existsSync } from 'fs';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Check if we can use native Node.js WASM execution
 */
export function canUseNativeWasm() {
    // Check for nodejs-targeted WASM build
    const wasmPath = join(__dirname, 'pkg-nodejs', 'hydrochess_bg.wasm');
    const jsPath = join(__dirname, 'pkg-nodejs', 'hydrochess.js');

    // Check if files exist
    if (!existsSync(wasmPath) || !existsSync(jsPath)) {
        return false;
    }

    // Check if WebAssembly is available
    if (typeof WebAssembly === 'undefined') {
        return false;
    }

    return true;
}

/**
 * Native WASM Game Runner using Node.js worker threads
 */
export class NativeWasmRunner {
    constructor(options) {
        this.options = options;
        this.workers = [];
        this.readyWorkers = [];
        this.numWorkers = options.concurrency || 8;
    }

    async init() {
        console.log('🚀 Using native Node.js WASM execution');
        console.log(`   Starting ${this.numWorkers} worker threads...`);

        // Create worker threads
        const workerScript = join(__dirname, 'native_worker.mjs');

        const initPromises = [];
        for (let i = 0; i < this.numWorkers; i++) {
            const worker = new Worker(workerScript, {
                workerData: {
                    wasmPath: join(__dirname, 'pkg-nodejs'),
                    workerId: i
                }
            });

            const initPromise = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Worker ${i} init timeout`));
                }, 30000);

                worker.once('message', (msg) => {
                    if (msg.type === 'ready') {
                        clearTimeout(timeout);
                        this.workers.push(worker);
                        this.readyWorkers.push(i);
                        resolve();
                    } else if (msg.type === 'error') {
                        clearTimeout(timeout);
                        reject(new Error(msg.error));
                    }
                });

                worker.once('error', (err) => {
                    clearTimeout(timeout);
                    reject(err);
                });
            });

            initPromises.push(initPromise);
        }

        await Promise.all(initPromises);
        console.log(`✅ ${this.workers.length} workers ready`);
    }

    /**
     * Run games between θ+ and θ- configurations
     */
    async runGames(thetaPlus, thetaMinus, numGames) {
        const results = { plusWins: 0, minusWins: 0, draws: 0 };
        const gameConfigs = [];

        // Create game configs
        for (let i = 0; i < numGames; i++) {
            // θ+ as White
            gameConfigs.push({
                thetaPlus,
                thetaMinus,
                plusPlaysWhite: true,
                timePerMove: this.options.tc,
                maxMoves: 150
            });
            // θ+ as Black
            gameConfigs.push({
                thetaPlus,
                thetaMinus,
                plusPlaysWhite: false,
                timePerMove: this.options.tc,
                maxMoves: 150
            });
        }

        // Shuffle games
        for (let i = gameConfigs.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [gameConfigs[i], gameConfigs[j]] = [gameConfigs[j], gameConfigs[i]];
        }

        // Run games using worker pool
        let completed = 0;
        let gameIdx = 0;
        const activeGames = new Map();

        return new Promise((resolve) => {
            const dispatchGame = (workerIdx) => {
                if (gameIdx >= gameConfigs.length) return false;

                const config = gameConfigs[gameIdx++];
                const worker = this.workers[workerIdx];

                activeGames.set(workerIdx, true);

                worker.postMessage({
                    type: 'runGame',
                    config
                });

                return true;
            };

            const handleResult = (workerIdx, result) => {
                activeGames.delete(workerIdx);
                completed++;

                if (result === 'plus') results.plusWins++;
                else if (result === 'minus') results.minusWins++;
                else results.draws++;

                if (completed % 10 === 0 || completed === gameConfigs.length) {
                    console.log(`   [Native] Games: ${completed}/${gameConfigs.length}`);
                }

                if (completed >= gameConfigs.length) {
                    resolve(results);
                    return;
                }

                // Dispatch next game
                dispatchGame(workerIdx);
            };

            // Set up message handlers
            this.workers.forEach((worker, idx) => {
                worker.on('message', (msg) => {
                    if (msg.type === 'result') {
                        handleResult(idx, msg.result);
                    }
                });
            });

            // Start initial batch
            for (let i = 0; i < this.workers.length; i++) {
                dispatchGame(i);
            }
        });
    }

    async close() {
        for (const worker of this.workers) {
            worker.terminate();
        }
        this.workers = [];
        this.readyWorkers = [];
    }
}

export default NativeWasmRunner;
