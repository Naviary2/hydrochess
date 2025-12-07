/**
 * Native WASM Worker for SPSA Testing
 * 
 * This worker runs in a Node.js worker thread and executes
 * chess games using the WASM engine directly.
 * 
 * Uses nodejs-targeted wasm-pack build for native Node.js execution.
 */

import { parentPort, workerData } from 'worker_threads';
import { join } from 'path';
import { readFileSync } from 'fs';
import { pathToFileURL } from 'url';

let Engine = null;
let wasmModule = null;
let initialized = false;

/**
 * Initialize the WASM module
 */
async function init() {
    try {
        const pkgPath = workerData.wasmPath;

        // Import the nodejs-targeted wasm module using file:// URL
        const jsPath = pathToFileURL(join(pkgPath, 'hydrochess.js')).href;
        wasmModule = await import(jsPath);

        // The nodejs target exports Engine directly after require()
        Engine = wasmModule.Engine;

        if (!Engine) {
            throw new Error('Engine class not found in WASM module');
        }

        initialized = true;
        parentPort.postMessage({ type: 'ready' });
    } catch (error) {
        parentPort.postMessage({ type: 'error', error: error.message + '\n' + error.stack });
    }
}

/**
 * Clone a position object
 */
function clonePosition(pos) {
    return JSON.parse(JSON.stringify(pos));
}

/**
 * Get starting position for Classical variant
 */
function getClassicalPosition() {
    const pieces = [];

    // White pieces
    pieces.push({ x: '1', y: '1', piece_type: 'r', color: 'w' });
    pieces.push({ x: '2', y: '1', piece_type: 'n', color: 'w' });
    pieces.push({ x: '3', y: '1', piece_type: 'b', color: 'w' });
    pieces.push({ x: '4', y: '1', piece_type: 'q', color: 'w' });
    pieces.push({ x: '5', y: '1', piece_type: 'k', color: 'w' });
    pieces.push({ x: '6', y: '1', piece_type: 'b', color: 'w' });
    pieces.push({ x: '7', y: '1', piece_type: 'n', color: 'w' });
    pieces.push({ x: '8', y: '1', piece_type: 'r', color: 'w' });
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '2', piece_type: 'p', color: 'w' });
    }

    // Black pieces
    pieces.push({ x: '1', y: '8', piece_type: 'r', color: 'b' });
    pieces.push({ x: '2', y: '8', piece_type: 'n', color: 'b' });
    pieces.push({ x: '3', y: '8', piece_type: 'b', color: 'b' });
    pieces.push({ x: '4', y: '8', piece_type: 'q', color: 'b' });
    pieces.push({ x: '5', y: '8', piece_type: 'k', color: 'b' });
    pieces.push({ x: '6', y: '8', piece_type: 'b', color: 'b' });
    pieces.push({ x: '7', y: '8', piece_type: 'n', color: 'b' });
    pieces.push({ x: '8', y: '8', piece_type: 'r', color: 'b' });
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '7', piece_type: 'p', color: 'b' });
    }

    return {
        board: { pieces },
        turn: 'w',
        special_rights: [
            '1,1', '5,1', '8,1', // White king and rooks
            '1,8', '5,8', '8,8', // Black king and rooks
            // Pawns with first-move rights
            '1,2', '2,2', '3,2', '4,2', '5,2', '6,2', '7,2', '8,2',
            '1,7', '2,7', '3,7', '4,7', '5,7', '6,7', '7,7', '8,7'
        ],
        fullmove_number: 1,
        gameRules: {
            promotionRanks: { white: [8], black: [1] }
        }
    };
}

/**
 * Play a single game between two parameter sets
 */
async function playSingleGame(config) {
    const { thetaPlus, thetaMinus, plusPlaysWhite, timePerMove, maxMoves } = config;

    const startPosition = getClassicalPosition();
    let position = clonePosition(startPosition);
    const moveHistory = [];

    for (let i = 0; i < maxMoves; i++) {
        const isWhiteTurn = position.turn === 'w';
        const isPlusTurn = isWhiteTurn === plusPlaysWhite;
        const paramsToUse = isPlusTurn ? thetaPlus : thetaMinus;

        // Create engine with full game state
        const gameInput = clonePosition(startPosition);
        gameInput.move_history = moveHistory.slice();

        const engine = new Engine(gameInput);

        // Apply search params if engine supports it
        if (typeof engine.set_search_params === 'function') {
            engine.set_search_params(JSON.stringify(paramsToUse));
        }

        const move = engine.get_best_move_with_time(timePerMove, true);
        engine.free();

        if (!move || !move.from || !move.to) {
            // No legal moves - loss for side to move
            const winner = isWhiteTurn ? 'black' : 'white';
            return plusPlaysWhite === (winner === 'white') ? 'minus' : 'plus';
        }

        // Apply move to position
        const [fromX, fromY] = move.from.split(',').map(Number);
        const [toX, toY] = move.to.split(',').map(Number);

        // Find and move piece
        const pieceIdx = position.board.pieces.findIndex(
            p => Number(p.x) === fromX && Number(p.y) === fromY
        );
        if (pieceIdx === -1) {
            return 'draw'; // Error case
        }

        // Remove captured piece if any
        const capIdx = position.board.pieces.findIndex(
            p => Number(p.x) === toX && Number(p.y) === toY
        );
        if (capIdx !== -1 && capIdx !== pieceIdx) {
            position.board.pieces.splice(capIdx, 1);
        }

        // Update moved piece position
        const piece = position.board.pieces.find(
            p => Number(p.x) === fromX && Number(p.y) === fromY
        );
        if (piece) {
            piece.x = String(toX);
            piece.y = String(toY);

            // Handle promotion
            if (move.promotion) {
                piece.piece_type = move.promotion;
            }
        }

        // Record move
        moveHistory.push({ from: move.from, to: move.to, promotion: move.promotion });

        // Switch turn
        position.turn = position.turn === 'w' ? 'b' : 'w';

        // Check for insufficient material
        if (position.board.pieces.length <= 2) {
            return 'draw';
        }
    }

    // Max moves reached
    return 'draw';
}

// Handle messages from main thread
parentPort.on('message', async (msg) => {
    if (msg.type === 'runGame') {
        try {
            const result = await playSingleGame(msg.config);
            parentPort.postMessage({ type: 'result', result });
        } catch (error) {
            parentPort.postMessage({ type: 'error', error: error.message });
        }
    }
});

// Initialize on startup
init().catch(err => {
    parentPort.postMessage({ type: 'error', error: err.message });
});
