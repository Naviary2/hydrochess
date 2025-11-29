import initOld, { Engine as EngineOld } from './pkg-old/hydrochess_wasm.js';
import initNew, { Engine as EngineNew } from './pkg-new/hydrochess_wasm.js';

let wasmReady = false;

function getStandardPosition() {
    const pieces = [];
    pieces.push({ x: '1', y: '1', piece_type: 'r', player: 'w' });
    pieces.push({ x: '2', y: '1', piece_type: 'n', player: 'w' });
    pieces.push({ x: '3', y: '1', piece_type: 'b', player: 'w' });
    pieces.push({ x: '4', y: '1', piece_type: 'q', player: 'w' });
    pieces.push({ x: '5', y: '1', piece_type: 'k', player: 'w' });
    pieces.push({ x: '6', y: '1', piece_type: 'b', player: 'w' });
    pieces.push({ x: '7', y: '1', piece_type: 'n', player: 'w' });
    pieces.push({ x: '8', y: '1', piece_type: 'r', player: 'w' });
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '2', piece_type: 'p', player: 'w' });
    }
    pieces.push({ x: '1', y: '8', piece_type: 'r', player: 'b' });
    pieces.push({ x: '2', y: '8', piece_type: 'n', player: 'b' });
    pieces.push({ x: '3', y: '8', piece_type: 'b', player: 'b' });
    pieces.push({ x: '4', y: '8', piece_type: 'q', player: 'b' });
    pieces.push({ x: '5', y: '8', piece_type: 'k', player: 'b' });
    pieces.push({ x: '6', y: '8', piece_type: 'b', player: 'b' });
    pieces.push({ x: '7', y: '8', piece_type: 'n', player: 'b' });
    pieces.push({ x: '8', y: '8', piece_type: 'r', player: 'b' });
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '7', piece_type: 'p', player: 'b' });
    }
    return {
        board: { pieces },
        turn: 'w',
        castling_rights: [],
        en_passant: null,
        halfmove_clock: 0,
        fullmove_number: 1,
        move_history: []
    };
}

function applyMove(position, move) {
    const pieces = position.board.pieces;
    const [fromX, fromY] = move.from.split(',');
    const [toX, toY] = move.to.split(',');

    const capturedIdx = pieces.findIndex(p => p.x === toX && p.y === toY);
    if (capturedIdx !== -1) {
        pieces.splice(capturedIdx, 1);
    }

    const movingPiece = pieces.find(p => p.x === fromX && p.y === fromY);
    if (!movingPiece) {
        throw new Error('No piece at ' + move.from);
    }
    movingPiece.x = toX;
    movingPiece.y = toY;

    if (move.promotion) {
        movingPiece.piece_type = move.promotion.toLowerCase();
    }

    position.turn = position.turn === 'w' ? 'b' : 'w';
    return position;
}

function isGameOver(position) {
    const kings = position.board.pieces.filter(p => p.piece_type === 'k');
    if (kings.length < 2) {
        return { over: true, reason: 'checkmate' };
    }
    if (position.board.pieces.length <= 2) {
        return { over: true, reason: 'draw' };
    }
    return { over: false };
}

async function ensureInit() {
    if (!wasmReady) {
        await initOld();
        await initNew();
        wasmReady = true;
    }
}

async function playSingleGame(timePerMove, maxMoves, newPlaysWhite, openingMove) {
    let position = getStandardPosition();
    const newColor = newPlaysWhite ? 'w' : 'b';
    const moveLines = [];

    // Apply opening move if provided (always white's first move)
    if (openingMove) {
        moveLines.push('W: ' + openingMove.from + '>' + openingMove.to);
        position = applyMove(position, openingMove);
        // Add to move_history for engine context
        position.move_history = [{
            from: openingMove.from,
            to: openingMove.to,
            promotion: openingMove.promotion || null
        }];
    }

    for (let i = 0; i < maxMoves; i++) {
        const sideToMove = position.turn;
        const isWhiteTurn = sideToMove === 'w';
        const EngineClass = isWhiteTurn
            ? (newPlaysWhite ? EngineNew : EngineOld)
            : (newPlaysWhite ? EngineOld : EngineNew);

        const engine = new EngineClass(position);
        const move = engine.get_best_move_with_time(timePerMove);
        engine.free();

        if (!move || !move.from || !move.to) {
            const result = sideToMove === newColor ? 'loss' : 'win';
            return { result, log: moveLines.join('\n') };
        }

        moveLines.push(
            (sideToMove === 'w' ? 'W' : 'B') + ': ' + move.from + '>' + move.to +
            (move.promotion ? '=' + move.promotion : '')
        );

        try {
            position = applyMove(position, move);
        } catch (e) {
            const result = sideToMove === newColor ? 'loss' : 'win';
            return { result, log: moveLines.join('\n') };
        }

        const gameState = isGameOver(position);
        if (gameState.over) {
            if (gameState.reason === 'draw') {
                return { result: 'draw', log: moveLines.join('\n') };
            }
            const result = sideToMove === newColor ? 'win' : 'loss';
            return { result, log: moveLines.join('\n') };
        }
    }

    return { result: 'draw', log: moveLines.join('\n') };
}

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'runGame') {
        try {
            await ensureInit();
            const { result, log } = await playSingleGame(msg.timePerMove, msg.maxMoves, msg.newPlaysWhite, msg.openingMove);
            self.postMessage({
                type: 'result',
                gameIndex: msg.gameIndex,
                result,
                log,
            });
        } catch (err) {
            self.postMessage({
                type: 'error',
                gameIndex: msg.gameIndex,
                error: err.message || String(err),
            });
        }
    }
};
