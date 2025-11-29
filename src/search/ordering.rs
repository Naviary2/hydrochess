use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::Move;

use super::Searcher;

// Move ordering helpers
pub fn sort_moves(searcher: &Searcher, game: &GameState, moves: &mut Vec<Move>, ply: usize, tt_move: &Option<Move>) {
    moves.sort_by_cached_key(|m| {
        let mut score = 0;
        
        // TT move gets highest priority
        if let Some(ttm) = tt_move {
            if m.from == ttm.from && m.to == ttm.to {
                return -20000;
            }
        }
        
        // Captures (MVV-LVA)
        if let Some(target) = game.board.get_piece(&m.to.x, &m.to.y) {
            score -= get_piece_value(target.piece_type) * 10 - get_piece_value(m.piece.piece_type);
        }
        
        // Killer moves
        if searcher.killers[ply][0].as_ref().map_or(false, |k| m.from == k.from && m.to == k.to) {
            score -= 9000;
        } else if searcher.killers[ply][1].as_ref().map_or(false, |k| m.from == k.from && m.to == k.to) {
            score -= 8000;
        }
        
        // History heuristic
        let idx = hash_move_dest(m);
        let history_score = searcher.history[m.piece.piece_type as usize][idx];
        score -= history_score;
        
        score
    });
}

pub fn sort_moves_root(searcher: &Searcher, game: &GameState, moves: &mut Vec<Move>, tt_move: &Option<Move>) {
    sort_moves(searcher, game, moves, 0, tt_move);
}

pub fn sort_captures(game: &GameState, moves: &mut Vec<Move>) {
    moves.sort_by_cached_key(|m| {
        let mut score = 0;
        if let Some(target) = game.board.get_piece(&m.to.x, &m.to.y) {
            score -= get_piece_value(target.piece_type) * 10 - get_piece_value(m.piece.piece_type);
        }
        score
    });
}

#[inline]
pub fn hash_move_dest(m: &Move) -> usize {
    ((m.to.x ^ m.to.y) & 0xFF) as usize
}
