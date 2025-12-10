// Palace Variant Evaluation
//
// Strategy: HYPER-AGGRESSIVE heavy pieces on top of solid fundamentals.
// 1. Base evaluation provides material, king safety, and structure.
// 2. Palace bonuses reward heavy piece advancement and enemy targeting.

use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::evaluation::base::{self, is_clear_line_between};
use crate::game::GameState;

// ==================== Constants ====================

// Bonuses
const HEAVY_PIECE_ADVANCE_BONUS: i32 = 30; // Reduced slightly as base eval handles some activity
const VISION_BONUS: i32 = 10; // Per enemy piece "seen"
const AMAZON_AGGRESSION_BONUS: i32 = 40;

// Piece Types considered "Heavy" for this variant
fn is_heavy_piece(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Queen
            | PieceType::Amazon
            | PieceType::RoyalQueen
            | PieceType::Chancellor
            | PieceType::Archbishop
    )
}

pub fn evaluate(game: &GameState) -> i32 {
    // 1. Start with the base evaluation.
    // base::evaluate returns score from the current player's perspective.
    // We need ABSOLUTE score (White - Black) to apply our symmetric logic correctly.
    let base_eval = base::evaluate(game);
    let mut score = if game.turn == PlayerColor::Black {
        -base_eval
    } else {
        base_eval
    };

    // 2. Add Palace-specific Aggression Bonuses

    // Collect piece lists efficiently
    let mut white_heavy: Vec<(Coordinate, PieceType)> = Vec::new();
    let mut black_heavy: Vec<(Coordinate, PieceType)> = Vec::new();
    let mut white_targets: Vec<Coordinate> = Vec::new();
    let mut black_targets: Vec<Coordinate> = Vec::new();

    for ((x, y), piece) in &game.board.pieces {
        let coord = Coordinate::new(*x, *y);
        if piece.color() == PlayerColor::White {
            white_targets.push(coord);
            if is_heavy_piece(piece.piece_type()) {
                white_heavy.push((coord, piece.piece_type()));
            }
        } else if piece.color() == PlayerColor::Black {
            black_targets.push(coord);
            if is_heavy_piece(piece.piece_type()) {
                black_heavy.push((coord, piece.piece_type()));
            }
        }
    }

    // Evaluate White Heavy Pieces
    for (pos, ptype) in &white_heavy {
        // 1. Advancement (Get them OUT!)
        // Assuming White starts at y=0/1 and moves towards positive Y
        if pos.y >= 3 {
            score += HEAVY_PIECE_ADVANCE_BONUS;
            if pos.y >= 6 {
                score += HEAVY_PIECE_ADVANCE_BONUS / 2; // Extra deep
            }
        }

        // 2. Vision (See enemy targets)
        for target in &black_targets {
            if can_attack(game, *pos, *target, *ptype) {
                score += VISION_BONUS;
            }
        }

        // 3. Amazon Special Aggression
        if *ptype == PieceType::Amazon {
            // Bonus for being central and advanced
            if pos.y >= 4 && pos.x >= 3 && pos.x <= 6 {
                score += AMAZON_AGGRESSION_BONUS;
            }
        }
    }

    // Evaluate Black Heavy Pieces
    for (pos, ptype) in &black_heavy {
        // 1. Advancement (Get them OUT!)
        // Standard setup is White Y=1, Black Y=8.
        // So Black advancing means Y <= 6.
        if pos.y <= 6 {
            score -= HEAVY_PIECE_ADVANCE_BONUS;
            if pos.y <= 3 {
                score -= HEAVY_PIECE_ADVANCE_BONUS / 2;
            }
        }

        // 2. Vision
        for target in &white_targets {
            if can_attack(game, *pos, *target, *ptype) {
                score -= VISION_BONUS;
            }
        }

        // 3. Amazon Special Aggression
        if *ptype == PieceType::Amazon {
            if pos.y <= 5 && pos.x >= 3 && pos.x <= 6 {
                score -= AMAZON_AGGRESSION_BONUS;
            }
        }
    }

    // Return perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

/// Check if piece at 'from' *could* attack 'to' (pseudo-legal check + blockage)
fn can_attack(game: &GameState, from: Coordinate, to: Coordinate, ptype: PieceType) -> bool {
    let dx = (to.x - from.x).abs();
    let dy = (to.y - from.y).abs();

    let is_diagonal = dx == dy;
    let is_orthogonal = dx == 0 || dy == 0;
    let is_knight_jump = (dx == 1 && dy == 2) || (dx == 2 && dy == 1);

    let can_move_diagonal = matches!(
        ptype,
        PieceType::Queen
            | PieceType::Amazon
            | PieceType::Bishop
            | PieceType::RoyalQueen
            | PieceType::Archbishop
    );
    let can_move_orthogonal = matches!(
        ptype,
        PieceType::Queen
            | PieceType::Amazon
            | PieceType::Rook
            | PieceType::RoyalQueen
            | PieceType::Chancellor
    );
    let can_move_knight = matches!(
        ptype,
        PieceType::Amazon | PieceType::Knight | PieceType::Chancellor | PieceType::Archbishop
    );

    if is_knight_jump {
        return can_move_knight; // Knights don't need line of sight
    }

    let valid_geometry =
        (is_diagonal && can_move_diagonal) || (is_orthogonal && can_move_orthogonal);

    if !valid_geometry {
        return false;
    }

    // Check for blockage for sliders
    if is_clear_line_between(&game.board, &from, &to) {
        return true;
    }

    false
}
