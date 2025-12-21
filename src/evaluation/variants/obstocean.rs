// Obstocean Variant Evaluator - SUPER SIMPLE
//
// RULE: Push edge pawns (x=1,8). Nothing else matters as much.

use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::evaluation::base;
use crate::game::GameState;

#[inline]
pub fn evaluate(game: &GameState) -> i32 {
    match crate::evaluation::insufficient_material::evaluate_insufficient_material(game) {
        Some(0) => return 0,
        Some(d) => return evaluate_inner(game) / d,
        None => {}
    }
    evaluate_inner(game)
}

#[inline]
fn evaluate_inner(game: &GameState) -> i32 {
    let mut score = game.material_score;
    let (wk, bk) = (game.white_king_pos, game.black_king_pos);

    // Mop-up
    if let Some(_) = crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::Black) {
        if let (Some(w), Some(b)) = (&wk, &bk) {
            score += crate::evaluation::mop_up::evaluate_mop_up_scaled(
                game,
                w,
                b,
                PlayerColor::White,
                PlayerColor::Black,
            );
        }
    } else if let Some(_) =
        crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::White)
    {
        if let (Some(w), Some(b)) = (&wk, &bk) {
            score -= crate::evaluation::mop_up::evaluate_mop_up_scaled(
                game,
                b,
                w,
                PlayerColor::Black,
                PlayerColor::White,
            );
        }
    }

    // Piece evaluation (minimal for non-pawns)
    for ((x, y), p) in game.board.iter() {
        if p.color() == PlayerColor::Neutral {
            continue;
        }

        let v = match p.piece_type() {
            PieceType::Pawn => eval_pawn(*x, *y, p.color(), game),
            PieceType::Rook | PieceType::Chancellor | PieceType::Amazon => {
                base::evaluate_rook(game, *x, *y, p.color(), &wk, &bk)
            }
            PieceType::Queen | PieceType::RoyalQueen => {
                base::evaluate_queen(game, *x, *y, p.color(), &wk, &bk)
            }
            PieceType::Bishop => base::evaluate_bishop(game, *x, *y, p.color(), &wk, &bk),
            _ => 0, // Knights get NO bonus - just material value
        };

        if p.color() == PlayerColor::White {
            score += v;
        } else {
            score -= v;
        }
    }

    score += base::evaluate_king_safety(game, &wk, &bk);
    score += race_eval(game);

    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

/// Pawn eval: HUGE bonus for edge/outside, penalty for center
#[inline]
fn eval_pawn(x: i64, y: i64, color: PlayerColor, game: &GameState) -> i32 {
    let dist = if color == PlayerColor::White {
        (game.white_promo_rank - y).max(0)
    } else {
        (y - game.black_promo_rank).max(0)
    };

    let mut b: i32 = 0;

    // Advancement: 10cp per rank (max 80cp)
    b += (8 - dist.min(8)) as i32 * 10;

    // LANE BONUS (the whole point)
    if x < 1 {
        // LEFT OUTSIDE: 100 + 15 per file out
        b += 100 + ((1 - x) as i32 * 15).min(90);
    } else if x > 8 {
        // RIGHT OUTSIDE: 100 + 15 per file out
        b += 100 + ((x - 8) as i32 * 15).min(90);
    } else if x == 1 || x == 8 {
        // EDGE FILES: Strong priority
        b += 80;
    } else if x == 2 || x == 7 {
        // NEAR EDGE
        b += 25;
    } else {
        // CENTER (x=3,4,5,6): PENALTY
        b -= 40;
    }

    b
}

/// Race evaluation: Who's closest to promoting on edge/outside?
fn race_eval(game: &GameState) -> i32 {
    let mut w_min: i64 = 100;
    let mut b_min: i64 = 100;

    for ((x, y), p) in game.board.iter() {
        if p.piece_type() != PieceType::Pawn {
            continue;
        }

        // Only x<=1 or x>=8 count as racers
        if *x > 1 && *x < 8 {
            continue;
        }

        if p.color() == PlayerColor::White {
            let d = (game.white_promo_rank - y).max(0);
            if d < w_min {
                w_min = d;
            }
        } else if p.color() == PlayerColor::Black {
            let d = (y - game.black_promo_rank).max(0);
            if d < b_min {
                b_min = d;
            }
        }
    }

    let mut s: i32 = 0;

    // Race comparison: 100cp per move, max 500cp
    if w_min < 100 && b_min < 100 {
        let diff = b_min - w_min;
        s += (diff as i32 * 100).clamp(-500, 500);
    } else if w_min < 100 {
        s += (10 - w_min).max(0) as i32 * 40;
    } else if b_min < 100 {
        s -= (10 - b_min).max(0) as i32 * 40;
    }

    s
}
