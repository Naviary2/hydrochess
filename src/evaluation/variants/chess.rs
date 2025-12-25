// Chess Variant Evaluation (Standard 8x8 Chess)
//
// Optimized HCE for classical chess. Uses piece values, PSTs, and key positional factors.
// PST format: index = (rank-1)*8 + (file-1), so a1=0, h1=7, a2=8, ..., h8=63

use crate::board::{Board, PieceType, PlayerColor};
use crate::game::GameState;

// ==================== Piece Values ====================
// Standard values in centipawns

const PAWN_VALUE: i32 = 100;
const KNIGHT_VALUE: i32 = 320;
const BISHOP_VALUE: i32 = 330;
const ROOK_VALUE: i32 = 500;
const QUEEN_VALUE: i32 = 900;

// ==================== Piece-Square Tables ====================
// These are from WHITE's perspective. Index = (rank-1)*8 + (file-1)
// For black, we mirror vertically (index XOR 56) and negate the value.

// Pawn PST - encourage central control and advancement
// Rank 1 is indices 0-7, Rank 8 is indices 56-63
#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
    //  a     b     c     d     e     f     g     h
        0,    0,    0,    0,    0,    0,    0,    0,   // rank 1 (no pawns here)
        5,   10,   10,  -20,  -20,   10,   10,    5,   // rank 2 - slight penalty for d/e pawns staying put
        5,   -5,  -10,    0,    0,  -10,   -5,    5,   // rank 3
        0,    0,    0,   20,   20,    0,    0,    0,   // rank 4 - bonus for central pawns
        5,    5,   10,   25,   25,   10,    5,    5,   // rank 5 - bigger central bonus
       10,   10,   20,   30,   30,   20,   10,   10,   // rank 6 - approaching promotion
       50,   50,   50,   50,   50,   50,   50,   50,   // rank 7 - about to promote
        0,    0,    0,    0,    0,    0,    0,    0,   // rank 8 (promoted, not pawn anymore)
];

// Knight PST - centralization is key, edges are bad
#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
    //  a     b     c     d     e     f     g     h
      -50,  -40,  -30,  -30,  -30,  -30,  -40,  -50,   // rank 1
      -40,  -20,    0,    5,    5,    0,  -20,  -40,   // rank 2
      -30,    5,   10,   15,   15,   10,    5,  -30,   // rank 3
      -30,    0,   15,   20,   20,   15,    0,  -30,   // rank 4
      -30,    5,   15,   20,   20,   15,    5,  -30,   // rank 5
      -30,    0,   10,   15,   15,   10,    0,  -30,   // rank 6
      -40,  -20,    0,    0,    0,    0,  -20,  -40,   // rank 7
      -50,  -40,  -30,  -30,  -30,  -30,  -40,  -50,   // rank 8
];

// Bishop PST - diagonals and center, avoid corners
#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
    //  a     b     c     d     e     f     g     h
      -20,  -10,  -10,  -10,  -10,  -10,  -10,  -20,   // rank 1
      -10,    5,    0,    0,    0,    0,    5,  -10,   // rank 2 - slight bonus for fianchetto
      -10,   10,   10,   10,   10,   10,   10,  -10,   // rank 3
      -10,    0,   10,   10,   10,   10,    0,  -10,   // rank 4
      -10,    5,    5,   10,   10,    5,    5,  -10,   // rank 5
      -10,    0,    5,   10,   10,    5,    0,  -10,   // rank 6
      -10,    0,    0,    0,    0,    0,    0,  -10,   // rank 7
      -20,  -10,  -10,  -10,  -10,  -10,  -10,  -20,   // rank 8
];

// Rook PST - 7th rank is great, open files (handled separately)
#[rustfmt::skip]
const ROOK_PST: [i32; 64] = [
    //  a     b     c     d     e     f     g     h
        0,    0,    5,   10,   10,    5,    0,    0,   // rank 1 - central files slightly better
       -5,    0,    0,    0,    0,    0,    0,   -5,   // rank 2
       -5,    0,    0,    0,    0,    0,    0,   -5,   // rank 3
       -5,    0,    0,    0,    0,    0,    0,   -5,   // rank 4
       -5,    0,    0,    0,    0,    0,    0,   -5,   // rank 5
       -5,    0,    0,    0,    0,    0,    0,   -5,   // rank 6
        5,   10,   10,   10,   10,   10,   10,    5,   // rank 7 - 7th rank bonus
        0,    0,    0,    0,    0,    0,    0,    0,   // rank 8
];

// Queen PST - avoid early development, modest centralization
#[rustfmt::skip]
const QUEEN_PST: [i32; 64] = [
    //  a     b     c     d     e     f     g     h
      -20,  -10,  -10,   -5,   -5,  -10,  -10,  -20,   // rank 1 - stay back early
      -10,    0,    5,    0,    0,    0,    0,  -10,   // rank 2
      -10,    5,    5,    5,    5,    5,    0,  -10,   // rank 3
        0,    0,    5,    5,    5,    5,    0,   -5,   // rank 4
       -5,    0,    5,    5,    5,    5,    0,   -5,   // rank 5
      -10,    0,    5,    5,    5,    5,    0,  -10,   // rank 6
      -10,    0,    0,    0,    0,    0,    0,  -10,   // rank 7
      -20,  -10,  -10,   -5,   -5,  -10,  -10,  -20,   // rank 8
];

// King PST (middlegame) - castle, stay safe
#[rustfmt::skip]
const KING_PST_MG: [i32; 64] = [
    //  a     b     c     d     e     f     g     h
       20,   30,   10,    0,    0,   10,   30,   20,   // rank 1 - castled positions are best
       20,   20,    0,    0,    0,    0,   20,   20,   // rank 2
      -10,  -20,  -20,  -20,  -20,  -20,  -20,  -10,   // rank 3
      -20,  -30,  -30,  -40,  -40,  -30,  -30,  -20,   // rank 4
      -30,  -40,  -40,  -50,  -50,  -40,  -40,  -30,   // rank 5
      -30,  -40,  -40,  -50,  -50,  -40,  -40,  -30,   // rank 6
      -30,  -40,  -40,  -50,  -50,  -40,  -40,  -30,   // rank 7
      -30,  -40,  -40,  -50,  -50,  -40,  -40,  -30,   // rank 8
];

// King PST (endgame) - centralize!
#[rustfmt::skip]
const KING_PST_EG: [i32; 64] = [
    //  a     b     c     d     e     f     g     h
      -50,  -30,  -30,  -30,  -30,  -30,  -30,  -50,   // rank 1
      -30,  -30,    0,    0,    0,    0,  -30,  -30,   // rank 2
      -30,  -10,   20,   30,   30,   20,  -10,  -30,   // rank 3
      -30,  -10,   30,   40,   40,   30,  -10,  -30,   // rank 4
      -30,  -10,   30,   40,   40,   30,  -10,  -30,   // rank 5
      -30,  -10,   20,   30,   30,   20,  -10,  -30,   // rank 6
      -30,  -20,  -10,    0,    0,  -10,  -20,  -30,   // rank 7
      -50,  -40,  -30,  -20,  -20,  -30,  -40,  -50,   // rank 8
];

// ==================== Evaluation Constants ====================

// Pawn structure
const DOUBLED_PAWN_PENALTY: i32 = 15;
const ISOLATED_PAWN_PENALTY: i32 = 20;
const PASSED_PAWN_BONUS: [i32; 8] = [0, 5, 10, 20, 35, 60, 100, 200]; // by rank advancement

// Piece bonuses
const BISHOP_PAIR_BONUS: i32 = 50;
const ROOK_OPEN_FILE_BONUS: i32 = 20;
const ROOK_SEMI_OPEN_BONUS: i32 = 10;
const ROOK_7TH_RANK_BONUS: i32 = 20;

// Mobility (per square)
const BISHOP_MOBILITY: i32 = 4;
const ROOK_MOBILITY: i32 = 2;

// Phase calculation
const TOTAL_PHASE: i32 = 24;

// ==================== Helper Functions ====================

/// Convert (file 1-8, rank 1-8) to PST index 0-63
#[inline]
fn coord_to_pst_index(x: i64, y: i64) -> usize {
    let file = (x - 1).clamp(0, 7) as usize;
    let rank = (y - 1).clamp(0, 7) as usize;
    rank * 8 + file
}

/// Check if on 8x8 board
#[inline]
fn on_board(x: i64, y: i64) -> bool {
    x >= 1 && x <= 8 && y >= 1 && y <= 8
}

/// Calculate game phase (24 = full pieces = middlegame, 0 = endgame)
fn get_phase(game: &GameState) -> i32 {
    let mut phase = 0;
    for (_, piece) in game.board.iter() {
        match piece.piece_type() {
            PieceType::Knight | PieceType::Bishop => phase += 1,
            PieceType::Rook => phase += 2,
            PieceType::Queen => phase += 4,
            _ => {}
        }
    }
    phase.min(TOTAL_PHASE)
}

// ==================== Main Evaluation ====================

pub fn evaluate(game: &GameState) -> i32 {
    let phase = get_phase(game);
    let mut score = 0;

    // Tracking
    let mut white_bishops = 0;
    let mut black_bishops = 0;
    let mut white_pawns: Vec<(i64, i64)> = Vec::with_capacity(8);
    let mut black_pawns: Vec<(i64, i64)> = Vec::with_capacity(8);
    let mut white_pawn_files = [false; 8];
    let mut black_pawn_files = [false; 8];
    let mut white_king_pos = (5i64, 1i64);
    let mut black_king_pos = (5i64, 8i64);

    // Main piece loop
    for ((x, y), piece) in game.board.iter() {
        if piece.color() == PlayerColor::Neutral || !on_board(*x, *y) {
            continue;
        }

        let is_white = piece.color() == PlayerColor::White;
        let pst_idx = coord_to_pst_index(*x, *y);
        // For black: mirror the rank (XOR 56 flips rank 1<->8, 2<->7, etc.)
        let black_pst_idx = pst_idx ^ 56;

        match piece.piece_type() {
            PieceType::Pawn => {
                let pst_val = if is_white {
                    PAWN_PST[pst_idx]
                } else {
                    PAWN_PST[black_pst_idx]
                };
                let val = PAWN_VALUE + pst_val;

                if is_white {
                    score += val;
                    white_pawns.push((*x, *y));
                    let file = (*x - 1) as usize;
                    white_pawn_files[file.min(7)] = true;
                } else {
                    score -= val;
                    black_pawns.push((*x, *y));
                    let file = (*x - 1) as usize;
                    black_pawn_files[file.min(7)] = true;
                }
            }
            PieceType::Knight => {
                let pst_val = if is_white {
                    KNIGHT_PST[pst_idx]
                } else {
                    KNIGHT_PST[black_pst_idx]
                };
                let val = KNIGHT_VALUE + pst_val;
                if is_white {
                    score += val;
                } else {
                    score -= val;
                }
            }
            PieceType::Bishop => {
                let pst_val = if is_white {
                    BISHOP_PST[pst_idx]
                } else {
                    BISHOP_PST[black_pst_idx]
                };
                let val = BISHOP_VALUE + pst_val;
                if is_white {
                    score += val;
                    white_bishops += 1;
                } else {
                    score -= val;
                    black_bishops += 1;
                }

                // Bishop mobility
                let mobility = count_bishop_mobility(&game.board, *x, *y);
                if is_white {
                    score += mobility * BISHOP_MOBILITY;
                } else {
                    score -= mobility * BISHOP_MOBILITY;
                }
            }
            PieceType::Rook => {
                let pst_val = if is_white {
                    ROOK_PST[pst_idx]
                } else {
                    ROOK_PST[black_pst_idx]
                };
                let val = ROOK_VALUE + pst_val;
                if is_white {
                    score += val;
                } else {
                    score -= val;
                }

                // Rook mobility
                let mobility = count_rook_mobility(&game.board, *x, *y);
                if is_white {
                    score += mobility * ROOK_MOBILITY;
                } else {
                    score -= mobility * ROOK_MOBILITY;
                }

                // Rook on 7th rank
                if (is_white && *y == 7) || (!is_white && *y == 2) {
                    if is_white {
                        score += ROOK_7TH_RANK_BONUS;
                    } else {
                        score -= ROOK_7TH_RANK_BONUS;
                    }
                }

                // Rook on open/semi-open file
                let file = ((*x - 1) as usize).min(7);
                let own_pawn = if is_white {
                    white_pawn_files[file]
                } else {
                    black_pawn_files[file]
                };
                let enemy_pawn = if is_white {
                    black_pawn_files[file]
                } else {
                    white_pawn_files[file]
                };
                if !own_pawn {
                    if !enemy_pawn {
                        if is_white {
                            score += ROOK_OPEN_FILE_BONUS;
                        } else {
                            score -= ROOK_OPEN_FILE_BONUS;
                        }
                    } else {
                        if is_white {
                            score += ROOK_SEMI_OPEN_BONUS;
                        } else {
                            score -= ROOK_SEMI_OPEN_BONUS;
                        }
                    }
                }
            }
            PieceType::Queen => {
                let pst_val = if is_white {
                    QUEEN_PST[pst_idx]
                } else {
                    QUEEN_PST[black_pst_idx]
                };
                let val = QUEEN_VALUE + pst_val;
                if is_white {
                    score += val;
                } else {
                    score -= val;
                }
            }
            PieceType::King => {
                // Tapered king PST
                let mg_pst = if is_white {
                    KING_PST_MG[pst_idx]
                } else {
                    KING_PST_MG[black_pst_idx]
                };
                let eg_pst = if is_white {
                    KING_PST_EG[pst_idx]
                } else {
                    KING_PST_EG[black_pst_idx]
                };
                let king_pst =
                    crate::simd::tapered_eval_simd(mg_pst, eg_pst, phase * 256 / TOTAL_PHASE);
                if is_white {
                    score += king_pst;
                    white_king_pos = (*x, *y);
                } else {
                    score -= king_pst;
                    black_king_pos = (*x, *y);
                }
            }
            _ => {}
        }
    }

    // Bishop pair
    if white_bishops >= 2 {
        score += BISHOP_PAIR_BONUS;
    }
    if black_bishops >= 2 {
        score -= BISHOP_PAIR_BONUS;
    }

    // Pawn structure
    score += evaluate_pawn_structure(
        &white_pawns,
        &black_pawns,
        &white_pawn_files,
        &black_pawn_files,
    );

    // Endgame: King distance to passed pawns
    if phase < 12 {
        score +=
            evaluate_king_pawn_distance(&white_pawns, &black_pawns, white_king_pos, black_king_pos);
    }

    // Return from current player's perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

// ==================== Mobility ====================

fn count_bishop_mobility(board: &Board, x: i64, y: i64) -> i32 {
    let dirs = [(1i64, 1i64), (1, -1), (-1, 1), (-1, -1)];
    let mut count = 0;
    for (dx, dy) in dirs {
        let mut nx = x + dx;
        let mut ny = y + dy;
        while on_board(nx, ny) {
            if board.get_piece(nx, ny).is_some() {
                break;
            }
            count += 1;
            nx += dx;
            ny += dy;
        }
    }
    count
}

fn count_rook_mobility(board: &Board, x: i64, y: i64) -> i32 {
    let dirs = [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)];
    let mut count = 0;
    for (dx, dy) in dirs {
        let mut nx = x + dx;
        let mut ny = y + dy;
        while on_board(nx, ny) {
            if board.get_piece(nx, ny).is_some() {
                break;
            }
            count += 1;
            nx += dx;
            ny += dy;
        }
    }
    count
}

// ==================== Pawn Structure ====================

fn evaluate_pawn_structure(
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
    white_files: &[bool; 8],
    black_files: &[bool; 8],
) -> i32 {
    let mut score = 0;

    // Count pawns per file for doubled detection
    let mut white_file_count = [0i32; 8];
    let mut black_file_count = [0i32; 8];

    for &(x, _) in white_pawns {
        let f = ((x - 1) as usize).min(7);
        white_file_count[f] += 1;
    }
    for &(x, _) in black_pawns {
        let f = ((x - 1) as usize).min(7);
        black_file_count[f] += 1;
    }

    // White pawns
    for &(x, y) in white_pawns {
        let file = ((x - 1) as usize).min(7);
        let rank = y as usize;

        // Doubled pawns
        if white_file_count[file] > 1 {
            score -= DOUBLED_PAWN_PENALTY;
        }

        // Isolated pawns
        let has_neighbor =
            (file > 0 && white_files[file - 1]) || (file < 7 && white_files[file + 1]);
        if !has_neighbor {
            score -= ISOLATED_PAWN_PENALTY;
        }

        // Passed pawn: no enemy pawn on same or adjacent files ahead
        let mut is_passed = true;
        for &(px, py) in black_pawns {
            if (px - x).abs() <= 1 && py > y {
                is_passed = false;
                break;
            }
        }
        if is_passed && rank >= 2 && rank <= 7 {
            score += PASSED_PAWN_BONUS[rank - 1];
        }
    }

    // Black pawns
    for &(x, y) in black_pawns {
        let file = ((x - 1) as usize).min(7);
        let rank = (9 - y) as usize; // Flip for black (rank 7 -> advancement 2, rank 2 -> advancement 7)

        // Doubled
        if black_file_count[file] > 1 {
            score += DOUBLED_PAWN_PENALTY;
        }

        // Isolated
        let has_neighbor =
            (file > 0 && black_files[file - 1]) || (file < 7 && black_files[file + 1]);
        if !has_neighbor {
            score += ISOLATED_PAWN_PENALTY;
        }

        // Passed pawn
        let mut is_passed = true;
        for &(px, py) in white_pawns {
            if (px - x).abs() <= 1 && py < y {
                is_passed = false;
                break;
            }
        }
        if is_passed && rank >= 2 && rank <= 7 {
            score -= PASSED_PAWN_BONUS[rank - 1];
        }
    }

    score
}

// ==================== Endgame: King-Pawn Distance ====================

fn evaluate_king_pawn_distance(
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
    white_king: (i64, i64),
    black_king: (i64, i64),
) -> i32 {
    let mut score = 0;

    // White passed pawns: white king close is good, black king far is good
    for &(px, py) in white_pawns {
        let is_passed = !black_pawns
            .iter()
            .any(|&(bx, by)| (bx - px).abs() <= 1 && by > py);
        if is_passed && py >= 4 {
            let white_dist = (white_king.0 - px).abs() + (white_king.1 - py).abs();
            let black_dist = (black_king.0 - px).abs() + (black_king.1 - py).abs();
            score += (black_dist - white_dist) as i32 * 5;
        }
    }

    // Black passed pawns
    for &(px, py) in black_pawns {
        let is_passed = !white_pawns
            .iter()
            .any(|&(wx, wy)| (wx - px).abs() <= 1 && wy < py);
        if is_passed && py <= 5 {
            let white_dist = (white_king.0 - px).abs() + (white_king.1 - py).abs();
            let black_dist = (black_king.0 - px).abs() + (black_king.1 - py).abs();
            score -= (white_dist - black_dist) as i32 * 5;
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece};
    use crate::game::GameState;

    fn create_chess_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game.variant = Some(crate::Variant::Chess);
        game
    }

    fn setup_standard_chess_opening(game: &mut GameState) {
        // Set up standard chess starting position
        // White pieces
        for file in 1..=8 {
            game.board
                .set_piece(file, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
            game.board
                .set_piece(file, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
        }
        game.board
            .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        game.board
            .set_piece(6, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));

        // Black pieces
        game.board
            .set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.board
            .set_piece(8, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.board
            .set_piece(2, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        game.board
            .set_piece(7, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        game.board
            .set_piece(3, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        game.board
            .set_piece(6, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        game.board
            .set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));

        game.recompute_piece_counts();
        game.recompute_hash();
    }

    // ======================== Helper Function Tests ========================

    #[test]
    fn test_coord_to_pst_index() {
        // a1 = (1,1) -> index 0
        assert_eq!(coord_to_pst_index(1, 1), 0);
        // h1 = (8,1) -> index 7
        assert_eq!(coord_to_pst_index(8, 1), 7);
        // a8 = (1,8) -> index 56
        assert_eq!(coord_to_pst_index(1, 8), 56);
        // e4 = (5,4) -> index 28
        assert_eq!(coord_to_pst_index(5, 4), 28);
    }

    #[test]
    fn test_on_board() {
        assert!(on_board(1, 1));
        assert!(on_board(8, 8));
        assert!(on_board(4, 4));
        assert!(!on_board(0, 1));
        assert!(!on_board(9, 1));
        assert!(!on_board(1, 0));
        assert!(!on_board(1, 9));
    }

    #[test]
    fn test_get_phase_starting() {
        let mut game = create_chess_game();
        setup_standard_chess_opening(&mut game);

        let phase = get_phase(&game);
        // 4 knights (4*1) + 4 bishops (4*1) + 4 rooks (4*2) + 2 queens (2*4) = 4+4+8+8 = 24
        assert_eq!(phase, TOTAL_PHASE, "Starting position should be full phase");
    }

    #[test]
    fn test_get_phase_endgame() {
        let mut game = create_chess_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.recompute_piece_counts();

        let phase = get_phase(&game);
        assert_eq!(phase, 2, "Rook only = phase 2");
    }

    // ======================== Mobility Tests ========================

    #[test]
    fn test_bishop_mobility_empty_board() {
        let board = Board::new();
        let mobility = count_bishop_mobility(&board, 4, 4);
        // Bishop in center can move 7 squares in each diagonal direction = 28 total
        // But on_board limits to 8x8
        assert!(mobility > 0, "Bishop should have mobility on empty board");
    }

    #[test]
    fn test_rook_mobility_empty_board() {
        let board = Board::new();
        let mobility = count_rook_mobility(&board, 4, 4);
        // Rook in center, limited by 8x8 board
        assert!(mobility > 0, "Rook should have mobility on empty board");
    }

    #[test]
    fn test_bishop_mobility_blocked() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Bishop, PlayerColor::White));
        // Block all diagonals adjacent
        board.set_piece(5, 5, Piece::new(PieceType::Pawn, PlayerColor::White));
        board.set_piece(3, 3, Piece::new(PieceType::Pawn, PlayerColor::White));
        board.set_piece(5, 3, Piece::new(PieceType::Pawn, PlayerColor::White));
        board.set_piece(3, 5, Piece::new(PieceType::Pawn, PlayerColor::White));

        let mobility = count_bishop_mobility(&board, 4, 4);
        assert_eq!(mobility, 0, "Blocked bishop should have 0 mobility");
    }

    // ======================== Evaluation Tests ========================

    #[test]
    fn test_evaluate_starting_position() {
        let mut game = create_chess_game();
        setup_standard_chess_opening(&mut game);
        game.turn = PlayerColor::White;

        let score = evaluate(&game);
        // Starting position should be roughly equal (within pawn value)
        assert!(
            score.abs() < 100,
            "Starting position should be near 0, got {}",
            score
        );
    }

    #[test]
    fn test_evaluate_queen_up() {
        let mut game = create_chess_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let score = evaluate(&game);
        assert!(
            score > QUEEN_VALUE / 2,
            "Queen up should be positive, got {}",
            score
        );
    }

    #[test]
    fn test_evaluate_bishop_pair_bonus() {
        let mut game = create_chess_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        game.board
            .set_piece(6, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let score_with_pair = evaluate(&game);

        // Now with only one bishop
        game.board.remove_piece(&6, &1);
        game.recompute_piece_counts();
        let score_with_one = evaluate(&game);

        let diff = score_with_pair - score_with_one;
        // Should be bishop value + pair bonus
        assert!(diff > BISHOP_VALUE, "Bishop pair bonus should add value");
    }

    #[test]
    fn test_evaluate_passed_pawn() {
        let mut game = create_chess_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // White passed pawn on 6th rank
        game.board
            .set_piece(4, 6, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let score = evaluate(&game);
        assert!(
            score > PAWN_VALUE,
            "Passed pawn on 6th should be very valuable"
        );
    }

    #[test]
    fn test_evaluate_doubled_pawns() {
        let mut game = create_chess_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // White doubled pawns
        game.board
            .set_piece(4, 3, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let score = evaluate(&game);
        // Doubled pawns are penalized, but still positive because white has 2 pawns
        assert!(
            score.abs() < 2 * PAWN_VALUE + 100,
            "Score should reflect doubled pawn penalty"
        );
    }

    #[test]
    fn test_evaluate_rook_on_7th() {
        let mut game = create_chess_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 7, Piece::new(PieceType::Rook, PlayerColor::White)); // 7th rank
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let score_on_7th = evaluate(&game);

        // Compare with rook on 4th
        game.board.remove_piece(&4, &7);
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.recompute_piece_counts();

        let score_on_4th = evaluate(&game);

        assert!(
            score_on_7th > score_on_4th,
            "Rook on 7th should score better"
        );
    }

    #[test]
    fn test_evaluate_from_black_perspective() {
        let mut game = create_chess_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));

        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        let score_white = evaluate(&game);

        game.turn = PlayerColor::Black;
        let score_black = evaluate(&game);

        // Scores should be negated
        assert_eq!(
            score_white, -score_black,
            "Score should negate for opposite side"
        );
    }
}
