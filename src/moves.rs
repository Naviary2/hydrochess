use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::game::{EnPassantState, GameRules};
use crate::utils::is_prime_i64;
use serde::{Serialize, Deserialize};
use std::collections::{HashSet, HashMap};

// World border for infinite chess. These are initialized to a very large box,
// but can be overridden from JS via the playableRegion values.
static mut COORD_MIN_X: i64 = -1_000_000_000_000_000; // default -1e15
static mut COORD_MAX_X: i64 =  1_000_000_000_000_000; // default  1e15
static mut COORD_MIN_Y: i64 = -1_000_000_000_000_000; // default -1e15
static mut COORD_MAX_Y: i64 =  1_000_000_000_000_000; // default  1e15

/// Update world borders from JS playableRegion (left, right, bottom, top).
/// Rounding errors from BigInt -> i64 conversion on the JS side are acceptable.
pub fn set_world_bounds(left: i64, right: i64, bottom: i64, top: i64) {
    unsafe {
        COORD_MIN_X = left.min(right);
        COORD_MAX_X = left.max(right);
        COORD_MIN_Y = bottom.min(top);
        COORD_MAX_Y = bottom.max(top);
    }
}

/// Check if a coordinate is within valid bounds (world border)
#[inline]
pub fn in_bounds(x: i64, y: i64) -> bool {
    unsafe {
        x >= COORD_MIN_X && x <= COORD_MAX_X && y >= COORD_MIN_Y && y <= COORD_MAX_Y
    }
}

pub struct SpatialIndices {
    pub rows: HashMap<i64, Vec<i64>>,
    pub cols: HashMap<i64, Vec<i64>>,
    pub diag1: HashMap<i64, Vec<i64>>, // x - y
    pub diag2: HashMap<i64, Vec<i64>>, // x + y
}

impl SpatialIndices {
    pub fn new(board: &Board) -> Self {
        let mut rows: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut cols: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut diag1: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut diag2: HashMap<i64, Vec<i64>> = HashMap::new();

        for ((x, y), _) in &board.pieces {
            rows.entry(*y).or_default().push(*x);
            cols.entry(*x).or_default().push(*y);
            diag1.entry(x - y).or_default().push(*x);
            diag2.entry(x + y).or_default().push(*x);
        }

        // Sort vectors for binary search or efficient scanning
        for list in rows.values_mut() { list.sort(); }
        for list in cols.values_mut() { list.sort(); }
        for list in diag1.values_mut() { list.sort(); }
        for list in diag2.values_mut() { list.sort(); }

        SpatialIndices { rows, cols, diag1, diag2 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Move {
    pub from: Coordinate,
    pub to: Coordinate,
    pub piece: Piece,
    pub promotion: Option<String>,
    pub rook_coord: Option<Coordinate>, // For castling: stores the rook's coordinate
}

impl Move {
    pub fn new(from: Coordinate, to: Coordinate, piece: Piece) -> Self {
        Move {
            from,
            to,
            piece,
            promotion: None,
            rook_coord: None,
        }
    }
}


#[inline]
fn is_enemy_piece(piece: &Piece, our_color: PlayerColor) -> bool {
    piece.color != our_color && piece.piece_type != PieceType::Void
}


pub fn get_legal_moves(board: &Board, turn: PlayerColor, special_rights: &HashSet<Coordinate>, en_passant: &Option<EnPassantState>, game_rules: &GameRules) -> Vec<Move> {
    let indices = SpatialIndices::new(board);
    let mut moves = Vec::new();

    for ((x, y), piece) in &board.pieces {
        // Skip neutral pieces and non-turn pieces
        if piece.color != turn || piece.color == PlayerColor::Neutral {
            continue;
        }

        let from = Coordinate::new(*x, *y);
        let piece_moves = get_pseudo_legal_moves_for_piece(board, piece, &from, special_rights, en_passant, Some(&indices), game_rules);
        moves.extend(piece_moves);
    }

    moves
}

pub fn get_pseudo_legal_moves_for_piece(board: &Board, piece: &Piece, from: &Coordinate, special_rights: &HashSet<Coordinate>, en_passant: &Option<EnPassantState>, indices: Option<&SpatialIndices>, game_rules: &GameRules) -> Vec<Move> {
    match piece.piece_type {
        // Neutral/blocking pieces cannot move
        PieceType::Void | PieceType::Obstacle => Vec::new(),
        PieceType::Pawn => generate_pawn_moves(board, from, piece, special_rights, en_passant, game_rules),
        PieceType::Knight => generate_leaper_moves(board, from, piece, 1, 2),
        PieceType::Hawk => {
            let mut m = generate_compass_moves(board, from, piece, 2);
            m.extend(generate_compass_moves(board, from, piece, 3));
            m
        },
        PieceType::King => {
            let mut m = generate_compass_moves(board, from, piece, 1);
            m.extend(generate_castling_moves(board, from, piece, special_rights, indices));
            m
        },
        PieceType::Guard => generate_compass_moves(board, from, piece, 1),
        PieceType::Rook => generate_sliding_moves(board, from, piece, &[(1, 0), (0, 1)], indices),
        PieceType::Bishop => generate_sliding_moves(board, from, piece, &[(1, 1), (1, -1)], indices),
        PieceType::Queen | PieceType::RoyalQueen => {
            let mut m = generate_sliding_moves(board, from, piece, &[(1, 0), (0, 1)], indices);
            m.extend(generate_sliding_moves(board, from, piece, &[(1, 1), (1, -1)], indices));
            m
        },
        PieceType::Chancellor => {
            let mut m = generate_leaper_moves(board, from, piece, 1, 2);
            m.extend(generate_sliding_moves(board, from, piece, &[(1, 0), (0, 1)], indices));
            m
        },
        PieceType::Archbishop => {
            let mut m = generate_leaper_moves(board, from, piece, 1, 2);
            m.extend(generate_sliding_moves(board, from, piece, &[(1, 1), (1, -1)], indices));
            m
        },
        PieceType::Amazon => {
            let mut m = generate_leaper_moves(board, from, piece, 1, 2);
            m.extend(generate_sliding_moves(board, from, piece, &[(1, 0), (0, 1)], indices));
            m.extend(generate_sliding_moves(board, from, piece, &[(1, 1), (1, -1)], indices));
            m
        },
        PieceType::Camel => generate_leaper_moves(board, from, piece, 1, 3),
        PieceType::Giraffe => generate_leaper_moves(board, from, piece, 1, 4),
        PieceType::Zebra => generate_leaper_moves(board, from, piece, 2, 3),
        PieceType::Knightrider => generate_sliding_moves(board, from, piece, &[(1, 2), (1, -2), (2, 1), (2, -1)], indices),
        PieceType::Centaur => {
            let mut m = generate_compass_moves(board, from, piece, 1);
            m.extend(generate_leaper_moves(board, from, piece, 1, 2));
            m
        },
        PieceType::RoyalCentaur => {
            let mut m = generate_compass_moves(board, from, piece, 1);
            m.extend(generate_leaper_moves(board, from, piece, 1, 2));
            m.extend(generate_castling_moves(board, from, piece, special_rights, indices));
            m
        },
        PieceType::Huygen => generate_huygen_moves(board, from, piece, indices),
        PieceType::Rose => generate_rose_moves(board, from, piece),
    }
}

pub fn is_square_attacked(board: &Board, target: &Coordinate, attacker_color: PlayerColor, indices: Option<&SpatialIndices>) -> bool {
    // 1. Check Leapers (Knight, Camel, Giraffe, Zebra, King/Guard/Centaur/RoyalCentaur)
    // We check the offsets *from* the target. If a piece is there, it can attack *to* the target.
    let leaper_checks = [
        (vec![(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)], vec![PieceType::Knight, PieceType::Chancellor, PieceType::Archbishop, PieceType::Amazon, PieceType::Centaur, PieceType::RoyalCentaur]),
        (vec![(1, 3), (1, -3), (3, 1), (3, -1), (-1, 3), (-1, -3), (-3, 1), (-3, -1)], vec![PieceType::Camel]),
        (vec![(1, 4), (1, -4), (4, 1), (4, -1), (-1, 4), (-1, -4), (-4, 1), (-4, -1)], vec![PieceType::Giraffe]),
        (vec![(2, 3), (2, -3), (3, 2), (3, -2), (-2, 3), (-2, -3), (-3, 2), (-3, -2)], vec![PieceType::Zebra]),
        (vec![(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)], vec![PieceType::King, PieceType::Guard, PieceType::Centaur, PieceType::RoyalCentaur]),
        // Hawk: (2,0), (3,0), (2,2), (3,3) and rotations
        (vec![(2, 0), (-2, 0), (0, 2), (0, -2), (3, 0), (-3, 0), (0, 3), (0, -3), (2, 2), (2, -2), (-2, 2), (-2, -2), (3, 3), (3, -3), (-3, 3), (-3, -3)], vec![PieceType::Hawk]),
    ];

    for (offsets, types) in &leaper_checks {
        for (dx, dy) in offsets {
            let x = target.x + dx;
            let y = target.y + dy;
            if let Some(piece) = board.get_piece(&x, &y) {
                if piece.color == attacker_color && types.contains(&piece.piece_type) {
                    return true;
                }
            }
        }
    }

    // 2. Check Pawns
    let pawn_dir = match attacker_color {
        PlayerColor::White => 1, // White pawns attack upwards (y+1), so they come from y-1
        PlayerColor::Black => -1, // Black pawns attack downwards (y-1), so they come from y+1
        PlayerColor::Neutral => 0, // Neutral pawns don't attack
    };
    // Attackers are at target.y - dir
    let pawn_y = target.y - pawn_dir;
    for pawn_dx in [-1, 1] {
        let pawn_x = target.x + pawn_dx;
        if let Some(piece) = board.get_piece(&pawn_x, &pawn_y) {
            if piece.color == attacker_color && piece.piece_type == PieceType::Pawn {
                return true;
            }
        }
    }

    // 3. Check Sliding Pieces (Orthogonal and Diagonal)
    // We look outwards from target. The first piece we hit must be a slider of the correct type.
    let ortho_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let diag_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    
    let ortho_types = [PieceType::Rook, PieceType::Queen, PieceType::Chancellor, PieceType::Amazon, PieceType::RoyalQueen];
    let diag_types = [PieceType::Bishop, PieceType::Queen, PieceType::Archbishop, PieceType::Amazon, PieceType::RoyalQueen];

    // Helper to check rays
    let check_ray = |dirs: &[(i64, i64)], valid_types: &[PieceType]| -> bool {
        for (dx, dy) in dirs {
            // Use SpatialIndices if available to jump to nearest piece
            let mut closest_piece: Option<Piece> = None;
            let mut found_via_indices = false;

            if let Some(indices) = indices {
                let line_vec = if *dx == 0 { indices.cols.get(&target.x) } else if *dy == 0 { indices.rows.get(&target.y) } else if *dx == *dy { indices.diag1.get(&(target.x - target.y)) } else { indices.diag2.get(&(target.x + target.y)) };
                
                if let Some(vec) = line_vec {
                    let val = if *dx == 0 { target.y } else { target.x };
                    if let Ok(idx) = vec.binary_search(&val) {
                        let step_dir = if *dx == 0 { *dy } else { *dx };
                        if step_dir > 0 {
                            if idx + 1 < vec.len() {
                                let next_val = vec[idx + 1];
                                let (tx, ty) = if *dx == 0 { (target.x, next_val) } else if *dy == 0 { (next_val, target.y) } else if *dx == *dy { (next_val, next_val - (target.x - target.y)) } else { (next_val, (target.x + target.y) - next_val) };
                                if let Some(p) = board.get_piece(&tx, &ty) { closest_piece = Some(p.clone()); }
                                found_via_indices = true;
                            }
                        } else {
                            if idx > 0 {
                                let prev_val = vec[idx - 1];
                                let (tx, ty) = if *dx == 0 { (target.x, prev_val) } else if *dy == 0 { (prev_val, target.y) } else if *dx == *dy { (prev_val, prev_val - (target.x - target.y)) } else { (prev_val, (target.x + target.y) - prev_val) };
                                if let Some(p) = board.get_piece(&tx, &ty) { closest_piece = Some(p.clone()); }
                                found_via_indices = true;
                            }
                        }
                    }
                }
            }

            if !found_via_indices {
                // Fallback ray scan
                let mut k = 1;
                loop {
                    let x = target.x + dx * k;
                    let y = target.y + dy * k;
                    
                    if let Some(piece) = board.get_piece(&x, &y) {
                        closest_piece = Some(piece.clone());
                        break;
                    }
                    k += 1;
                    if k > 50 { break; } // Safety limit for fallback
                }
            }

            if let Some(piece) = closest_piece {
                if piece.color == attacker_color && valid_types.contains(&piece.piece_type) {
                    return true;
                }
            }
        }
        false
    };

    if check_ray(&ortho_dirs, &ortho_types) { return true; }
    if check_ray(&diag_dirs, &diag_types) { return true; }

    // 4. Check Knightrider (Sliding Knight)
    // Vectors: (1,2), (1,-2), (2,1), (2,-1) etc.
    // We check rays in these 8 directions.
    let kr_dirs = [(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)];
    // Reuse check_ray logic but without indices (indices don't support knight lines yet)
    // Or just manual scan
    for (dx, dy) in kr_dirs {
        let mut k = 1;
        loop {
            let x = target.x + dx * k;
            let y = target.y + dy * k;
            if let Some(piece) = board.get_piece(&x, &y) {
                if piece.color == attacker_color && piece.piece_type == PieceType::Knightrider {
                    return true;
                }
                break; // Blocked
            }
            k += 1;
            if k > 25 { break; } // Safety
        }
    }

    // 5. Check Huygen (Prime Leaper/Slider)
    // Orthogonal directions. Check all prime distances.
    // Optimization: Use indices to find pieces on the line, then check if distance is prime.
    for (dx, dy) in ortho_dirs {
         if let Some(indices) = indices {
             let line_vec = if dx == 0 { indices.cols.get(&target.x) } else { indices.rows.get(&target.y) };
             if let Some(vec) = line_vec {
                 // Iterate all pieces on this line
                 for val in vec {
                     let dist = if dx == 0 { val - target.y } else { val - target.x };
                     let abs_dist = dist.abs();
                     if abs_dist > 0 && is_prime_i64(abs_dist) {
                         // Check direction
                         let sign = if dist > 0 { 1 } else { -1 };
                         let dir_check = if dx == 0 { if dy == sign { true } else { false } } else { if dx == sign { true } else { false } };
                         
                         if dir_check {
                             let (tx, ty) = if dx == 0 { (target.x, *val) } else { (*val, target.y) };
                             if let Some(piece) = board.get_piece(&tx, &ty) {
                                 if piece.color == attacker_color && piece.piece_type == PieceType::Huygen {
                                     return true;
                                 }
                             }
                         }
                     }
                 }
             }
         } else {
             // Fallback: Scan? No, infinite board.
             // Just check known pieces?
             // Since this is fallback, maybe skip or do slow check.
             // But indices should be available.
         }
    }

    // 6. Check Rose (Circular Knight)
    // Max 8 directions * 7 steps = 56 squares.
    // We can scan outwards from target in reverse rose moves.
    // So we generate rose moves from target and see if we hit a Rose.
    // My generate_rose_moves checks `board.get_piece` and breaks if blocked.
    // So it is blocked by pieces.
    // So we can trace out from target.
    let rose_moves = generate_rose_moves(board, target, &Piece::new(PieceType::Rose, attacker_color)); // Dummy piece
    for m in rose_moves {
        if let Some(piece) = board.get_piece(&m.to.x, &m.to.y) {
            if piece.color == attacker_color && piece.piece_type == PieceType::Rose {
                return true;
            }
        }
    }

    false
}

fn generate_pawn_moves(board: &Board, from: &Coordinate, piece: &Piece, special_rights: &HashSet<Coordinate>, en_passant: &Option<EnPassantState>, game_rules: &GameRules) -> Vec<Move> {
    let mut moves = Vec::new();
    let direction = match piece.color {
        PlayerColor::White => 1,
        PlayerColor::Black => -1,
        PlayerColor::Neutral => return moves, // Neutral pawns can't move
    };

    // Get promotion ranks for this color (default to 8 for white, 1 for black if not specified)
    let promotion_ranks: Vec<i64> = if let Some(ref ranks) = game_rules.promotion_ranks {
        match piece.color {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => vec![],
        }
    } else {
        // Default promotion ranks for standard chess
        match piece.color {
            PlayerColor::White => vec![8],
            PlayerColor::Black => vec![1],
            PlayerColor::Neutral => vec![],
        }
    };

    // Get allowed promotion pieces (default to Q, R, B, N)
    let promotion_pieces: Vec<&str> = if let Some(ref allowed) = game_rules.promotions_allowed {
        allowed.iter().map(|s| s.as_str()).collect()
    } else {
        vec!["q", "r", "b", "n"] // Default promotions
    };

    // Helper function to add pawn move with promotion handling
    fn add_pawn_move_inner(
        moves: &mut Vec<Move>,
        from: &Coordinate,
        to_x: i64,
        to_y: i64,
        piece: &Piece,
        promotion_ranks: &[i64],
        promotion_pieces: &[&str],
    ) {
        if promotion_ranks.contains(&to_y) {
            // Generate a move for each possible promotion piece
            for promo in promotion_pieces {
                let mut m = Move::new(from.clone(), Coordinate::new(to_x, to_y), piece.clone());
                m.promotion = Some(promo.to_string());
                moves.push(m);
            }
        } else {
            moves.push(Move::new(from.clone(), Coordinate::new(to_x, to_y), piece.clone()));
        }
    }

    // Move forward 1
    let to_y = from.y + direction;
    let to_x = from.x;
    
    // Check if square is blocked
    let forward_blocked = board.get_piece(&to_x, &to_y).is_some();
    
    if !forward_blocked {
        add_pawn_move_inner(&mut moves, from, to_x, to_y, piece, &promotion_ranks, &promotion_pieces);
        
        // Move forward 2 if pawn has special rights (double-move available)
        // This is now dynamic - based on special_rights set, not hardcoded starting rank
        // Note: double-move cannot result in promotion, so no need to check
        if special_rights.contains(from) {
            let to_y_2 = from.y + (direction * 2);
            // Must also check that the target square isn't blocked
            if board.get_piece(&to_x, &to_y_2).is_none() {
                moves.push(Move::new(from.clone(), Coordinate::new(to_x, to_y_2), piece.clone()));
            }
        }
    }

    // Captures (including neutral pieces - they can be captured)
    for dx in [-1i64, 1] {
        let capture_x = from.x + dx;
        let capture_y = from.y + direction;
        
        if let Some(target) = board.get_piece(&capture_x, &capture_y) {
            // Can capture any piece that's not the same color as us
            // This includes neutral pieces (obstacles can be captured)
            if is_enemy_piece(target, piece.color) {
                add_pawn_move_inner(&mut moves, from, capture_x, capture_y, piece, &promotion_ranks, &promotion_pieces);
            }
        } else {
            // En Passant - cannot result in promotion so no promotion check needed
            if let Some(ep) = en_passant {
                if ep.square.x == capture_x && ep.square.y == capture_y {
                    moves.push(Move::new(from.clone(), Coordinate::new(capture_x, capture_y), piece.clone()));
                }
            }
        }
    }

    moves
}

fn generate_castling_moves(board: &Board, from: &Coordinate, piece: &Piece, special_rights: &HashSet<Coordinate>, indices: Option<&SpatialIndices>) -> Vec<Move> {
    let mut moves = Vec::new();
    
    // King must have special rights to castle
    if !special_rights.contains(from) {
        return moves;
    }
    
    // Find all pieces with special rights that could be castling partners
    for coord in special_rights.iter() {
        if let Some(target_piece) = board.get_piece(&coord.x, &coord.y) {
            // Must be same color and a valid castling partner (rook-like piece, not pawn)
            if target_piece.color == piece.color && 
               target_piece.piece_type != PieceType::Pawn &&
               !target_piece.piece_type.is_royal() {
                let dx = coord.x - from.x;
                let dy = coord.y - from.y;
                
                if dy == 0 {
                    let dir = if dx > 0 { 1 } else { -1 };
                    
                    let mut clear = true;
                    let mut current_x = from.x + dir;
                    while current_x != coord.x {
                        if board.get_piece(&current_x, &from.y).is_some() {
                            clear = false;
                            break;
                        }
                        current_x += dir;
                    }

                    if clear {
                        let opponent = piece.color.opponent();

                        let path_1 = from.x + dir;
                        let path_2 = from.x + (dir * 2);
                        
                        let pos_1 = Coordinate::new(path_1, from.y);
                        let pos_2 = Coordinate::new(path_2, from.y);

                        {
                            if !is_square_attacked(board, from, opponent, indices) &&
                               !is_square_attacked(board, &pos_1, opponent, indices) &&
                               !is_square_attacked(board, &pos_2, opponent, indices) {
                                   
                                let to_x = from.x + (dir * 2);
                                let mut castling_move = Move::new(from.clone(), Coordinate::new(to_x, from.y), piece.clone());
                                castling_move.rook_coord = Some(coord.clone());
                                moves.push(castling_move);
                            }
                        }
                    }
                }
            }
        }
    }
    moves
}

fn generate_compass_moves(board: &Board, from: &Coordinate, piece: &Piece, distance: i64) -> Vec<Move> {
    let mut moves = Vec::new();
    let dist = distance;
    let offsets = [
        (-dist, dist), (0, dist), (dist, dist),
        (-dist, 0), (dist, 0),
        (-dist, -dist), (0, -dist), (dist, -dist)
    ];

    for (dx, dy) in offsets {
        let to_x = from.x + dx;
        let to_y = from.y + dy;
        
        // Skip if outside world border
        if !in_bounds(to_x, to_y) { continue; }
        
        if let Some(target) = board.get_piece(&to_x, &to_y) {
            if is_enemy_piece(target, piece.color) {
                moves.push(Move::new(from.clone(), Coordinate::new(to_x, to_y), piece.clone()));
            }
        } else {
            moves.push(Move::new(from.clone(), Coordinate::new(to_x, to_y), piece.clone()));
        }
    }
    moves
}

fn generate_leaper_moves(board: &Board, from: &Coordinate, piece: &Piece, m: i64, n: i64) -> Vec<Move> {
    let mut moves = Vec::new();
    
    let offsets = [
        (-n, m), (-m, n), (m, n), (n, m),
        (-n, -m), (-m, -n), (m, -n), (n, -m)
    ];

    for (dx, dy) in offsets {
        let to_x = from.x + dx;
        let to_y = from.y + dy;
        
        // Skip if outside world border
        if !in_bounds(to_x, to_y) { continue; }
        
        if let Some(target) = board.get_piece(&to_x, &to_y) {
            if is_enemy_piece(target, piece.color) {
                moves.push(Move::new(from.clone(), Coordinate::new(to_x, to_y), piece.clone()));
            }
        } else {
            moves.push(Move::new(from.clone(), Coordinate::new(to_x, to_y), piece.clone()));
        }
    }
    moves
}

fn generate_sliding_moves(board: &Board, from: &Coordinate, piece: &Piece, directions: &[(i64, i64)], _indices: Option<&SpatialIndices>) -> Vec<Move> {
    let mut moves = Vec::new();
    let wiggle_room: i64 = 2;
    let friend_wiggle_room: i64 = 1;
    const MAX_SLIDE_CHECK: i64 = 50;

    // Collect all piece positions for alignment checks
    let mut enemy_positions: Vec<(i64, i64)> = Vec::new();
    let mut all_positions: Vec<(i64, i64)> = Vec::new();
    
    for ((px, py), p) in &board.pieces {
        all_positions.push((*px, *py));
        if is_enemy_piece(p, piece.color) {
            enemy_positions.push((*px, *py));
        }
    }

    // Build piece capability flags from directions
    let can_horiz = directions.iter().any(|(dx, dy)| *dy == 0 && *dx != 0);
    let can_vert = directions.iter().any(|(dx, dy)| *dx == 0 && *dy != 0);
    let can_diag = directions.iter().any(|(dx, dy)| *dx != 0 && *dy != 0);

    // Helper: Check if there's a clear path to an enemy on a row from (sq_x, sq_y)
    let has_clear_enemy_row = |sq_x: i64, sq_y: i64| -> bool {
        for dir in [-1i64, 1i64] {
            for step in 1..=MAX_SLIDE_CHECK {
                let x = sq_x + dir * step;
                if all_positions.contains(&(x, sq_y)) {
                    if enemy_positions.contains(&(x, sq_y)) {
                        return true;
                    }
                    break; // blocked by friendly
                }
            }
        }
        false
    };

    // Helper: Check if there's a clear path to an enemy on a column from (sq_x, sq_y)
    let has_clear_enemy_col = |sq_x: i64, sq_y: i64| -> bool {
        for dir in [-1i64, 1i64] {
            for step in 1..=MAX_SLIDE_CHECK {
                let y = sq_y + dir * step;
                if all_positions.contains(&(sq_x, y)) {
                    if enemy_positions.contains(&(sq_x, y)) {
                        return true;
                    }
                    break;
                }
            }
        }
        false
    };

    // Helper: Check if there's a clear path to an enemy on a diagonal from (sq_x, sq_y)
    let has_clear_enemy_diag = |sq_x: i64, sq_y: i64| -> bool {
        for dir_x in [-1i64, 1i64] {
            for dir_y in [-1i64, 1i64] {
                for step in 1..=MAX_SLIDE_CHECK {
                    let x = sq_x + dir_x * step;
                    let y = sq_y + dir_y * step;
                    if all_positions.contains(&(x, y)) {
                        if enemy_positions.contains(&(x, y)) {
                            return true;
                        }
                        break;
                    }
                }
            }
        }
        false
    };

    // Pre-compute enemy alignment maps for fast lookup
    let mut enemy_rows: HashSet<i64> = HashSet::new();
    let mut enemy_cols: HashSet<i64> = HashSet::new();
    let mut enemy_diag1: HashSet<i64> = HashSet::new(); // x - y
    let mut enemy_diag2: HashSet<i64> = HashSet::new(); // x + y

    for (ex, ey) in &enemy_positions {
        enemy_rows.insert(*ey);
        enemy_cols.insert(*ex);
        enemy_diag1.insert(ex - ey);
        enemy_diag2.insert(ex + ey);
    }

    // Collect friendly piece positions for wiggle room
    let mut friendly_positions: Vec<(i64, i64)> = Vec::new();
    for ((px, py), p) in &board.pieces {
        if p.color == piece.color && (*px != from.x || *py != from.y) {
            friendly_positions.push((*px, *py));
        }
    }

    for (dx_raw, dy_raw) in directions {
        for sign in [1i64, -1i64] {
            let dir_x = *dx_raw * sign;
            let dir_y = *dy_raw * sign;
            
            if dir_x == 0 && dir_y == 0 {
                continue;
            }

            let is_vertical = dir_x == 0;
            let is_horizontal = dir_y == 0;

            // Find closest blocker on this ray
            let mut closest_dist: Option<i64> = None;
            let mut closest_is_enemy = false;

            for (px, py) in &all_positions {
                let dx = px - from.x;
                let dy = py - from.y;

                // Check if this piece is on our ray
                let on_ray = if is_vertical {
                    dx == 0 && dy.signum() == dir_y.signum()
                } else if is_horizontal {
                    dy == 0 && dx.signum() == dir_x.signum()
                } else {
                    // Diagonal: check if (dx, dy) is a positive multiple of (dir_x, dir_y)
                    if dir_x != 0 && dir_y != 0 {
                        dx.abs() == dy.abs() && dx.signum() == dir_x.signum() && dy.signum() == dir_y.signum()
                    } else {
                        false
                    }
                };

                if on_ray {
                    let dist = if is_vertical { dy.abs() } else { dx.abs() };
                    if dist > 0 && closest_dist.map_or(true, |d| dist < d) {
                        closest_dist = Some(dist);
                        closest_is_enemy = enemy_positions.contains(&(*px, *py));
                    }
                }
            }

            let effective_max = closest_dist.unwrap_or(MAX_SLIDE_CHECK);

            // Generate moves along this ray
            for d in 1..=effective_max {
                let sq_x = from.x + dir_x * d;
                let sq_y = from.y + dir_y * d;
                
                // Check world border bounds
                if !in_bounds(sq_x, sq_y) { break; }

                // Can only land on the closest piece if it's an enemy
                let at_blocker = Some(d) == closest_dist;
                if at_blocker && !closest_is_enemy {
                    break; // Can't capture own piece, stop here
                }

                // Determine if this square is "interesting" for infinite chess
                let mut aligned = false;
                let mut wiggled = false;

                // Check alignment: can we attack an enemy from this square?
                if is_vertical {
                    // Moving vertically, check if we can attack horizontally or diagonally
                    if can_horiz && enemy_rows.contains(&sq_y) && has_clear_enemy_row(sq_x, sq_y) {
                        aligned = true;
                    }
                    if !aligned && can_diag {
                        if (enemy_diag1.contains(&(sq_x - sq_y)) || enemy_diag2.contains(&(sq_x + sq_y))) 
                           && has_clear_enemy_diag(sq_x, sq_y) {
                            aligned = true;
                        }
                    }
                } else if is_horizontal {
                    // Moving horizontally, check if we can attack vertically or diagonally
                    if can_vert && enemy_cols.contains(&sq_x) && has_clear_enemy_col(sq_x, sq_y) {
                        aligned = true;
                    }
                    if !aligned && can_diag {
                        if (enemy_diag1.contains(&(sq_x - sq_y)) || enemy_diag2.contains(&(sq_x + sq_y))) 
                           && has_clear_enemy_diag(sq_x, sq_y) {
                            aligned = true;
                        }
                    }
                } else {
                    // Moving diagonally, check if we can attack orthogonally or on other diagonals
                    if can_horiz && enemy_rows.contains(&sq_y) && has_clear_enemy_row(sq_x, sq_y) {
                        aligned = true;
                    }
                    if !aligned && can_vert && enemy_cols.contains(&sq_x) && has_clear_enemy_col(sq_x, sq_y) {
                        aligned = true;
                    }
                    if !aligned && can_diag {
                        // Check the OTHER diagonal
                        let our_diag1 = from.x - from.y;
                        let our_diag2 = from.x + from.y;
                        let sq_diag1 = sq_x - sq_y;
                        let sq_diag2 = sq_x + sq_y;
                        
                        // Only check if enemy is on a different diagonal that we can reach
                        if (enemy_diag1.contains(&sq_diag1) && sq_diag1 != our_diag1) 
                           || (enemy_diag2.contains(&sq_diag2) && sq_diag2 != our_diag2) {
                            if has_clear_enemy_diag(sq_x, sq_y) {
                                aligned = true;
                            }
                        }
                    }
                }

                // Wiggle room: near starting position
                if !aligned && !at_blocker && d <= wiggle_room {
                    wiggled = true;
                }

                // Wiggle room: near friendly pieces (project their position onto our ray)
                if !aligned && !wiggled && !at_blocker {
                    for (fx, fy) in &friendly_positions {
                        // Check if friendly piece is within wiggle room of this square
                        let dist_to_friendly = if is_vertical {
                            // For vertical movement, check if friendly is on same column within wiggle room
                            if (fx - sq_x).abs() <= friend_wiggle_room {
                                (fy - from.y).abs()
                            } else {
                                i64::MAX
                            }
                        } else if is_horizontal {
                            // For horizontal movement, check if friendly is on same row within wiggle room
                            if (fy - sq_y).abs() <= friend_wiggle_room {
                                (fx - from.x).abs()
                            } else {
                                i64::MAX
                            }
                        } else {
                            i64::MAX // Skip for diagonal for simplicity
                        };

                        if dist_to_friendly != i64::MAX {
                            // Check if our current step d is within wiggle room of the friend's projection
                            if (d as i64 - dist_to_friendly).abs() <= friend_wiggle_room {
                                wiggled = true;
                                break;
                            }
                        }
                    }
                }

                // Wiggle room: near the blocker piece
                if !aligned && !wiggled && closest_dist.is_some() {
                    let blocker_dist = closest_dist.unwrap();
                    let wr = if closest_is_enemy { wiggle_room } else { friend_wiggle_room };
                    if d >= blocker_dist.saturating_sub(wr) && d <= blocker_dist {
                        wiggled = true;
                    }
                }

                // Add move if interesting
                if at_blocker || aligned || wiggled {
                    moves.push(Move::new(from.clone(), Coordinate::new(sq_x, sq_y), piece.clone()));
                }
            }
        }
    }
    moves
}

fn generate_huygen_moves(board: &Board, from: &Coordinate, piece: &Piece, indices: Option<&SpatialIndices>) -> Vec<Move> {
    let mut moves = Vec::new();
    let directions = [(1, 0), (0, 1)];

    for (dx_raw, dy_raw) in directions {
        for sign in [1, -1] {
            let dir_x = dx_raw * sign;
            let dir_y = dy_raw * sign;
            
            let mut closest_prime_dist: Option<i64> = None;
            let mut closest_piece_color: Option<PlayerColor> = None;

            let mut found_via_indices = false;
            if let Some(indices) = indices {
                 let line_vec = if dx_raw == 0 { indices.cols.get(&from.x) } else { indices.rows.get(&from.y) };
                 if let Some(vec) = line_vec {
                     let val = if dx_raw == 0 { from.y } else { from.x };
                     if let Ok(idx) = vec.binary_search(&val) {
                         let step_dir = if dx_raw == 0 { dir_y } else { dir_x };
                         if step_dir > 0 {
                             for i in (idx + 1)..vec.len() {
                                 let next_val = vec[i];
                                 let dist = next_val - val;
                                 if is_prime_i64(dist) {
                                     closest_prime_dist = Some(dist);
                                     let (tx, ty) = if dx_raw == 0 { (from.x, next_val) } else { (next_val, from.y) };
                                     if let Some(p) = board.get_piece(&tx, &ty) {
                                         // Treat Void as friendly for capture purposes
                                         closest_piece_color = Some(if p.piece_type == PieceType::Void { piece.color } else { p.color });
                                     }
                                     break;
                                 }
                             }
                         } else {
                             for i in (0..idx).rev() {
                                 let prev_val = vec[i];
                                 let dist = val - prev_val;
                                 if is_prime_i64(dist) {
                                     closest_prime_dist = Some(dist);
                                     let (tx, ty) = if dx_raw == 0 { (from.x, prev_val) } else { (prev_val, from.y) };
                                     if let Some(p) = board.get_piece(&tx, &ty) {
                                         // Treat Void as friendly for capture purposes
                                         closest_piece_color = Some(if p.piece_type == PieceType::Void { piece.color } else { p.color });
                                     }
                                     break;
                                 }
                             }
                         }
                         found_via_indices = true; 
                     }
                 }
            }

            if !found_via_indices {
                for ((px, py), target_piece) in &board.pieces {
                    let dx = px - from.x;
                    let dy = py - from.y;
                    let k = if dir_x != 0 { if dx % dir_x == 0 && dy == 0 { Some(dx / dir_x) } else { None } } else { if dy % dir_y == 0 && dx == 0 { Some(dy / dir_y) } else { None } };

                    if let Some(dist) = k {
                        if dist > 0 {
                            if is_prime_i64(dist) {
                                if closest_prime_dist.as_ref().map_or(true, |d| dist < *d) {
                                    closest_prime_dist = Some(dist);
                                    // Treat Void as friendly for capture purposes
                                    closest_piece_color = Some(if target_piece.piece_type == PieceType::Void { piece.color } else { target_piece.color });
                                }
                            }
                        }
                    }
                }
            }

            let limit = closest_prime_dist.unwrap_or(100); 
            let scan_limit = if closest_prime_dist.is_some() { limit } else { 50 };
            
            for s in 2..=scan_limit {
                if is_prime_i64(s) {
                    let to_x = from.x + (dir_x * s);
                    let to_y = from.y + (dir_y * s);
                    
                    {
                        if s == limit && closest_prime_dist.is_some() {
                            if closest_piece_color != Some(piece.color) {
                                moves.push(Move::new(from.clone(), Coordinate::new(to_x, to_y), piece.clone()));
                            }
                        } else if s < limit {
                            moves.push(Move::new(from.clone(), Coordinate::new(to_x, to_y), piece.clone()));
                        }
                    }
                }
            }
        }
    }
    moves
}

fn generate_rose_moves(board: &Board, from: &Coordinate, piece: &Piece) -> Vec<Move> {
    let mut moves = Vec::new();
    let knight_moves = [
        (-2, -1), (-1, -2), (1, -2), (2, -1),
        (2, 1), (1, 2), (-1, 2), (-2, 1)
    ];
    
    for (start_idx, _) in knight_moves.iter().enumerate() {
        for direction in [1, -1] {
            let mut current_x = from.x;
            let mut current_y = from.y;
            let mut current_idx = start_idx as i32;
            
            for _ in 0..7 {
                let idx = (current_idx as usize) % 8;
                let (dx, dy) = knight_moves[idx];
                
                current_x += dx;
                current_y += dy;

                if let Some(target) = board.get_piece(&current_x, &current_y) {
                    if is_enemy_piece(target, piece.color) {
                        moves.push(Move::new(from.clone(), Coordinate::new(current_x, current_y), piece.clone()));
                    }
                    break; 
                } else {
                    moves.push(Move::new(from.clone(), Coordinate::new(current_x, current_y), piece.clone()));
                }
                
                current_idx += direction;
                if current_idx < 0 { current_idx += 8; }
            }
        }
    }
    
    moves
}
