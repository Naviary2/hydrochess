use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::moves::{get_legal_moves, Move, is_square_attacked, SpatialIndices};
use crate::evaluation::{get_piece_value, calculate_initial_material};
use std::collections::HashSet;
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct EnPassantState {
    pub square: Coordinate,
    pub pawn_square: Coordinate,
}

/// Promotion ranks configuration for a variant
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct PromotionRanks {
    pub white: Vec<i64>,
    pub black: Vec<i64>,
}

/// Game rules that can vary between chess variants
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct GameRules {
    pub promotion_ranks: Option<PromotionRanks>,
    pub promotions_allowed: Option<Vec<String>>, // Piece type codes allowed for promotion
}

#[derive(Clone)]
pub struct UndoMove {
    pub captured_piece: Option<Piece>,
    pub old_en_passant: Option<EnPassantState>,
    pub old_special_rights: HashSet<Coordinate>,
    pub old_halfmove_clock: u32,
    pub special_rights_removed: Vec<Coordinate>, // Track which special rights were removed
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: Board,
    pub turn: PlayerColor,
    /// Special rights for pieces - includes both castling rights (kings/rooks) AND
    /// pawn double-move rights. A piece with its coordinate in this set has its special rights.
    pub special_rights: HashSet<Coordinate>,
    pub en_passant: Option<EnPassantState>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    pub material_score: i32, // Positive = White advantage
    pub game_rules: GameRules, // Variant-specific rules
    #[serde(skip)]
    pub hash_stack: Vec<u64>, // Position hashes for repetition detection
    #[serde(skip)]
    pub null_moves: u8, // Counter for null moves (for repetition detection)
    #[serde(skip)]
    pub white_piece_count: u16,
    #[serde(skip)]
    pub black_piece_count: u16,
}

// For backwards compatibility, keep castling_rights as an alias
impl GameState {
    /// Returns pieces that can castle (kings and rooks with special rights)
    pub fn castling_rights(&self) -> HashSet<Coordinate> {
        let mut rights = HashSet::new();
        for coord in &self.special_rights {
            if let Some(piece) = self.board.get_piece(&coord.x, &coord.y) {
                // Only include kings and rooks (not pawns) in castling rights
                if piece.piece_type == PieceType::King || 
                   piece.piece_type == PieceType::Rook ||
                   piece.piece_type == PieceType::RoyalCentaur {
                    rights.insert(coord.clone());
                }
            }
        }
        rights
    }
    
    /// Check if a piece at the given coordinate has its special rights
    pub fn has_special_right(&self, coord: &Coordinate) -> bool {
        self.special_rights.contains(coord)
    }
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            board: Board::new(),
            turn: PlayerColor::White,
            special_rights: HashSet::new(),
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            material_score: 0,
            game_rules: GameRules::default(),
            hash_stack: Vec::with_capacity(128),
            null_moves: 0,
            white_piece_count: 0,
            black_piece_count: 0,
        }
    }
    
    pub fn new_with_rules(game_rules: GameRules) -> Self {
        GameState {
            board: Board::new(),
            turn: PlayerColor::White,
            special_rights: HashSet::new(),
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            material_score: 0,
            game_rules,
            hash_stack: Vec::with_capacity(128),
            null_moves: 0,
            white_piece_count: 0,
            black_piece_count: 0,
        }
    }

    pub fn recompute_piece_counts(&mut self) {
        let mut white: u16 = 0;
        let mut black: u16 = 0;
        for (_, piece) in &self.board.pieces {
            match piece.color {
                PlayerColor::White => white = white.saturating_add(1),
                PlayerColor::Black => black = black.saturating_add(1),
                PlayerColor::Neutral => {},
            }
        }
        self.white_piece_count = white;
        self.black_piece_count = black;
    }

    #[inline]
    pub fn has_pieces(&self, color: PlayerColor) -> bool {
        match color {
            PlayerColor::White => self.white_piece_count > 0,
            PlayerColor::Black => self.black_piece_count > 0,
            PlayerColor::Neutral => false,
        }
    }
    
    /// Check for threefold repetition
    pub fn is_threefold(&self) -> bool {
        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }
        
        // Need at least 6 positions to have a potential threefold
        if self.hash_stack.len() < 6 {
            return false;
        }
        
        // Generate current position hash
        let current_hash = self.generate_hash();
        
        let mut repetitions_count = 1;
        // Only look back as far as halfmove_clock allows (captures/pawn moves reset repetition)
        let lookback = (self.halfmove_clock as usize).min(self.hash_stack.len());
        let from = self.hash_stack.len().saturating_sub(lookback);
        let to = self.hash_stack.len().saturating_sub(1);
        
        if to <= from {
            return false;
        }
        
        // Check every other position (same side to move)
        for hash_index in (from..to).rev().step_by(2) {
            if self.hash_stack[hash_index] == current_hash {
                repetitions_count += 1;
                
                if repetitions_count >= 3 {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Check if this is a lone king endgame (one side only has a king)
    pub fn is_lone_king_endgame(&self) -> bool {
        use crate::board::{PieceType, PlayerColor};
        
        let mut white_has_non_king = false;
        let mut black_has_non_king = false;
        
        for (_, piece) in &self.board.pieces {
            if piece.piece_type != PieceType::King {
                if piece.color == PlayerColor::White {
                    white_has_non_king = true;
                } else {
                    black_has_non_king = true;
                }
            }
        }
        
        // One side has only a king (or nothing)
        !white_has_non_king || !black_has_non_king
    }
    
    /// Check if position is a draw by 50-move rule
    pub fn is_fifty(&self) -> bool {
        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }
        self.halfmove_clock >= 100
    }
    
    /// Make a null move (just flip turn, for null move pruning)
    pub fn make_null_move(&mut self) {
        // Push current hash
        let current_hash = self.generate_hash();
        self.hash_stack.push(current_hash);
        
        // Clear en passant
        self.en_passant = None;
        
        // Flip turn
        self.turn = self.turn.opponent();
        
        self.null_moves += 1;
    }
    
    /// Unmake a null move
    pub fn unmake_null_move(&mut self) {
        // Pop hash
        self.hash_stack.pop();
        
        // Flip turn back
        self.turn = self.turn.opponent();
        
        self.null_moves -= 1;
    }
    
    /// Generate a hash for the current position
    pub fn generate_hash(&self) -> u64 {
        use crate::search::TranspositionTable;
        TranspositionTable::generate_hash(self)
    }



    /// Returns pseudo-legal moves. Legality (not leaving king in check) 
    /// is checked in the search after making each move.
    pub fn get_legal_moves(&self) -> Vec<Move> {
        get_legal_moves(&self.board, self.turn, &self.special_rights, &self.en_passant, &self.game_rules)
    }
    
    /// Check if the side that just moved left their royal piece(s) in check (illegal move).
    /// Call this AFTER make_move to verify legality.
    /// Checks all royal pieces: King, RoyalQueen, RoyalCentaur
    pub fn is_move_illegal(&self) -> bool {
        // After make_move, self.turn is the opponent.
        // We need to check if the side that just moved (opponent of current turn) has any royal in check.
        let moved_color = self.turn.opponent();
        let indices = SpatialIndices::new(&self.board);
        
        // Find ALL royal pieces of the side that just moved and check if any are attacked
        for ((x, y), piece) in &self.board.pieces {
            if piece.color == moved_color && piece.piece_type.is_royal() {
                let pos = Coordinate::new(*x, *y);
                if is_square_attacked(&self.board, &pos, self.turn, Some(&indices)) {
                    return true;
                }
            }
        }
        false
    }

    pub fn is_in_check(&self) -> bool {
        let indices = SpatialIndices::new(&self.board);
        let attacker_color = self.turn.opponent();
        
        // Check if ANY royal piece of current player is attacked
        for ((x, y), piece) in &self.board.pieces {
            if piece.color == self.turn && piece.piece_type.is_royal() {
                let pos = Coordinate::new(*x, *y);
                if is_square_attacked(&self.board, &pos, attacker_color, Some(&indices)) {
                    return true;
                }
            }
        }
        false
    }

    /// Make a move given just from/to coordinates and optional promotion.
    /// Like UCI - we trust the input is valid and just execute it directly.
    /// This is much faster than generating all legal moves for history replay.
    pub fn make_move_coords(&mut self, from_x: i64, from_y: i64, to_x: i64, to_y: i64, promotion: Option<&str>) {
        // Push current position hash BEFORE making the move
        let current_hash = self.generate_hash();
        self.hash_stack.push(current_hash);
        
        let piece = match self.board.remove_piece(&from_x, &from_y) {
            Some(p) => p,
            None => return, // No piece at from - invalid move, just skip
        };
        
        // Handle capture
        let captured = self.board.remove_piece(&to_x, &to_y);
        let is_capture = captured.is_some();
        
        if let Some(ref cap) = captured {
            let value = get_piece_value(cap.piece_type);
            if cap.color == PlayerColor::White {
                self.material_score -= value;
                self.white_piece_count = self.white_piece_count.saturating_sub(1);
            } else {
                self.material_score += value;
                self.black_piece_count = self.black_piece_count.saturating_sub(1);
            }
        }
        
        // Handle en passant capture
        let mut is_ep_capture = false;
        if piece.piece_type == PieceType::Pawn {
            if let Some(ep) = &self.en_passant {
                if to_x == ep.square.x && to_y == ep.square.y {
                    if let Some(captured_pawn) = self.board.remove_piece(&ep.pawn_square.x, &ep.pawn_square.y) {
                        is_ep_capture = true;
                        let value = get_piece_value(captured_pawn.piece_type);
                        if captured_pawn.color == PlayerColor::White {
                            self.material_score -= value;
                            self.white_piece_count = self.white_piece_count.saturating_sub(1);
                        } else {
                            self.material_score += value;
                            self.black_piece_count = self.black_piece_count.saturating_sub(1);
                        }
                    }
                }
            }
        }
        
        // Handle promotion material
        if let Some(promo_str) = promotion {
            let pawn_val = get_piece_value(PieceType::Pawn);
            if piece.color == PlayerColor::White {
                self.material_score -= pawn_val;
            } else {
                self.material_score += pawn_val;
            }
            
            let promo_type = PieceType::from_str(promo_str).unwrap_or(PieceType::Queen);
            let promo_val = get_piece_value(promo_type);
            if piece.color == PlayerColor::White {
                self.material_score += promo_val;
            } else {
                self.material_score -= promo_val;
            }
        }
        
        // Update special rights - moving piece loses its rights
        self.special_rights.remove(&Coordinate::new(from_x, from_y));
        // Captured piece (if any) loses its rights
        if is_capture {
            self.special_rights.remove(&Coordinate::new(to_x, to_y));
        }
        
        // Handle castling (king moves more than 1 square horizontally)
        if piece.piece_type == PieceType::King || piece.piece_type == PieceType::RoyalCentaur {
            let dx = to_x - from_x;
            if dx.abs() > 1 {
                // Find the rook BEYOND the king's destination (rook is outside the path)
                // e.g., kingside: king 5,1->7,1, rook at 8,1 moves to 6,1
                let rook_dir = if dx > 0 { 1 } else { -1 };
                let mut rook_x = to_x + rook_dir; // Start searching past king's destination
                while rook_x >= -1_000_000 && rook_x <= 1_000_000 {
                    if let Some(r) = self.board.get_piece(&rook_x, &from_y) {
                        if r.piece_type == PieceType::Rook && r.color == piece.color {
                            // Found the rook - move it to the square the king jumped over
                            let rook = self.board.remove_piece(&rook_x, &from_y).unwrap();
                            let rook_to_x = to_x - rook_dir; // Rook goes on the other side of king
                            self.board.set_piece(rook_to_x, from_y, rook);
                            self.special_rights.remove(&Coordinate::new(rook_x, from_y));
                            break;
                        }
                        break; // Hit a non-rook piece, stop searching
                    }
                    rook_x += rook_dir;
                }
            }
        }
        
        // Place the piece (with promotion if applicable)
        let final_piece = if let Some(promo_str) = promotion {
            let promo_type = PieceType::from_str(promo_str).unwrap_or(PieceType::Queen);
            Piece::new(promo_type, piece.color)
        } else {
            piece.clone()
        };
        self.board.set_piece(to_x, to_y, final_piece);
        
        // Update en passant state
        self.en_passant = None;
        if piece.piece_type == PieceType::Pawn {
            let dy = to_y - from_y;
            if dy.abs() == 2 {
                let ep_y = from_y + (dy / 2);
                self.en_passant = Some(EnPassantState {
                    square: Coordinate::new(from_x, ep_y),
                    pawn_square: Coordinate::new(to_x, to_y),
                });
            }
        }
        
        // Update clocks
        if piece.piece_type == PieceType::Pawn || is_capture || is_ep_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }
        
        if self.turn == PlayerColor::Black {
            self.fullmove_number += 1;
        }
        
        self.turn = self.turn.opponent();
    }

    pub fn make_move(&mut self, m: &Move) -> UndoMove {
        // Push current position hash BEFORE making the move (for repetition detection)
        let current_hash = self.generate_hash();
        self.hash_stack.push(current_hash);
        
        let piece = self.board.remove_piece(&m.from.x, &m.from.y).unwrap();
        
        let mut undo_info = UndoMove {
            captured_piece: self.board.get_piece(&m.to.x, &m.to.y).cloned(),
            old_en_passant: self.en_passant.clone(),
            old_special_rights: self.special_rights.clone(),
            old_halfmove_clock: self.halfmove_clock,
            special_rights_removed: Vec::new(),
        };

        // Handle captures (reset halfmove clock)
        let is_capture = undo_info.captured_piece.is_some();
        
        if let Some(captured) = &undo_info.captured_piece {
            let value = get_piece_value(captured.piece_type);
            if captured.color == PlayerColor::White {
                self.material_score -= value;
                self.white_piece_count = self.white_piece_count.saturating_sub(1);
            } else {
                self.material_score += value;
                self.black_piece_count = self.black_piece_count.saturating_sub(1);
            }
        }
        
        // Handle En Passant capture
        let mut is_ep_capture = false;
        if piece.piece_type == PieceType::Pawn {
            if let Some(ep) = &self.en_passant {
                if m.to.x == ep.square.x && m.to.y == ep.square.y {
                    if let Some(captured_pawn) = self.board.remove_piece(&ep.pawn_square.x, &ep.pawn_square.y) {
                        is_ep_capture = true;
                        // Update material for EP capture
                        let value = get_piece_value(captured_pawn.piece_type);
                        if captured_pawn.color == PlayerColor::White {
                            self.material_score -= value;
                            self.white_piece_count = self.white_piece_count.saturating_sub(1);
                        } else {
                            self.material_score += value;
                            self.black_piece_count = self.black_piece_count.saturating_sub(1);
                        }
                    }
                }
            }
        }

        // Handle Promotion
        if let Some(promo_str) = &m.promotion {
             // Remove pawn value
             let pawn_val = get_piece_value(PieceType::Pawn);
             if piece.color == PlayerColor::White {
                 self.material_score -= pawn_val;
             } else {
                 self.material_score += pawn_val;
             }
             
             // Add promoted piece value - use from_str for all piece types
             let promo_type = PieceType::from_str(promo_str.as_str())
                 .unwrap_or(PieceType::Queen);
             
             let promo_val = get_piece_value(promo_type);
             if piece.color == PlayerColor::White {
                 self.material_score += promo_val;
             } else {
                 self.material_score -= promo_val;
             }
        }

        // Update special rights (castling and pawn double-move)
        // Any piece that moves loses its special rights
        if self.special_rights.remove(&m.from) {
            undo_info.special_rights_removed.push(m.from.clone());
        }
        // If a piece with special rights is captured, it loses them
        if is_capture {
            if self.special_rights.remove(&m.to) {
                undo_info.special_rights_removed.push(m.to.clone());
            }
        }

        // Handle Castling Move (King moves > 1 square)
        if piece.piece_type == PieceType::King {
            let dx = m.to.x - m.from.x;
            if dx.abs() > 1 {
                // Castling!
                if let Some(rook_coord) = &m.rook_coord {
                     if let Some(rook) = self.board.remove_piece(&rook_coord.x, &rook_coord.y) {
                        let rook_to_x = m.from.x + (if dx > 0 { 1 } else { -1 });
                        self.board.set_piece(rook_to_x, m.from.y, rook);
                    }
                }
            }
        }

        // Move piece (handle promotion if needed)
        let final_piece = if let Some(promo_str) = &m.promotion {
             let promo_type = PieceType::from_str(promo_str.as_str())
                 .unwrap_or(PieceType::Queen);
             Piece::new(promo_type, piece.color)
        } else {
            piece.clone()
        };

        self.board.set_piece(m.to.x, m.to.y, final_piece);

        // Update En Passant state
        self.en_passant = None;
        if piece.piece_type == PieceType::Pawn {
            let dy = m.to.y - m.from.y;
            if dy.abs() == 2 {
                let ep_y = m.from.y + (dy / 2);
                self.en_passant = Some(EnPassantState {
                    square: Coordinate::new(m.from.x, ep_y),
                    pawn_square: m.to.clone(),
                });
            }
        }

        // Update clocks
        if piece.piece_type == PieceType::Pawn || is_capture || is_ep_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        if self.turn == PlayerColor::Black {
            self.fullmove_number += 1;
        }

        self.turn = self.turn.opponent();
        
        undo_info
    }

    pub fn undo_move(&mut self, m: &Move, undo: UndoMove) {
        // Pop the hash that was pushed in make_move
        self.hash_stack.pop();
        
        // Revert turn
        self.turn = self.turn.opponent();
        
        if self.turn == PlayerColor::Black {
            self.fullmove_number -= 1;
        }

        // Revert piece move
        // Get the piece from the 'to' square
        let mut piece = self.board.remove_piece(&m.to.x, &m.to.y).unwrap();
        
        // Handle Promotion Revert
        if m.promotion.is_some() {
            // Convert back to pawn
            let promo_val = get_piece_value(piece.piece_type);
            let pawn_val = get_piece_value(PieceType::Pawn);
            
            if piece.color == PlayerColor::White {
                self.material_score -= promo_val;
                self.material_score += pawn_val;
            } else {
                self.material_score += promo_val;
                self.material_score -= pawn_val;
            }
            piece.piece_type = PieceType::Pawn;
        }

        // Move back to 'from'
        self.board.set_piece(m.from.x, m.from.y, piece.clone());

        // Restore captured piece
        if let Some(captured) = undo.captured_piece {
            let value = get_piece_value(captured.piece_type);
            if captured.color == PlayerColor::White {
                self.material_score += value;
                self.white_piece_count = self.white_piece_count.saturating_add(1);
            } else {
                self.material_score -= value;
                self.black_piece_count = self.black_piece_count.saturating_add(1);
            }
            self.board.set_piece(m.to.x, m.to.y, captured);
        }

        // Handle En Passant Capture Revert
        // If it was an EP capture, the captured pawn was on 'pawn_square' of the OLD en_passant state
        // But wait, we don't store "is_ep_capture" in UndoMove.
        // We can infer it: if piece is pawn, and to_square matches old_ep.square
        if piece.piece_type == PieceType::Pawn {
             if let Some(ep) = &undo.old_en_passant {
                 if m.to.x == ep.square.x && m.to.y == ep.square.y {
                     // It was an EP capture!
                     // Restore the captured pawn
                     let captured_pawn = Piece::new(PieceType::Pawn, piece.color.opponent());
                     
                     self.board.set_piece(ep.pawn_square.x, ep.pawn_square.y, captured_pawn.clone());
                     
                     // Restore material
                     let value = get_piece_value(PieceType::Pawn);
                     if captured_pawn.color == PlayerColor::White {
                         self.material_score += value;
                         self.white_piece_count = self.white_piece_count.saturating_add(1);
                     } else {
                         self.material_score -= value;
                         self.black_piece_count = self.black_piece_count.saturating_add(1);
                     }
                 }
             }
        }

        // Handle Castling Revert
        if piece.piece_type == PieceType::King {
            let dx = m.to.x - m.from.x;
            if dx.abs() > 1 {
                // Castling was performed. Move rook back.
                if let Some(rook_coord) = &m.rook_coord {
                    let rook_to_x = m.from.x + (if dx > 0 { 1 } else { -1 });
                    if let Some(rook) = self.board.remove_piece(&rook_to_x, &m.from.y) {
                        self.board.set_piece(rook_coord.x, rook_coord.y, rook);
                    }
                }
            }
        }

        // Restore state
        self.en_passant = undo.old_en_passant;
        self.special_rights = undo.old_special_rights;
        self.halfmove_clock = undo.old_halfmove_clock;
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = self.get_legal_moves();
        let mut nodes = 0;

        for m in moves {
            let undo = self.make_move(&m);
            nodes += self.perft(depth - 1);
            self.undo_move(&m, undo);
        }

        nodes
    }

    pub fn setup_standard_chess(&mut self) {
        self.board = Board::new();
        self.special_rights.clear();
        self.en_passant = None;
        self.turn = PlayerColor::White;
        self.halfmove_clock = 0;
        self.fullmove_number = 1;
        self.material_score = 0;

        // White Pieces
        self.board.set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        self.board.set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        self.board.set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        self.board.set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        self.board.set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        self.board.set_piece(6, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        self.board.set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        self.board.set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White));

        for x in 1..=8 {
            self.board.set_piece(x, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        }

        // Black Pieces
        self.board.set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        self.board.set_piece(2, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        self.board.set_piece(3, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        self.board.set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
        self.board.set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        self.board.set_piece(6, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        self.board.set_piece(7, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        self.board.set_piece(8, 8, Piece::new(PieceType::Rook, PlayerColor::Black));

        for x in 1..=8 {
            self.board.set_piece(x, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
        }

        // Special Rights - Kings, Rooks (castling) and Pawns (double move)
        self.special_rights.insert(Coordinate::new(1, 1)); // Rook
        self.special_rights.insert(Coordinate::new(5, 1)); // King
        self.special_rights.insert(Coordinate::new(8, 1)); // Rook
        
        self.special_rights.insert(Coordinate::new(1, 8)); // Rook
        self.special_rights.insert(Coordinate::new(5, 8)); // King
        self.special_rights.insert(Coordinate::new(8, 8)); // Rook
        
        // Pawn double-move rights
        for x in 1..=8 {
            self.special_rights.insert(Coordinate::new(x, 2)); // White pawns
            self.special_rights.insert(Coordinate::new(x, 7)); // Black pawns
        }
        
        // Calculate initial material
        self.material_score = calculate_initial_material(&self.board);
    }
}
