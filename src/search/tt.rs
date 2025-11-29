use std::collections::HashMap;

use crate::game::GameState;
use crate::moves::Move;

use super::{INFINITY, MATE_SCORE};

// TT Entry flags
#[derive(Clone, Copy, PartialEq)]
pub enum TTFlag {
    Exact,
    LowerBound, // Failed low (score is at most this)
    UpperBound, // Failed high (score is at least this)
}

/// Transposition Table entry for infinite chess
#[derive(Clone)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: u8,
    pub flag: TTFlag,
    pub score: i32,
    pub best_move: Option<Move>,
    pub age: u8,
}

/// Transposition Table adapted for infinite chess (coordinate-based hashing)
pub struct TranspositionTable {
    pub table: HashMap<u64, TTEntry>,
    pub size: usize,
    pub age: u8,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        // Rough estimate: each entry ~100 bytes
        let size = (size_mb * 1024 * 1024) / 100;
        TranspositionTable {
            table: HashMap::with_capacity(size),
            size,
            age: 0,
        }
    }
    
    /// Generate a hash for the current board position (infinite chess adapted)
    pub fn generate_hash(game: &GameState) -> u64 {
        let mut hash: u64 = 0;
        
        for ((x, y), piece) in &game.board.pieces {
            // Normalize coordinates for hashing (handle large coords)
            let norm_x = normalize_coord(*x);
            let norm_y = normalize_coord(*y);
            
            // Combine coordinates
            let coord_hash = (norm_x as u64) ^ ((norm_y as u64) << 16);
            
            // Mix with piece type and color
            let piece_val = (piece.piece_type as u64) | ((piece.color as u64) << 8);
            let mixed = mix_bits(coord_hash ^ piece_val);
            
            hash ^= mixed;
        }
        
        // Mix in the turn
        hash ^= (game.turn as u64) * 0x9E3779B97F4A7C15;
        
        mix_bits(hash)
    }
    
    /// Probe the TT for a position
    pub fn probe(&self, hash: u64, alpha: i32, beta: i32, depth: usize, ply: usize) -> Option<(i32, Option<Move>)> {
        if let Some(entry) = self.table.get(&hash) {
            if entry.hash == hash {
                // Always return the best move for move ordering
                let best_move = entry.best_move.clone();
                
                // Only use score if depth is sufficient
                if entry.depth as usize >= depth {
                    let mut score = entry.score;
                    
                    // Adjust mate scores for current ply
                    if score > MATE_SCORE {
                        score -= ply as i32;
                    } else if score < -MATE_SCORE {
                        score += ply as i32;
                    }
                    
                    match entry.flag {
                        TTFlag::Exact => return Some((score, best_move)),
                        TTFlag::LowerBound if score >= beta => return Some((beta, best_move)),
                        TTFlag::UpperBound if score <= alpha => return Some((alpha, best_move)),
                        _ => return Some((INFINITY + 1, best_move)), // Signal: use move but not score
                    }
                }
                
                return Some((INFINITY + 1, best_move)); // Return move for ordering
            }
        }
        None
    }
    
    /// Store an entry in the TT
    pub fn store(&mut self, hash: u64, depth: usize, flag: TTFlag, score: i32, best_move: Option<Move>, ply: usize) {
        // Adjust mate scores for storage
        let mut adjusted_score = score;
        if score > MATE_SCORE {
            adjusted_score += ply as i32;
        } else if score < -MATE_SCORE {
            adjusted_score -= ply as i32;
        }
        
        // Replacement strategy: replace if deeper, same position, or older
        let should_replace = if let Some(existing) = self.table.get(&hash) {
            existing.hash != hash || // Different position (collision)
            depth >= existing.depth as usize || // Deeper search
            self.age != existing.age || // Older entry
            flag == TTFlag::Exact // Exact scores are valuable
        } else {
            true
        };
        
        if should_replace {
            self.table.insert(hash, TTEntry {
                hash,
                depth: depth as u8,
                flag,
                score: adjusted_score,
                best_move,
                age: self.age,
            });
        }
        
        // Cleanup if table is too large
        if self.table.len() > self.size {
            self.cleanup_old_entries();
        }
    }
    
    pub fn increment_age(&mut self) {
        self.age = self.age.wrapping_add(1);
    }
    
    fn cleanup_old_entries(&mut self) {
        let current_age = self.age;
        self.table.retain(|_, entry| {
            current_age.wrapping_sub(entry.age) < 3
        });
    }

    pub fn clear(&mut self) {
        self.table.clear();
        self.age = 0;
    }
}

/// Normalize coordinate for hashing (handle infinite board)
#[inline]
fn normalize_coord(coord: i64) -> i32 {
    const BOUND: i64 = 150;
    const BUCKETS: i64 = 8;
    
    if coord.abs() <= BOUND {
        coord as i32
    } else {
        let sign = coord.signum();
        let delta = (coord - sign * BOUND) % BUCKETS;
        (sign * BOUND + delta) as i32
    }
}

/// Bit mixing function for better hash distribution
#[inline]
fn mix_bits(mut n: u64) -> u64 {
    n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    n = (n ^ (n >> 27)).wrapping_mul(0x94d049bb133111eb);
    n ^ (n >> 31)
}
