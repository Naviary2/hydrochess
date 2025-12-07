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
///
/// For better performance on WASM (and generally), this implementation uses a
/// fixed-size array-backed table indexed directly by masked hash values
/// instead of a HashMap. This avoids dynamic allocations and hashing in hot
/// search code while preserving the public API.
pub struct TranspositionTable {
    table: Vec<Option<TTEntry>>,
    /// Bitmask for indexing into `table` (capacity is always a power of two).
    mask: usize,
    pub age: u8,
    /// Number of occupied slots, used for an approximate fill ratio.
    used: usize,
}

impl TranspositionTable {
    /// Create a new TT with approximately `size_mb` megabytes of storage.
    pub fn new(size_mb: usize) -> Self {
        // Convert megabytes to bytes and estimate how many TTEntry values fit.
        let bytes = size_mb.max(128) as usize * 1024 * 1024;
        let entry_size = std::mem::size_of::<TTEntry>().max(1);
        let capacity = (bytes / entry_size).max(1);

        // Round DOWN to the nearest power of two to simplify masking.
        let mut cap_pow2 = 1usize;
        while cap_pow2.saturating_mul(2) <= capacity {
            cap_pow2 = cap_pow2.saturating_mul(2);
        }

        let table = vec![None; cap_pow2];

        TranspositionTable {
            table,
            mask: cap_pow2 - 1,
            age: 0,
            used: 0,
        }
    }

    /// Get the hash for the current position (uses incrementally maintained hash)
    #[inline]
    pub fn generate_hash(game: &GameState) -> u64 {
        game.hash
    }

    /// Capacity (number of slots) in the underlying table.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.table.len()
    }

    /// Number of TT entries currently in use.
    #[inline]
    pub fn used_entries(&self) -> usize {
        self.used
    }

    /// Approximate transposition table fill in permille (0-1000).
    #[inline]
    pub fn fill_permille(&self) -> u32 {
        if self.table.is_empty() {
            return 0;
        }
        ((self.used as u64 * 1000) / self.table.len() as u64) as u32
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Probe the TT for a position.
    ///
    /// Returns `Some((score, best_move))` where:
    /// - If `score` is a valid cutoff score (not `INFINITY + 1`), the caller can use it directly.
    /// - If `score == INFINITY + 1`, the score is not usable but `best_move` may still be valid
    ///   for move ordering purposes.
    pub fn probe(&self, hash: u64, alpha: i32, beta: i32, depth: usize, ply: usize) -> Option<(i32, Option<Move>)> {
        let idx = self.index(hash);
        if let Some(entry) = &self.table[idx] {
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

                    // Check if we can use this score for a cutoff
                    let usable_score = match entry.flag {
                        TTFlag::Exact => Some(score),
                        TTFlag::LowerBound if score >= beta => Some(score),
                        TTFlag::UpperBound if score <= alpha => Some(score),
                        _ => None,
                    };

                    if let Some(s) = usable_score {
                        return Some((s, best_move));
                    }
                }

                // Depth too shallow or bounds don't allow cutoff: still return move for ordering.
                return Some((INFINITY + 1, best_move));
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

        let idx = self.index(hash);

        // Replacement strategy (depth-preferred with aging):
        // - Always replace empty slots
        // - Replace if existing entry is from an older search (different age)
        // - Replace if new entry has equal or greater depth
        // - Replace if new entry is exact and existing is not (same depth)
        let replace = match &self.table[idx] {
            None => true,
            Some(existing) => {
                self.age != existing.age  // Old entry from previous search
                    || depth > existing.depth as usize  // Deeper search
                    || (depth == existing.depth as usize && flag == TTFlag::Exact)  // Same depth, prefer exact
            }
        };

        if replace {
            if self.table[idx].is_none() {
                self.used += 1;
            }

            self.table[idx] = Some(TTEntry {
                hash,
                depth: depth as u8,
                flag,
                score: adjusted_score,
                best_move,
                age: self.age,
            });
        }
    }

    pub fn increment_age(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    pub fn clear(&mut self) {
        for slot in &mut self.table {
            *slot = None;
        }
        self.age = 0;
        self.used = 0;
    }
}
