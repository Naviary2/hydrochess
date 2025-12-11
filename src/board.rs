use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Number of piece types (for packed piece encoding)
// ============================================================================

/// Total number of piece types in the game (22 types: Void..Pawn)
pub const NUM_PIECE_TYPES: u8 = 22;

// ============================================================================
// Coordinate
// ============================================================================

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Coordinate {
    pub x: i64,
    pub y: i64,
}

impl Coordinate {
    #[inline]
    pub fn new(x: i64, y: i64) -> Self {
        Coordinate { x, y }
    }

    #[inline]
    pub fn get_x(&self) -> i64 {
        self.x
    }

    #[inline]
    pub fn get_y(&self) -> i64 {
        self.y
    }
}

// ============================================================================
// PlayerColor
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PlayerColor {
    Neutral = 0,
    White = 1,
    Black = 2,
}

impl PlayerColor {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "n" | "neutral" => Some(PlayerColor::Neutral),
            "w" | "white" => Some(PlayerColor::White),
            "b" | "black" => Some(PlayerColor::Black),
            _ => None,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            PlayerColor::Neutral => "n",
            PlayerColor::White => "w",
            PlayerColor::Black => "b",
        }
    }

    #[inline]
    pub fn opponent(&self) -> Self {
        match self {
            PlayerColor::White => PlayerColor::Black,
            PlayerColor::Black => PlayerColor::White,
            PlayerColor::Neutral => PlayerColor::Neutral,
        }
    }

    /// Convert from u8
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        unsafe { std::mem::transmute(v) }
    }
}

// ============================================================================
// PieceType
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PieceType {
    Void = 0,
    Obstacle = 1,
    King = 2,
    Giraffe = 3,
    Camel = 4,
    Zebra = 5,
    Knightrider = 6,
    Amazon = 7,
    Queen = 8,
    RoyalQueen = 9,
    Hawk = 10,
    Chancellor = 11,
    Archbishop = 12,
    Centaur = 13,
    RoyalCentaur = 14,
    Rose = 15,
    Knight = 16,
    Guard = 17,
    Huygen = 18,
    Rook = 19,
    Bishop = 20,
    Pawn = 21,
}

impl PieceType {
    /// Parse piece type from single-character string (matches JS typeutil)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "v" => Some(PieceType::Void),
            "x" => Some(PieceType::Obstacle),
            "k" => Some(PieceType::King),
            "i" => Some(PieceType::Giraffe),
            "l" => Some(PieceType::Camel),
            "z" => Some(PieceType::Zebra),
            "s" => Some(PieceType::Knightrider),
            "m" => Some(PieceType::Amazon),
            "q" => Some(PieceType::Queen),
            "y" => Some(PieceType::RoyalQueen),
            "h" => Some(PieceType::Hawk),
            "c" => Some(PieceType::Chancellor),
            "a" => Some(PieceType::Archbishop),
            "e" => Some(PieceType::Centaur),
            "d" => Some(PieceType::RoyalCentaur),
            "o" => Some(PieceType::Rose),
            "n" => Some(PieceType::Knight),
            "g" => Some(PieceType::Guard),
            "u" => Some(PieceType::Huygen),
            "r" => Some(PieceType::Rook),
            "b" => Some(PieceType::Bishop),
            "p" => Some(PieceType::Pawn),
            _ => None,
        }
    }

    /// Convert piece type to single-character string
    pub fn to_str(&self) -> &'static str {
        match self {
            PieceType::Void => "v",
            PieceType::Obstacle => "x",
            PieceType::King => "k",
            PieceType::Giraffe => "i",
            PieceType::Camel => "l",
            PieceType::Zebra => "z",
            PieceType::Knightrider => "s",
            PieceType::Amazon => "m",
            PieceType::Queen => "q",
            PieceType::RoyalQueen => "y",
            PieceType::Hawk => "h",
            PieceType::Chancellor => "c",
            PieceType::Archbishop => "a",
            PieceType::Centaur => "e",
            PieceType::RoyalCentaur => "d",
            PieceType::Rose => "o",
            PieceType::Knight => "n",
            PieceType::Guard => "g",
            PieceType::Huygen => "u",
            PieceType::Rook => "r",
            PieceType::Bishop => "b",
            PieceType::Pawn => "p",
        }
    }

    /// Check if this piece type is a neutral/blocking type (can't be moved by players)
    #[inline]
    pub fn is_neutral_type(&self) -> bool {
        matches!(self, PieceType::Void | PieceType::Obstacle)
    }

    /// Check if this piece type is a royal (king-like) piece
    #[inline]
    pub fn is_royal(&self) -> bool {
        matches!(
            self,
            PieceType::King | PieceType::RoyalQueen | PieceType::RoyalCentaur
        )
    }

    /// Get all promotable piece types (for dynamic promotion)
    pub fn promotable_types() -> &'static [PieceType] {
        &[
            PieceType::Queen,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Knight,
            PieceType::Chancellor,
            PieceType::Archbishop,
            PieceType::Amazon,
            PieceType::Hawk,
            PieceType::Guard,
            PieceType::Centaur,
            PieceType::Camel,
            PieceType::Giraffe,
            PieceType::Zebra,
            PieceType::Knightrider,
            PieceType::Rose,
            PieceType::Huygen,
        ]
    }

    /// Convert from u8
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        unsafe { std::mem::transmute(v) }
    }
}

// ============================================================================
// Piece - Packed representation (1 byte)
// ============================================================================

/// Packed piece representation: encodes both piece type and color in a single byte.
///
/// Encoding: `packed = color * NUM_PIECE_TYPES + piece_type`
/// - This matches the infinitechess.org JS encoding
/// - Allows 3 colors × 22 types = 66 values (fits in 7 bits)
///
/// For compatibility, we provide `piece_type()` and `color()` accessor methods
/// that decode the packed value.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Piece(u8);

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Piece")
            .field("piece_type", &self.piece_type())
            .field("color", &self.color())
            .field("packed", &self.0)
            .finish()
    }
}

// Custom serde to preserve JSON compatibility with old format
impl Serialize for Piece {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Piece", 2)?;
        s.serialize_field("piece_type", &self.piece_type())?;
        s.serialize_field("color", &self.color())?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for Piece {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct PieceFields {
            piece_type: PieceType,
            color: PlayerColor,
        }
        let fields = PieceFields::deserialize(deserializer)?;
        Ok(Piece::new(fields.piece_type, fields.color))
    }
}

impl Piece {
    /// Create a new piece from type and color
    #[inline]
    pub fn new(piece_type: PieceType, color: PlayerColor) -> Self {
        Piece((color as u8) * NUM_PIECE_TYPES + (piece_type as u8))
    }

    /// Create a piece from a packed u8 value
    #[inline]
    pub fn from_packed(packed: u8) -> Self {
        Piece(packed)
    }

    /// Get the raw packed value
    #[inline]
    pub fn packed(&self) -> u8 {
        self.0
    }

    /// Decode the piece type
    #[inline]
    pub fn piece_type(&self) -> PieceType {
        PieceType::from_u8(self.0 % NUM_PIECE_TYPES)
    }

    /// Decode the color
    #[inline]
    pub fn color(&self) -> PlayerColor {
        PlayerColor::from_u8(self.0 / NUM_PIECE_TYPES)
    }
}

// ============================================================================
// Board
// ============================================================================

#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "BoardRaw", into = "BoardRaw")]
pub struct Board {
    // Pure sorted Vec implementation
    // We keep 'pieces_vec' private to enforce sorted invariant
    pieces: Vec<((i64, i64), Piece)>,
    #[serde(skip)]
    pub active_coords: Option<HashSet<(i64, i64)>>,
}

/// Raw representation for serialization
#[derive(Serialize, Deserialize)]
struct BoardRaw {
    pieces: HashMap<(i64, i64), Piece>,
}

impl From<BoardRaw> for Board {
    fn from(raw: BoardRaw) -> Self {
        let has_neutral = raw
            .pieces
            .values()
            .any(|p| p.piece_type().is_neutral_type());

        let active_coords = if has_neutral {
            let mut set = HashSet::new();
            for (pos, piece) in &raw.pieces {
                if !piece.piece_type().is_neutral_type() {
                    set.insert(*pos);
                }
            }
            Some(set)
        } else {
            None
        };

        let mut v: Vec<_> = raw.pieces.into_iter().collect();
        v.sort_by_key(|(k, _)| *k);

        Board {
            pieces: v,
            active_coords,
        }
    }
}

impl From<Board> for BoardRaw {
    fn from(board: Board) -> Self {
        BoardRaw {
            pieces: board.pieces.into_iter().collect(),
        }
    }
}

pub struct BoardIter<'a> {
    iter: std::slice::Iter<'a, ((i64, i64), Piece)>,
}

impl<'a> Iterator for BoardIter<'a> {
    type Item = (&'a (i64, i64), &'a Piece);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, v)| (k, v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for BoardIter<'_> {}

impl Board {
    pub fn new() -> Self {
        Board {
            pieces: Vec::new(),
            active_coords: None,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn iter(&self) -> BoardIter<'_> {
        BoardIter {
            iter: self.pieces.iter(),
        }
    }

    pub fn set_piece(&mut self, x: i64, y: i64, piece: Piece) {
        let pos = (x, y);

        // Active coords tracking
        if piece.piece_type().is_neutral_type() && self.active_coords.is_none() {
            // Need to initialize active_coords by scanning current pieces
            let mut set = HashSet::new();
            for (p_pos, p) in self.iter() {
                if !p.piece_type().is_neutral_type() {
                    set.insert(*p_pos);
                }
            }
            self.active_coords = Some(set);
        }

        match self.pieces.binary_search_by_key(&pos, |(p, _)| *p) {
            Ok(index) => {
                self.pieces[index].1 = piece;
            }
            Err(index) => {
                self.pieces.insert(index, (pos, piece));
            }
        }

        if let Some(ref mut active) = self.active_coords {
            if !piece.piece_type().is_neutral_type() {
                active.insert(pos);
            } else {
                active.remove(&pos);
            }
        }
    }

    #[inline]
    pub fn get_piece(&self, x: &i64, y: &i64) -> Option<&Piece> {
        let pos = (*x, *y);
        self.pieces
            .binary_search_by_key(&pos, |(p, _)| *p)
            .ok()
            .map(|i| &self.pieces[i].1)
    }

    pub fn remove_piece(&mut self, x: &i64, y: &i64) -> Option<Piece> {
        let pos = (*x, *y);
        let removed = if let Ok(index) = self.pieces.binary_search_by_key(&pos, |(p, _)| *p) {
            Some(self.pieces.remove(index).1)
        } else {
            None
        };

        if let Some(ref piece) = removed {
            if let Some(ref mut active) = self.active_coords {
                if !piece.piece_type().is_neutral_type() {
                    active.remove(&(*x, *y));
                }
            }
        }
        removed
    }

    pub fn clear(&mut self) {
        self.pieces.clear();
        self.active_coords = None;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_packing() {
        // Test all combinations of color and type
        for color in [PlayerColor::Neutral, PlayerColor::White, PlayerColor::Black] {
            for pt in [
                PieceType::Void,
                PieceType::Obstacle,
                PieceType::King,
                PieceType::Queen,
                PieceType::Rook,
                PieceType::Bishop,
                PieceType::Knight,
                PieceType::Pawn,
            ] {
                let piece = Piece::new(pt, color);
                assert_eq!(
                    piece.piece_type(),
                    pt,
                    "piece_type mismatch for {:?}/{:?}",
                    pt,
                    color
                );
                assert_eq!(
                    piece.color(),
                    color,
                    "color mismatch for {:?}/{:?}",
                    pt,
                    color
                );
            }
        }
    }

    #[test]
    fn test_piece_size() {
        assert_eq!(std::mem::size_of::<Piece>(), 1, "Piece should be 1 byte");
    }

    #[test]
    fn test_piece_packed_values() {
        // Test that packed values match JS encoding: color * NUM_TYPES + type
        let white_pawn = Piece::new(PieceType::Pawn, PlayerColor::White);
        assert_eq!(white_pawn.packed(), 1 * 22 + 21); // White=1, Pawn=21

        let black_king = Piece::new(PieceType::King, PlayerColor::Black);
        assert_eq!(black_king.packed(), 2 * 22 + 2); // Black=2, King=2
    }
}
