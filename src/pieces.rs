use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerColor {
    White,
    Black,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Pawn,
    Knight,
    Hawk,
    King,
    Guard,
    Rook,
    Bishop,
    Queen,
    RoyalQueen,
    Chancellor,
    Archbishop,
    Amazon,
    Camel,
    Giraffe,
    Zebra,
    Knightrider,
    Centaur,
    RoyalCentaur,
    Huygen,
    Rose,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: PlayerColor,
    pub has_moved: bool,
}

impl Piece {
    pub fn new(piece_type: PieceType, color: PlayerColor) -> Self {
        Piece {
            piece_type,
            color,
            has_moved: false,
        }
    }
}
