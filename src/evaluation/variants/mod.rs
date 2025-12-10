// Variant-specific evaluation modules
//
// Design: Only variants with custom logic have files here.
// All other variants use base.rs evaluation automatically.

pub mod chess;
pub mod confined_classical;
pub mod obstocean;
pub mod palace;
pub mod pawn_horde;

// Future variants can be added here:
// pub mod space;
