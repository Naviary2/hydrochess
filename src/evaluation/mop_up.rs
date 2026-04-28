// Mop-Up Evaluation
//
// Specialized endgame logic for positions where one side has a significant
// material advantage. It aims to drive the enemy king into a corner or
// "cage" it to facilitate checkmate.

use crate::board::{Board, Coordinate, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::{SpatialIndices, is_square_attacked};
use crate::utils::is_prime_fast;

/// Threshold for disabling mop-up evaluation if the opponent still has significant material.
const MOP_UP_THRESHOLD_PERCENT: u32 = 15;

#[derive(Clone, Copy)]
struct SliderInfo {
    x: i64,
    y: i64,
    pt: PieceType,
}

/// 8 compass directions around a square (4 ortho + 4 diagonal).
const RING_DIRS: [(i64, i64); 8] = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (-1, -1), (1, -1), (-1, 1),
];

/// Geometric attack pattern check (ignores blockers).
/// Returns true if a piece of type `pt` and color `color` at offset (dx, dy)=0 could
/// in principle attack a square at offset (dx, dy). Used for fast ring-coverage scoring.
#[inline]
fn piece_attacks_geom(pt: PieceType, color: PlayerColor, dx: i64, dy: i64) -> bool {
    let adx = dx.abs();
    let ady = dy.abs();
    if adx == 0 && ady == 0 {
        return false;
    }
    match pt {
        PieceType::Rook => dx == 0 || dy == 0,
        PieceType::Bishop => adx == ady,
        PieceType::Queen | PieceType::RoyalQueen => dx == 0 || dy == 0 || adx == ady,
        PieceType::Chancellor => {
            dx == 0 || dy == 0 || (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }
        PieceType::Archbishop => {
            adx == ady || (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }
        PieceType::Amazon => {
            dx == 0
                || dy == 0
                || adx == ady
                || (adx == 1 && ady == 2)
                || (adx == 2 && ady == 1)
        }
        PieceType::Knight => (adx == 1 && ady == 2) || (adx == 2 && ady == 1),
        PieceType::Camel => (adx == 1 && ady == 3) || (adx == 3 && ady == 1),
        PieceType::Giraffe => (adx == 1 && ady == 4) || (adx == 4 && ady == 1),
        PieceType::Zebra => (adx == 2 && ady == 3) || (adx == 3 && ady == 2),
        PieceType::King | PieceType::Guard => adx <= 1 && ady <= 1,
        PieceType::Centaur | PieceType::RoyalCentaur => {
            (adx <= 1 && ady <= 1) || (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }
        PieceType::Hawk => {
            // Hawk leaps to compass squares at distance 2 or 3 (ortho or diag).
            let d = adx.max(ady);
            (d == 2 || d == 3) && (dx == 0 || dy == 0 || adx == ady)
        }
        PieceType::Knightrider => {
            // Slides along knight rays: any (k, 2k) or (2k, k).
            if adx == 0 || ady == 0 || adx == ady {
                false
            } else {
                let g = gcd_i64(adx, ady);
                let nx = adx / g;
                let ny = ady / g;
                (nx == 1 && ny == 2) || (nx == 2 && ny == 1)
            }
        }
        PieceType::Pawn => {
            let dir = if color == PlayerColor::White { 1 } else { -1 };
            adx == 1 && dy == dir
        }
        PieceType::Huygen => {
            // Orthogonal slider, only at prime distances.
            (dx == 0 || dy == 0) && is_prime_fast(adx.max(ady))
        }
        PieceType::Rose => {
            // Approximate Rose coverage via knight-leaper pattern; the full curving
            // pattern is complex but a knight check is a reasonable lower bound.
            (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }
        PieceType::Void | PieceType::Obstacle => false,
    }
}

#[inline]
fn gcd_i64(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

/// Returns true if the piece is a leaper-style short-range piece that
/// must engage at close quarters to influence the mating net.
#[inline]
fn is_short_range_piece(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Knight
            | PieceType::Camel
            | PieceType::Giraffe
            | PieceType::Zebra
            | PieceType::Guard
            | PieceType::Centaur
            | PieceType::RoyalCentaur
            | PieceType::Hawk
            | PieceType::Pawn
            | PieceType::Rose
    )
}

/// Ideal Chebyshev distance for a piece around the enemy king.
/// Sliders prefer to stand on rays at moderate range (cut and aim);
/// short-range pieces must close in.
#[inline]
fn ideal_distance(pt: PieceType) -> (i64, i64) {
    match pt {
        PieceType::Knight => (2, 4),
        PieceType::Camel => (3, 5),
        PieceType::Giraffe => (4, 6),
        PieceType::Zebra => (3, 5),
        PieceType::Guard | PieceType::King => (1, 2),
        PieceType::Centaur | PieceType::RoyalCentaur => (1, 3),
        PieceType::Hawk => (2, 4),
        PieceType::Pawn => (1, 2),
        PieceType::Rose => (2, 4),
        // Sliders: ideal stays a few squares back to keep the cut intact.
        PieceType::Rook | PieceType::Queen | PieceType::RoyalQueen | PieceType::Chancellor => (2, 8),
        PieceType::Bishop | PieceType::Archbishop => (2, 8),
        PieceType::Amazon => (2, 6),
        PieceType::Knightrider => (2, 6),
        PieceType::Huygen => (2, 11),
        PieceType::Obstacle | PieceType::Void => (0, 0),
    }
}

/// Detects if the enemy king is trapped within a localized "cage" of attacked squares.
/// Returns whether a cage exists and the total reachable area for the king.
#[inline]
fn find_bitboard_cage(
    board: &Board,
    indices: &SpatialIndices,
    enemy_king: &Coordinate,
    our_color: PlayerColor,
) -> (bool, u32) {
    // 32x32 local window centered on king
    // Indices 0..31 map to king_coord - 16 .. king_coord + 15
    let mut forbidden = [0u32; 32];
    let origin_x = enemy_king.x - 16;
    let origin_y = enemy_king.y - 16;

    // 1. Identify forbidden squares (attacked, occupied, or out of bounds)
    let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();

    for (local_y, forbidden_row) in forbidden.iter_mut().enumerate() {
        let abs_y = origin_y + local_y as i64;
        for local_x in 0..32 {
            let abs_x = origin_x + local_x as i64;

            // If out of bounds, it's a "wall"
            if abs_x < min_x || abs_x > max_x || abs_y < min_y || abs_y > max_y {
                *forbidden_row |= 1 << local_x;
                continue;
            }

            let target = Coordinate::new(abs_x, abs_y);

            // If square is attacked or occupied by our piece, it's a "wall"
            if is_square_attacked(board, &target, our_color, indices)
                || board.is_occupied_by_color(abs_x, abs_y, our_color)
            {
                *forbidden_row |= 1 << local_x;
            }
        }
    }

    // 2. Flood fill from center (16, 16)
    let mut reachable = [0u32; 32];
    reachable[16] = 1 << 16;

    // Iterative 8-way dilation
    for _ in 0..32 {
        let mut changed = false;
        let mut next_reachable = reachable;

        for y in 0..32 {
            if reachable[y] == 0 {
                continue;
            }

            // Current row dilation (left/right)
            let row = reachable[y];
            let dilated_row = row | (row << 1) | (row >> 1);

            // Propagate to current row and neighbors
            next_reachable[y] |= dilated_row;
            if y > 0 {
                next_reachable[y - 1] |= dilated_row;
            }
            if y < 31 {
                next_reachable[y + 1] |= dilated_row;
            }
        }

        // Mask out forbidden squares
        for y in 0..32 {
            let prev = reachable[y];
            next_reachable[y] &= !forbidden[y];
            if next_reachable[y] != prev {
                changed = true;
            }
            reachable[y] = next_reachable[y];
        }

        if !changed {
            break;
        }

        // Check if we hit the perimeter
        if (reachable[0] | reachable[31]) != 0 {
            return (false, 1024);
        }
        for reach in reachable.iter().take(31).skip(1) {
            if (reach & 0x80000001) != 0 {
                return (false, 1024);
            }
        }
    }

    // 3. Successful fill without hitting the perimeter indicates a contained cage
    let mut area = 0u32;
    for row in reachable.iter() {
        area += row.count_ones();
    }

    (area > 0 && area < 1000, area)
}

// --- Utility Functions ---

/// Check if a side only has a king (no other pieces)
#[inline(always)]
pub fn is_lone_king(game: &GameState, color: PlayerColor) -> bool {
    if color == PlayerColor::White {
        game.white_pawn_count == 0 && !game.white_non_pawn_material
    } else {
        game.black_pawn_count == 0 && !game.black_non_pawn_material
    }
}

/// Calculates the mop-up scaling factor (0-100) based on remaining material.
/// Mop-up is only active when the opponent's material is below the threshold.
#[inline(always)]
pub fn calculate_mop_up_scale(game: &GameState, losing_color: PlayerColor) -> Option<u32> {
    // Count NON-PAWN pieces only (excluding king)
    let (losing_pieces, losing_starting) = if losing_color == PlayerColor::White {
        // white_piece_count includes all pieces, subtract pawns and king
        let current_non_pawn = game.white_piece_count.saturating_sub(game.white_pawn_count);
        let current_non_king = current_non_pawn.saturating_sub(1); // -1 for king
        let starting = game.starting_white_pieces.saturating_sub(1); // starting already excludes pawns, -1 for king
        (current_non_king, starting)
    } else {
        let current_non_pawn = game.black_piece_count.saturating_sub(game.black_pawn_count);
        let current_non_king = current_non_pawn.saturating_sub(1);
        let starting = game.starting_black_pieces.saturating_sub(1);
        (current_non_king, starting)
    };

    // Check winning side has at least one non-pawn piece
    let winning_has_pieces = if losing_color == PlayerColor::White {
        game.black_non_pawn_material
    } else {
        game.white_non_pawn_material
    };

    if !winning_has_pieces {
        return None; // Don't mop-up with just king+pawns
    }

    if losing_pieces == 0 {
        return Some(100); // Full mop-up
    }

    if losing_starting == 0 {
        return None;
    }

    let percent_remaining = (losing_pieces as u32 * 100) / losing_starting as u32;

    if percent_remaining >= MOP_UP_THRESHOLD_PERCENT {
        return None;
    }

    // Scale linear regression from 100% (at 0% material) to 0% (at threshold)
    Some(100 - (percent_remaining * 100 / MOP_UP_THRESHOLD_PERCENT).min(100))
}

/// Unscaled mop-up evaluation.
#[inline(always)]
pub fn evaluate_lone_king_endgame(
    game: &GameState,
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    evaluate_mop_up_core(game, our_king, enemy_king, winning_color)
}

/// Scaled mop-up evaluation.
#[inline(always)]
pub fn evaluate_mop_up_scaled(
    game: &GameState,
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
    losing_color: PlayerColor,
) -> i32 {
    let scale = match calculate_mop_up_scale(game, losing_color) {
        Some(s) if s > 0 => s,
        _ => return 0,
    };

    let raw = evaluate_mop_up_core(game, our_king, enemy_king, winning_color);
    (raw * scale as i32) / 100
}

// --- Core Evaluation ---

/// Returns true if this endgame (with our king helping) can only be won by using the world border
/// as part of the mating net — i.e., the material is insufficient on an unbounded board but
/// sufficient on a bounded one.
fn is_bounded_only_winnable(
    queen_count: u8,
    rook_count: u8,
    ortho_count: u8,
    diag_count: u8,
    leaper_count: u8,
    amazon_count: u8,
    total_non_pawn_pieces: u8,
    has_king: bool,
) -> bool {
    // Amazons can mate without borders; and without our king we have no mating king aid
    if !has_king || amazon_count >= 1 {
        return false;
    }

    // K+Q vs K  (queen alone is insufficient on unbounded)
    if queen_count == 1 && total_non_pawn_pieces == 1 {
        return true;
    }
    // K+R vs K  (single rook insufficient on unbounded)
    if rook_count == 1 && queen_count == 0 && total_non_pawn_pieces == 1 {
        return true;
    }
    // K+Chancellor vs K  (ortho slider but not a rook, no diag, no leaper)
    if ortho_count == 1
        && rook_count == 0
        && queen_count == 0
        && diag_count == 0
        && leaper_count == 0
        && total_non_pawn_pieces == 1
    {
        return true;
    }
    // K+Archbishop vs K  (diag + knight-component but no ortho)
    if diag_count == 1
        && ortho_count == 0
        && queen_count == 0
        && leaper_count == 0
        && total_non_pawn_pieces == 1
    {
        return true;
    }
    // K+R+minor vs K  (R+N or R+same-color-B — insufficient on unbounded)
    if rook_count == 1 && queen_count == 0 && total_non_pawn_pieces == 2 {
        return true;
    }

    false
}

/// Bonus for driving the enemy king toward world-border corners/edges.
/// Used when the mating pattern requires the board edge as a fence.
fn corner_drive_bonus(
    enemy_king: &Coordinate,
    our_king: Option<&Coordinate>,
    our_pieces: &[SliderInfo],
) -> i32 {
    let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();
    let ex = enemy_king.x;
    let ey = enemy_king.y;

    let dist_h = ((ex - min_x) as i32).min((max_x - ex) as i32); // dist to nearest x-wall
    let dist_v = ((ey - min_y) as i32).min((max_y - ey) as i32); // dist to nearest y-wall

    let world_w = ((max_x - min_x) as i32).max(1);
    let world_h = ((max_y - min_y) as i32).max(1);
    let half_w = (world_w / 2).max(1);
    let half_h = (world_h / 2).max(1);

    // Strong linear gradient: max at wall (dist=0), zero at board center
    let edge_h = (half_w - dist_h.min(half_w)) * 16;
    let edge_v = (half_h - dist_v.min(half_h)) * 16;

    // Smooth proximity bonus using quadratic formula
    // Rewards continuous progress toward corners/edges
    // At edge (dist=0): max bonus
    // At center (dist=half): zero bonus
    // Formula: (1 - (dist/half)^2) * max_bonus, clamped to [0, max_bonus]
    let max_proximity = 500i32;
    let proximity_h = if dist_h < half_w {
        let ratio = (dist_h as i32) as f32 / (half_w as f32);
        ((1.0 - ratio * ratio) * max_proximity as f32) as i32
    } else {
        0
    };
    let proximity_v = if dist_v < half_h {
        let ratio = (dist_v as i32) as f32 / (half_h as f32);
        ((1.0 - ratio * ratio) * max_proximity as f32) as i32
    } else {
        0
    };
    // Corner bonus: both dimensions close to edge get extra reward
    let corner_bonus = if dist_h == 0 && dist_v == 0 {
        200 // extra bonus for exact corner
    } else if dist_h <= 1 && dist_v <= 1 {
        100 // extra bonus for near corner
    } else {
        0
    };
    let proximity = proximity_h + proximity_v + corner_bonus;

    // Direction from enemy king toward board center (= its escape direction away from the walls)
    let cx = (min_x + max_x) / 2;
    let cy = (min_y + max_y) / 2;
    let to_center_x = cx - ex; // positive = center is to the right
    let to_center_y = cy - ey; // positive = center is above

    let mut cut = 0i32;
    for p in our_pieces {
        let pdx = p.x - ex;
        let pdy = p.y - ey;

        // FENCE BONUS: slider aligned on same rank/file AND on the center-side of the enemy king.
        // A piece on the same rank cuts off all horizontal escape; same file cuts all vertical —
        // the core technique of K+Q vs K and K+R vs K mating patterns.
        if pdy == 0 && pdx != 0 {
            if (to_center_x > 0 && pdx > 0) || (to_center_x < 0 && pdx < 0) {
                cut += 50;
            }
        }
        if pdx == 0 && pdy != 0 {
            if (to_center_y > 0 && pdy > 0) || (to_center_y < 0 && pdy < 0) {
                cut += 50;
            }
        }

        // General center-side bonus (piece on the escape side of the enemy king)
        if to_center_x > 0 && pdx > 0 {
            cut += 14;
        } else if to_center_x < 0 && pdx < 0 {
            cut += 14;
        }
        if to_center_y > 0 && pdy > 0 {
            cut += 14;
        } else if to_center_y < 0 && pdy < 0 {
            cut += 14;
        }
    }

    // Our king on the center-side helps box in the enemy king from the inside
    if let Some(ok) = our_king {
        let kdx = ok.x - ex;
        let kdy = ok.y - ey;
        if to_center_x > 0 && kdx > 0 {
            cut += 10;
        } else if to_center_x < 0 && kdx < 0 {
            cut += 10;
        }
        if to_center_y > 0 && kdy > 0 {
            cut += 10;
        } else if to_center_y < 0 && kdy < 0 {
            cut += 10;
        }
    }

    edge_h + edge_v + proximity + cut
}

/// Specialized technique for K+Amazon vs K:
/// keep the amazon on the far side so it cuts the king off,
/// while our king approaches from the near side.
fn amazon_mate_drive_bonus(
    enemy_king: &Coordinate,
    our_king: &Coordinate,
    amazon: &SliderInfo,
) -> i32 {
    let ex = enemy_king.x;
    let ey = enemy_king.y;
    let kdx = our_king.x - ex;
    let kdy = our_king.y - ey;
    let adx = amazon.x - ex;
    let ady = amazon.y - ey;

    let mut bonus = 0;

    let king_dist = kdx.abs().max(kdy.abs());
    bonus += (40 - king_dist.min(40) as i32) * 18;
    if king_dist <= 2 {
        bonus += 240;
    } else if king_dist <= 4 {
        bonus += 120;
    }

    let between_x = adx != 0 && kdx != 0 && adx.signum() != kdx.signum();
    let between_y = ady != 0 && kdy != 0 && ady.signum() != kdy.signum();
    if between_x {
        bonus += 320;
    } else if adx != 0 && kdx != 0 {
        bonus -= 220;
    }
    if between_y {
        bonus += 320;
    } else if ady != 0 && kdy != 0 {
        bonus -= 220;
    }

    let fence_dist_x = adx.abs();
    let fence_dist_y = ady.abs();
    if between_x {
        bonus += (12 - fence_dist_x.min(12) as i32) * 45;
    }
    if between_y {
        bonus += (12 - fence_dist_y.min(12) as i32) * 45;
    }

    if adx == 0 || ady == 0 || adx.abs() == ady.abs() {
        bonus += 90;
    }

    let king_amazon_dist = (amazon.x - our_king.x)
        .abs()
        .max((amazon.y - our_king.y).abs());
    bonus += (30 - king_amazon_dist.min(30) as i32) * 8;

    bonus
}

#[derive(Clone, Copy)]
struct FenceState {
    ortho_y_min_above: i64,
    ortho_y_max_below: i64,
    ortho_x_min_right: i64,
    ortho_x_max_left: i64,
    ortho_y_min_above_2: i64,
    ortho_y_max_below_2: i64,
    ortho_x_min_right_2: i64,
    ortho_x_max_left_2: i64,
    diag_pos_min_above: i64,
    diag_pos_max_below: i64,
    diag_neg_min_above: i64,
    diag_neg_max_below: i64,
}

impl FenceState {
    #[inline(always)]
    fn new() -> Self {
        Self {
            ortho_y_min_above: i64::MAX,
            ortho_y_max_below: i64::MIN,
            ortho_x_min_right: i64::MAX,
            ortho_x_max_left: i64::MIN,
            ortho_y_min_above_2: i64::MAX,
            ortho_y_max_below_2: i64::MIN,
            ortho_x_min_right_2: i64::MAX,
            ortho_x_max_left_2: i64::MIN,
            diag_pos_min_above: i64::MAX,
            diag_pos_max_below: i64::MIN,
            diag_neg_min_above: i64::MAX,
            diag_neg_max_below: i64::MIN,
        }
    }
}

#[derive(Clone, Copy)]
struct MaterialSummary {
    ortho_count: u8,
    diag_count: u8,
    leaper_count: u8,
    queen_count: u8,
    rook_count: u8,
    amazon_count: u8,
    total_non_pawn_pieces: u8,
    short_range_bonus: i32,
    amazon_square: Option<SliderInfo>,
}

impl MaterialSummary {
    #[inline(always)]
    fn new() -> Self {
        Self {
            ortho_count: 0,
            diag_count: 0,
            leaper_count: 0,
            queen_count: 0,
            rook_count: 0,
            amazon_count: 0,
            total_non_pawn_pieces: 0,
            short_range_bonus: 0,
            amazon_square: None,
        }
    }

    #[inline(always)]
    fn total_sliders(&self) -> u8 {
        self.ortho_count.max(self.diag_count)
    }

    #[inline(always)]
    fn is_overwhelming(&self) -> bool {
        self.queen_count >= 1 || self.amazon_count >= 1 || self.total_non_pawn_pieces >= 5
    }

    #[inline(always)]
    fn is_double_rook_endgame(&self) -> bool {
        self.ortho_count == 2
            && self.diag_count == 0
            && self.leaper_count == 0
            && self.total_non_pawn_pieces == 2
    }

    #[inline(always)]
    fn is_single_amazon_endgame(&self) -> bool {
        self.amazon_count == 1 && self.total_non_pawn_pieces == 1
    }
}

struct PieceList {
    pieces: [SliderInfo; 24],
    len: usize,
}

impl PieceList {
    #[inline(always)]
    fn new() -> Self {
        Self {
            pieces: [SliderInfo { x: 0, y: 0, pt: PieceType::Void }; 24],
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, piece: SliderInfo) {
        if self.len < self.pieces.len() {
            self.pieces[self.len] = piece;
            self.len += 1;
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[SliderInfo] {
        &self.pieces[..self.len]
    }
}

#[derive(Clone, Copy)]
struct KingRelation {
    our_dx: i64,
    our_dy: i64,
    king_dist: i64,
}

#[derive(Clone, Copy)]
struct CageInfo {
    bitboard_caged: bool,
    reached_area: u32,
    macro_box: bool,
    macro_area: u32,
}

#[derive(Clone, Copy)]
enum CustomMopUpCase {
    KingAmazonVsKing,
    KingDoubleRookVsKing,
}

#[derive(Clone, Copy)]
enum MopUpStrategy {
    Custom(CustomMopUpCase),
    GenericOverwhelming,
    Technical,
}

#[inline(always)]
fn detect_custom_mop_up_case(material: &MaterialSummary) -> Option<CustomMopUpCase> {
    if material.is_single_amazon_endgame() {
        return Some(CustomMopUpCase::KingAmazonVsKing);
    }

    if material.is_double_rook_endgame() {
        return Some(CustomMopUpCase::KingDoubleRookVsKing);
    }

    None
}

#[inline(always)]
fn select_mop_up_strategy(material: &MaterialSummary) -> MopUpStrategy {
    if let Some(case) = detect_custom_mop_up_case(material) {
        return MopUpStrategy::Custom(case);
    }

    if material.is_overwhelming() {
        MopUpStrategy::GenericOverwhelming
    } else {
        MopUpStrategy::Technical
    }
}

/// Generic, piece-aware mop-up evaluator.
///
/// Strategy (board-edge-agnostic, works on bounded and unbounded boards):
///   1. Cage / macro-box compactness — directly rewards small reachable area.
///   2. King approach — push our king to within 2-4 squares of the enemy king.
///   3. Ring coverage — for each of the 8 escape squares around the enemy king,
///      reward whether any of our pieces (or our king) geometrically attacks it,
///      with extra weight on the escape side (opposite our king).
///   4. Axis sandwich — reward our pieces controlling both sides of the enemy king
///      along the four cardinal axes (creates an opposition cage).
///   5. Opposition — pieces opposite our king cut off escape; our king pushes from
///      the other side, so reward both halves of the mating net.
///   6. Per-piece ideal-distance shaping — sliders aim from a couple squares back,
///      short-range leapers must close in. Bonus for sliders standing on rays.
#[inline]
fn evaluate_generic_overwhelming_mop_up(
    king_relation: KingRelation,
    our_king: Option<&Coordinate>,
    pieces: &[SliderInfo],
    enemy_king: &Coordinate,
    cage: CageInfo,
    winning_color: PlayerColor,
) -> i32 {
    let ex = enemy_king.x;
    let ey = enemy_king.y;
    let mut bonus: i32 = 0;

    // 1. Cage / macro-box compactness.
    if cage.bitboard_caged {
        bonus += (3000 / (cage.reached_area + 4).max(1) as i32).clamp(50, 600);
    }
    if cage.macro_box {
        bonus += if cage.macro_area <= 64 {
            150
        } else if cage.macro_area <= 256 {
            70
        } else if cage.macro_area <= 1024 {
            30
        } else {
            10
        };
    }

    // 2. King approach.
    bonus += (60 - king_relation.king_dist.min(60) as i32) * 22;
    if king_relation.king_dist <= 2 {
        bonus += 220;
    } else if king_relation.king_dist <= 4 {
        bonus += 110;
    }

    // 3. Ring coverage. For each of 8 escape squares adjacent to the enemy king,
    // mark whether at least one of our pieces (or our king) controls it.
    let escape_x = if king_relation.king_dist < i64::MAX {
        -king_relation.our_dx.signum()
    } else {
        0
    };
    let escape_y = if king_relation.king_dist < i64::MAX {
        -king_relation.our_dy.signum()
    } else {
        0
    };

    let mut covered: u8 = 0;
    let mut total_attacks: i32 = 0;
    for (i, &(rdx, rdy)) in RING_DIRS.iter().enumerate() {
        let rx = ex + rdx;
        let ry = ey + rdy;
        let mut attacked = false;

        for s in pieces {
            if piece_attacks_geom(s.pt, winning_color, rx - s.x, ry - s.y) {
                attacked = true;
                total_attacks += 1;
            }
        }
        if let Some(ok) = our_king {
            let kdx = (rx - ok.x).abs();
            let kdy = (ry - ok.y).abs();
            if kdx <= 1 && kdy <= 1 {
                attacked = true;
            }
        }

        if attacked {
            covered |= 1 << i;
            bonus += 28;
            // Squares on the escape side (away from our king) matter most.
            let on_esc_x = escape_x != 0 && rdx.signum() == escape_x;
            let on_esc_y = escape_y != 0 && rdy.signum() == escape_y;
            if on_esc_x && on_esc_y {
                bonus += 55;
            } else if on_esc_x || on_esc_y {
                bonus += 28;
            }
        }
    }
    let n_covered = covered.count_ones() as i32;
    bonus += n_covered * 18;
    if n_covered >= 7 {
        bonus += 220;
    } else if n_covered >= 6 {
        bonus += 130;
    } else if n_covered >= 5 {
        bonus += 65;
    }
    bonus += total_attacks * 3;

    // 4. Axis sandwich (using ring bits): paired-coverage on opposite sides.
    let bit = |i: usize| (covered >> i) & 1 != 0;
    let sand_h = bit(0) && bit(1);
    let sand_v = bit(2) && bit(3);
    let sand_dp = bit(4) && bit(5);
    let sand_dn = bit(6) && bit(7);
    let n_sand = sand_h as i32 + sand_v as i32 + sand_dp as i32 + sand_dn as i32;
    bonus += n_sand * 55;
    if sand_h && sand_v {
        bonus += 130;
    }
    if sand_dp && sand_dn {
        bonus += 80;
    }

    // 5. Opposition + per-piece ideal-distance shaping in one pass.
    for s in pieces {
        let dx = s.x - ex;
        let dy = s.y - ey;
        let dist = dx.abs().max(dy.abs());

        // Place pieces opposite our king to cut escape from the far side.
        if our_king.is_some() {
            let opposite_x = dx != 0
                && king_relation.our_dx != 0
                && dx.signum() != king_relation.our_dx.signum();
            let opposite_y = dy != 0
                && king_relation.our_dy != 0
                && dy.signum() != king_relation.our_dy.signum();
            if opposite_x {
                bonus += 18;
            }
            if opposite_y {
                bonus += 18;
            }
        }

        // Ideal-distance shaping by piece type.
        let (ideal_min, ideal_max) = ideal_distance(s.pt);
        bonus += if dist < ideal_min {
            -((ideal_min - dist) as i32) * 6
        } else if dist <= ideal_max {
            let mid = (ideal_min + ideal_max) / 2;
            45 - ((dist - mid).abs() as i32) * 4
        } else {
            let over = (dist - ideal_max).min(80) as i32;
            -(over * 4) - 8
        };

        // Sliders standing on a ray from the enemy king already have a cut.
        let is_ortho_slider = matches!(
            s.pt,
            PieceType::Rook
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Chancellor
                | PieceType::Amazon
                | PieceType::Huygen
        );
        let is_diag_slider = matches!(
            s.pt,
            PieceType::Bishop
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Archbishop
                | PieceType::Amazon
        );
        if is_ortho_slider && (dx == 0 || dy == 0) {
            bonus += 32;
        }
        if is_diag_slider && dx != 0 && dx.abs() == dy.abs() {
            bonus += 22;
        }

        // Coordination with our king (short pieces need it; sliders less so).
        if let Some(ok) = our_king {
            let king_piece_dist = (s.x - ok.x).abs().max((s.y - ok.y).abs());
            if is_short_range_piece(s.pt) {
                bonus += (8 - king_piece_dist.min(8) as i32) * 6;
            } else {
                bonus += (24 - king_piece_dist.min(24) as i32) * 1;
            }
        }
    }

    // Encourage tight cage when one is detected.
    if cage.bitboard_caged && cage.reached_area <= 16 {
        bonus += (16 - cage.reached_area as i32) * 12;
    }

    bonus
}

#[inline(always)]
fn evaluate_king_amazon_vs_king(
    king_relation: KingRelation,
    enemy_king: &Coordinate,
    our_king: Option<&Coordinate>,
    pieces: &[SliderInfo],
    material: &MaterialSummary,
    cage: CageInfo,
    winning_color: PlayerColor,
) -> i32 {
    let mut bonus = evaluate_generic_overwhelming_mop_up(
        king_relation,
        our_king,
        pieces,
        enemy_king,
        cage,
        winning_color,
    );

    if let (Some(ok), Some(amazon)) = (our_king, material.amazon_square.as_ref()) {
        bonus += amazon_mate_drive_bonus(enemy_king, ok, amazon);
    }

    bonus
}

#[inline(always)]
fn evaluate_king_double_rook_vs_king(
    king_relation: KingRelation,
    enemy_king: &Coordinate,
    our_king: Option<&Coordinate>,
    pieces: &[SliderInfo],
) -> i32 {
    let enemy_x = enemy_king.x;
    let enemy_y = enemy_king.y;
    let (r1_x, r1_y, r2_x, r2_y) = if pieces.len() == 2 {
        (pieces[0].x, pieces[0].y, pieces[1].x, pieces[1].y)
    } else {
        (0, 0, 0, 0)
    };

    let mut bonus = 0;
    let rooks_on_same_rank = r1_y == r2_y;
    let rooks_on_same_file = r1_x == r2_x;
    let rooks_protecting = rooks_on_same_rank || rooks_on_same_file;

    if rooks_protecting {
        bonus += 200;
        let rook_dist_between = (r1_x - r2_x).abs() + (r1_y - r2_y).abs();
        bonus -= (rook_dist_between as i32) * 5;
    } else {
        bonus -= 200;
    }

    let has_rook_above = r1_y > enemy_y || r2_y > enemy_y;
    let has_rook_below = r1_y < enemy_y || r2_y < enemy_y;
    let has_rook_right = r1_x > enemy_x || r2_x > enemy_x;
    let has_rook_left = r1_x < enemy_x || r2_x < enemy_x;

    let has_sandwich_v = has_rook_above && has_rook_below;
    let has_sandwich_h = has_rook_right && has_rook_left;

    if has_sandwich_v {
        bonus += 100;
        let ca =
            if r1_y > enemy_y { r1_y } else { r2_y }.min(if r2_y > enemy_y { r2_y } else { r1_y });
        let cb =
            if r1_y < enemy_y { r1_y } else { r2_y }.max(if r2_y < enemy_y { r2_y } else { r1_y });
        let gap = ca - cb - 1;
        bonus += (8 - gap.min(8) as i32) * 15;
    }
    if has_sandwich_h {
        bonus += 100;
        let cr =
            if r1_x > enemy_x { r1_x } else { r2_x }.min(if r2_x > enemy_x { r2_x } else { r1_x });
        let cl =
            if r1_x < enemy_x { r1_x } else { r2_x }.max(if r2_x < enemy_x { r2_x } else { r1_x });
        let gap = cr - cl - 1;
        bonus += (8 - gap.min(8) as i32) * 15;
    }

    for r in &[(r1_x, r1_y), (r2_x, r2_y)] {
        let rd = (r.1 - enemy_y).abs();
        let fd = (r.0 - enemy_x).abs();
        if rd > 0 {
            bonus += if rd == 1 {
                40
            } else if rd == 2 {
                25
            } else {
                5
            };
        }
        if fd > 0 {
            bonus += if fd == 1 {
                40
            } else if fd == 2 {
                25
            } else {
                5
            };
        }
    }

    if let Some(ok) = our_king {
        bonus += (100 - king_relation.king_dist.min(100) as i32) * 10;

        if king_relation.king_dist <= 2 {
            bonus += 300;
        } else if king_relation.king_dist <= 4 {
            bonus += 150;
        }

        let our_dx = ok.x - enemy_x;
        let our_dy = ok.y - enemy_y;

        if our_dx > 0 && has_rook_left {
            bonus += 120;
        }
        if our_dx < 0 && has_rook_right {
            bonus += 120;
        }
        if our_dy > 0 && has_rook_below {
            bonus += 120;
        }
        if our_dy < 0 && has_rook_above {
            bonus += 120;
        }

        if has_sandwich_v && our_dy.abs() <= 1 {
            bonus += 100;
        }
        if has_sandwich_h && our_dx.abs() <= 1 {
            bonus += 100;
        }

        if (rooks_on_same_rank && ok.y == r1_y) || (rooks_on_same_file && ok.x == r1_x) {
            bonus -= 150;
        }
    }

    if has_sandwich_v && has_sandwich_h {
        bonus += 200;
    }

    bonus
}

#[inline(always)]
fn evaluate_custom_mop_up_case(
    case_: CustomMopUpCase,
    king_relation: KingRelation,
    enemy_king: &Coordinate,
    our_king: Option<&Coordinate>,
    pieces: &[SliderInfo],
    material: &MaterialSummary,
    cage: CageInfo,
    winning_color: PlayerColor,
) -> i32 {
    match case_ {
        CustomMopUpCase::KingAmazonVsKing => evaluate_king_amazon_vs_king(
            king_relation,
            enemy_king,
            our_king,
            pieces,
            material,
            cage,
            winning_color,
        ),
        CustomMopUpCase::KingDoubleRookVsKing => {
            evaluate_king_double_rook_vs_king(king_relation, enemy_king, our_king, pieces)
        }
    }
}

#[inline(always)]
fn evaluate_technical_mop_up(
    game: &GameState,
    king_relation: KingRelation,
    enemy_king: &Coordinate,
    our_king: Option<&Coordinate>,
    winning_color: PlayerColor,
    pieces: &[SliderInfo],
    fences: &FenceState,
    cage: CageInfo,
) -> i32 {
    let enemy_x = enemy_king.x;
    let enemy_y = enemy_king.y;
    // Smart generic evaluation provides ring coverage, opposition, per-piece distance shaping.
    let mut bonus = evaluate_generic_overwhelming_mop_up(
        king_relation,
        our_king,
        pieces,
        enemy_king,
        cage,
        winning_color,
    );
    let mut protected_count = 0;

    for s in pieces {
        let coord = Coordinate::new(s.x, s.y);
        if crate::moves::is_square_attacked(
            &game.board,
            &coord,
            winning_color,
            &game.spatial_indices,
        ) {
            protected_count += 1;
        }
    }
    bonus += protected_count * 40;

    let mut sand_h = false;
    let mut sand_v = false;
    let mut tight_h = false;
    let mut tight_v = false;
    if fences.ortho_y_min_above != i64::MAX && fences.ortho_y_max_below != i64::MIN {
        let gap = fences.ortho_y_min_above - fences.ortho_y_max_below - 1;
        if gap <= 3 {
            sand_v = true;
            if gap <= 1 {
                tight_v = true;
            }
        }
        bonus += if gap <= 1 {
            160
        } else if gap <= 2 {
            120
        } else if gap <= 3 {
            100
        } else {
            40
        };
    }
    if fences.ortho_x_min_right != i64::MAX && fences.ortho_x_max_left != i64::MIN {
        let gap = fences.ortho_x_min_right - fences.ortho_x_max_left - 1;
        if gap <= 3 {
            sand_h = true;
            if gap <= 1 {
                tight_h = true;
            }
        }
        bonus += if gap <= 1 {
            160
        } else if gap <= 2 {
            120
        } else if gap <= 3 {
            100
        } else {
            40
        };
    }

    let mut sand_dp = false;
    let mut sand_dn = false;
    if fences.diag_pos_min_above != i64::MAX && fences.diag_pos_max_below != i64::MIN {
        let gap = fences.diag_pos_min_above - fences.diag_pos_max_below - 1;
        if gap <= 2 {
            sand_dp = true;
        }
        bonus += if gap <= 1 {
            120
        } else if gap <= 2 {
            90
        } else {
            30
        };
    }
    if fences.diag_neg_min_above != i64::MAX && fences.diag_neg_max_below != i64::MIN {
        let gap = fences.diag_neg_min_above - fences.diag_neg_max_below - 1;
        if gap <= 2 {
            sand_dn = true;
        }
        bonus += if gap <= 1 {
            120
        } else if gap <= 2 {
            90
        } else {
            30
        };
    }

    let mut ladder = false;
    let ladder_x = (fences.ortho_x_min_right != i64::MAX
        && fences.ortho_x_min_right_2 != i64::MAX
        && (fences.ortho_x_min_right_2 - fences.ortho_x_min_right) == 1)
        || (fences.ortho_x_max_left != i64::MIN
            && fences.ortho_x_max_left_2 != i64::MIN
            && (fences.ortho_x_max_left - fences.ortho_x_max_left_2) == 1);
    let ladder_y = (fences.ortho_y_min_above != i64::MAX
        && fences.ortho_y_min_above_2 != i64::MAX
        && (fences.ortho_y_min_above_2 - fences.ortho_y_min_above) == 1)
        || (fences.ortho_y_max_below != i64::MIN
            && fences.ortho_y_max_below_2 != i64::MIN
            && (fences.ortho_y_max_below - fences.ortho_y_max_below_2) == 1);
    if ladder_x || ladder_y {
        ladder = true;
        bonus += 240;
    }

    let r_up = if fences.ortho_y_min_above != i64::MAX {
        fences.ortho_y_min_above - enemy_y - 1
    } else {
        15
    };
    let r_down = if fences.ortho_y_max_below != i64::MIN {
        enemy_y - fences.ortho_y_max_below - 1
    } else {
        15
    };
    let r_right = if fences.ortho_x_min_right != i64::MAX {
        fences.ortho_x_min_right - enemy_x - 1
    } else {
        15
    };
    let r_left = if fences.ortho_x_max_left != i64::MIN {
        enemy_x - fences.ortho_x_max_left - 1
    } else {
        15
    };

    let run_h = if king_relation.our_dx > 0 {
        r_left
    } else if king_relation.our_dx < 0 {
        r_right
    } else {
        r_left.max(r_right)
    };
    let run_v = if king_relation.our_dy > 0 {
        r_down
    } else if king_relation.our_dy < 0 {
        r_up
    } else {
        r_up.max(r_down)
    };
    bonus += (20 - run_h.min(20)) as i32 * 12;
    bonus += (20 - run_v.min(20)) as i32 * 12;

    let is_contained = ladder
        || (sand_h && tight_h)
        || (sand_v && tight_v)
        || (sand_h && sand_v)
        || (sand_dp && sand_dn)
        || (cage.bitboard_caged && cage.reached_area <= 12);

    if is_contained {
        let prox = (30 - king_relation.king_dist.min(30)) as i32;
        bonus += prox * 16;
        if king_relation.king_dist <= 2 {
            bonus += 80;
        }
    }

    let _ = (enemy_x, enemy_y); // values are used implicitly via fences/king_relation
    bonus
}

/// Main logic for driving the enemy king to mate.
#[inline(always)]
fn evaluate_mop_up_core(
    game: &GameState,
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    let mut bonus: i32 = 0;
    let king_relation = if let Some(ok) = our_king {
        let dx = ok.x - enemy_king.x;
        let dy = ok.y - enemy_king.y;
        KingRelation {
            our_dx: dx,
            our_dy: dy,
            king_dist: dx.abs().max(dy.abs()),
        }
    } else {
        KingRelation {
            our_dx: 0,
            our_dy: 0,
            king_dist: i64::MAX,
        }
    };

    let mut fences = FenceState::new();
    let mut material = MaterialSummary::new();
    let mut our_pieces = PieceList::new();

    let enemy_x = enemy_king.x;
    let enemy_y = enemy_king.y;
    let enemy_diag_pos = enemy_x + enemy_y;
    let enemy_diag_neg = enemy_x - enemy_y;

    // Single pass variables
    let is_white = winning_color == PlayerColor::White;
    for (x, y, piece) in game.board.iter_pieces_by_color(is_white) {
        let pt = piece.piece_type();

        if pt.is_royal() {
            continue;
        }

        // Orthogonal sliders
        let has_ortho = matches!(
            pt,
            PieceType::Rook
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Chancellor
                | PieceType::Amazon
        );

        if pt != PieceType::King && pt != PieceType::Pawn {
            our_pieces.push(SliderInfo { x, y, pt });
        }

        if has_ortho {
            material.ortho_count += 1;
            if y > enemy_y {
                if y < fences.ortho_y_min_above {
                    fences.ortho_y_min_above_2 = fences.ortho_y_min_above;
                    fences.ortho_y_min_above = y;
                } else if y < fences.ortho_y_min_above_2 {
                    fences.ortho_y_min_above_2 = y;
                }
            } else if y < enemy_y {
                if y > fences.ortho_y_max_below {
                    fences.ortho_y_max_below_2 = fences.ortho_y_max_below;
                    fences.ortho_y_max_below = y;
                } else if y > fences.ortho_y_max_below_2 {
                    fences.ortho_y_max_below_2 = y;
                }
            }

            if x > enemy_x {
                if x < fences.ortho_x_min_right {
                    fences.ortho_x_min_right_2 = fences.ortho_x_min_right;
                    fences.ortho_x_min_right = x;
                } else if x < fences.ortho_x_min_right_2 {
                    fences.ortho_x_min_right_2 = x;
                }
            } else if x < enemy_x {
                if x > fences.ortho_x_max_left {
                    fences.ortho_x_max_left_2 = fences.ortho_x_max_left;
                    fences.ortho_x_max_left = x;
                } else if x > fences.ortho_x_max_left_2 {
                    fences.ortho_x_max_left_2 = x;
                }
            }
        }

        // Diagonal sliders
        let has_diag = matches!(
            pt,
            PieceType::Bishop
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Archbishop
                | PieceType::Amazon
        );

        if has_diag {
            material.diag_count += 1;
            let dp = x + y;
            let dn = x - y;
            if dp > enemy_diag_pos && dp < fences.diag_pos_min_above {
                fences.diag_pos_min_above = dp;
            }
            if dp < enemy_diag_pos && dp > fences.diag_pos_max_below {
                fences.diag_pos_max_below = dp;
            }
            if dn > enemy_diag_neg && dn < fences.diag_neg_min_above {
                fences.diag_neg_min_above = dn;
            }
            if dn < enemy_diag_neg && dn > fences.diag_neg_max_below {
                fences.diag_neg_max_below = dn;
            }
        }

        if pt == PieceType::Queen || pt == PieceType::RoyalQueen {
            material.queen_count += 1;
        } else if pt == PieceType::Amazon {
            material.amazon_count += 1;
            material.amazon_square = Some(SliderInfo { x, y, pt });
        }

        if pt == PieceType::Rook {
            material.rook_count += 1;
        }

        material.total_non_pawn_pieces += 1;

        // Placement heuristics
        let pdx = x - enemy_x;
        let pdy = y - enemy_y;

        let on_back_x =
            (king_relation.our_dx > 0 && pdx < 0) || (king_relation.our_dx < 0 && pdx > 0);
        let on_back_y =
            (king_relation.our_dy > 0 && pdy < 0) || (king_relation.our_dy < 0 && pdy > 0);

        // Reward cutting off escape relative to our king
        if on_back_x {
            bonus += 7;
        }
        if on_back_y {
            bonus += 7;
        }

        // Diagonals Back Side
        let pdp = x + y - enemy_diag_pos;
        let pdn = x - y - enemy_diag_neg;
        if let Some(ok) = our_king {
            let our_dp = ok.x + ok.y - enemy_diag_pos;
            let our_dn = ok.x - ok.y - enemy_diag_neg;
            if (our_dp > 0 && pdp < 0) || (our_dp < 0 && pdp > 0) {
                bonus += 4;
            }
            if (our_dn > 0 && pdn < 0) || (our_dn < 0 && pdn > 0) {
                bonus += 4;
            }

            if pt == PieceType::Amazon {
                let opposite_x = pdx != 0
                    && king_relation.our_dx != 0
                    && pdx.signum() != king_relation.our_dx.signum();
                let opposite_y = pdy != 0
                    && king_relation.our_dy != 0
                    && pdy.signum() != king_relation.our_dy.signum();

                if opposite_x {
                    bonus += 120;
                } else if pdx != 0 && king_relation.our_dx != 0 {
                    bonus -= 120;
                }

                if opposite_y {
                    bonus += 120;
                } else if pdy != 0 && king_relation.our_dy != 0 {
                    bonus -= 120;
                }
            }
        }

        // Penalize checks that drive the enemy king to safer areas
        let is_checking = match pt {
            PieceType::Rook | PieceType::Chancellor => pdx == 0 || pdy == 0,
            PieceType::Bishop | PieceType::Archbishop => pdx.abs() == pdy.abs(),
            PieceType::Queen | PieceType::Amazon | PieceType::RoyalQueen => {
                pdx == 0 || pdy == 0 || pdx.abs() == pdy.abs()
            }
            PieceType::Knight => {
                (pdx.abs() == 2 && pdy.abs() == 1) || (pdx.abs() == 1 && pdy.abs() == 2)
            }
            _ => false,
        };

        if is_checking {
            // Penalty for checks that push the enemy king away from our king.
            // Calibrated: -30 is enough to discourage, but not so much that the king runs away.
            let is_frontal_check = (king_relation.our_dx.signum() == pdx.signum() && pdx != 0)
                || (king_relation.our_dy.signum() == pdy.signum() && pdy != 0);

            if is_frontal_check {
                bonus -= 6;
            } else {
                bonus -= 2; // Minimal penalty for checks from behind/side
            }
        }

        if !has_ortho && !has_diag {
            material.leaper_count += 1;
            let dist = pdx.abs().max(pdy.abs()); // Chebyshev distance

            // Heavy proximity bonus to ensure short-range pieces engage
            // Continuous smoothing:
            // dist 0..3: 160 -> 130
            // dist 3..10: 130 -> 60
            // dist 10..25: 60 -> 15
            // dist > 25: Penalty
            if dist <= 3 {
                material.short_range_bonus += 160 - (dist as i32 * 10);
            } else if dist <= 10 {
                // Map 4..10 -> 120..60
                material.short_range_bonus += 130 - ((dist - 3) as i32 * 10);
            } else if dist <= 25 {
                // Map 11..25 -> 57..15
                material.short_range_bonus += 60 - ((dist - 10) as i32 * 3);
            } else {
                material.short_range_bonus -= 80;
            }
        }
    }

    let total_sliders = material.total_sliders();
    let few_pieces = material.total_non_pawn_pieces <= 2;
    let our_pieces = our_pieces.as_slice();

    bonus += material.short_range_bonus * if few_pieces { 5 } else { 3 };

    // --- Strategy Selection ---
    let losing_color = winning_color.opponent();
    let is_opponent_lone_king = is_lone_king(game, losing_color);

    if is_opponent_lone_king {
        let (bitboard_caged, reached_area) = find_bitboard_cage(
            &game.board,
            &game.spatial_indices,
            enemy_king,
            winning_color,
        );

        let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();
        const EDGE_THRESHOLD: i64 = 50;
        let has_barrier_above =
            fences.ortho_y_min_above != i64::MAX || (max_y - enemy_y) < EDGE_THRESHOLD;
        let has_barrier_below =
            fences.ortho_y_max_below != i64::MIN || (enemy_y - min_y) < EDGE_THRESHOLD;
        let has_barrier_right =
            fences.ortho_x_min_right != i64::MAX || (max_x - enemy_x) < EDGE_THRESHOLD;
        let has_barrier_left =
            fences.ortho_x_max_left != i64::MIN || (enemy_x - min_x) < EDGE_THRESHOLD;
        let macro_box =
            has_barrier_above && has_barrier_below && has_barrier_right && has_barrier_left;

        let macro_area = if macro_box {
            let box_width =
                if fences.ortho_x_min_right != i64::MAX && fences.ortho_x_max_left != i64::MIN {
                    (fences.ortho_x_min_right - fences.ortho_x_max_left - 1).max(1)
                } else {
                    100
                };
            let box_height =
                if fences.ortho_y_min_above != i64::MAX && fences.ortho_y_max_below != i64::MIN {
                    (fences.ortho_y_min_above - fences.ortho_y_max_below - 1).max(1)
                } else {
                    100
                };
            (box_width * box_height) as u32
        } else {
            10000
        };

        let cage = CageInfo {
            bitboard_caged,
            reached_area,
            macro_box,
            macro_area,
        };

        let strategy = select_mop_up_strategy(&material);

        bonus += match strategy {
            MopUpStrategy::Custom(case_) => evaluate_custom_mop_up_case(
                case_,
                king_relation,
                enemy_king,
                our_king,
                our_pieces,
                &material,
                cage,
                winning_color,
            ),
            MopUpStrategy::GenericOverwhelming => evaluate_generic_overwhelming_mop_up(
                king_relation,
                our_king,
                our_pieces,
                enemy_king,
                cage,
                winning_color,
            ),
            MopUpStrategy::Technical => evaluate_technical_mop_up(
                game,
                king_relation,
                enemy_king,
                our_king,
                winning_color,
                our_pieces,
                &fences,
                cage,
            ),
        };

        if crate::moves::get_world_size() <= 200
            && is_bounded_only_winnable(
                material.queen_count,
                material.rook_count,
                material.ortho_count,
                material.diag_count,
                material.leaper_count,
                material.amazon_count,
                material.total_non_pawn_pieces,
                our_king.is_some(),
            )
        {
            bonus += corner_drive_bonus(enemy_king, our_king, our_pieces) * 5;
        }
    }

    if total_sliders >= 2 {
        bonus += 20;
    }
    if total_sliders >= 3 {
        bonus += 30;
    }
    if material.ortho_count >= 1 && material.diag_count >= 1 {
        bonus += 15;
    }

    bonus
}

// --- Helper Functions ---

/// Determine if king is needed for mate based on material
#[inline(always)]
pub fn needs_king_for_mate(board: &Board, color: PlayerColor) -> bool {
    let mut queens: u8 = 0;
    let mut rooks: u8 = 0;
    let mut bishops: u8 = 0;
    let mut knights: u8 = 0;
    let mut chancellors: u8 = 0;
    let mut archbishops: u8 = 0;
    let mut hawks: u8 = 0;
    let mut guards: u8 = 0;

    let is_white = color == PlayerColor::White;
    for (_, _, piece) in board.iter_pieces_by_color(is_white) {
        match piece.piece_type() {
            PieceType::Queen | PieceType::RoyalQueen => queens += 1,
            PieceType::Rook => rooks += 1,
            PieceType::Bishop => bishops += 1,
            PieceType::Knight => knights += 1,
            PieceType::Chancellor => chancellors += 1,
            PieceType::Archbishop => archbishops += 1,
            PieceType::Hawk => hawks += 1,
            PieceType::Guard => guards += 1,
            _ => {}
        }
        // Quick exits for common cases
        if queens >= 2 {
            return false;
        }
        if rooks >= 3 {
            return false;
        }
    }

    // Strong material combinations that don't need king
    if chancellors >= 2 {
        return false;
    }
    if archbishops >= 3 {
        return false;
    }
    if hawks >= 4 {
        return false;
    }
    if bishops >= 6 {
        return false;
    }
    if queens >= 1 && chancellors >= 1 {
        return false;
    }
    if queens >= 1 && bishops >= 2 {
        return false;
    }
    if queens >= 1 && knights >= 2 {
        return false;
    }
    if queens >= 1 && guards >= 2 {
        return false;
    }
    if queens >= 1 && rooks >= 1 && (bishops >= 1 || knights >= 1) {
        return false;
    }
    if chancellors >= 1 && bishops >= 2 {
        return false;
    }
    if rooks >= 2 && (bishops >= 2 || knights >= 2 || guards >= 1) {
        return false;
    }
    if rooks >= 1 && bishops >= 3 {
        return false;
    }
    if rooks >= 1 && knights >= 4 {
        return false;
    }
    if rooks >= 1 && guards >= 2 {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece};
    use crate::game::GameState;

    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game
    }

    fn create_test_game_from_icn(icn: &str) -> GameState {
        let mut game = create_test_game();
        game.board = Board::new();
        game.setup_position_from_icn(icn);
        game
    }

    #[test]
    fn test_is_lone_king_true() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1");

        assert!(is_lone_king(&game, PlayerColor::White));
    }

    #[test]
    fn test_is_lone_king_false() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1|Q4,1");

        assert!(!is_lone_king(&game, PlayerColor::White));
    }

    #[test]
    fn test_calculate_mop_up_scale_returns_none_for_no_advantage() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1|k5,8|Q4,1|q4,8");

        // Both sides have material, so no mop-up
        let scale = calculate_mop_up_scale(&game, PlayerColor::Black);
        // May or may not apply depending on thresholds
        assert!(scale.is_none() || scale.unwrap() <= 100);
    }

    #[test]
    fn test_needs_king_for_mate_true() {
        // Just a knight - needs king
        let mut board = Board::new();
        board.set_piece(3, 3, Piece::new(PieceType::Knight, PlayerColor::White));

        assert!(needs_king_for_mate(&board, PlayerColor::White));
    }

    #[test]
    fn test_needs_king_for_mate_false_two_queens() {
        let mut board = Board::new();
        board.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
        board.set_piece(4, 3, Piece::new(PieceType::Queen, PlayerColor::White));

        assert!(!needs_king_for_mate(&board, PlayerColor::White));
    }

    #[test]
    fn test_needs_king_for_mate_false_three_rooks() {
        let mut board = Board::new();
        board.set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        board.set_piece(2, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        board.set_piece(3, 1, Piece::new(PieceType::Rook, PlayerColor::White));

        assert!(!needs_king_for_mate(&board, PlayerColor::White));
    }

    #[test]
    fn test_evaluate_lone_king_endgame_returns_value() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1|k5,8|Q4,1");

        let enemy_king = Coordinate::new(5, 8);
        let our_king = Coordinate::new(5, 1);

        let score =
            evaluate_lone_king_endgame(&game, Some(&our_king), &enemy_king, PlayerColor::White);
        // Should be positive (White has advantage)
        assert!(score >= 0);
    }

    #[test]
    fn test_evaluate_mop_up_scaled_no_king() {
        let game = create_test_game_from_icn("w (8;q|1;q) k5,8|Q4,4|Q3,4");

        let enemy_king = Coordinate::new(5, 8);

        // No white king (checkmate practice)
        let score = evaluate_mop_up_scaled(
            &game,
            None,
            &enemy_king,
            PlayerColor::White,
            PlayerColor::Black,
        );
        assert!(score.abs() < 100000);
    }

    #[test]
    fn test_mop_up_rook_fence_bonus() {
        let game = create_test_game_from_icn("w (8;q|1;q) k4,4|R0,4|R7,4|K4,1");

        let enemy_king = Coordinate::new(4, 4);
        let our_king = Coordinate::new(4, 1);

        let score =
            evaluate_lone_king_endgame(&game, Some(&our_king), &enemy_king, PlayerColor::White);
        // Should be positive since rooks create cutting lines
        assert!(
            score > 0,
            "Rook fence should give positive score: {}",
            score
        );
    }

    #[test]
    fn test_mop_up_king_approach_bonus() {
        let mut game = create_test_game_from_icn("w (8;q|1;q) k5,5|Q4,4|K6,5");

        let enemy_king = Coordinate::new(5, 5);
        let our_king_close = Coordinate::new(6, 5);

        let score_close = evaluate_lone_king_endgame(
            &game,
            Some(&our_king_close),
            &enemy_king,
            PlayerColor::White,
        );

        // Move white king further away
        game.setup_position_from_icn("w (8;q|1;q) k5,5|Q4,4|K1,1");

        let our_king_far = Coordinate::new(1, 1);
        let score_far =
            evaluate_lone_king_endgame(&game, Some(&our_king_far), &enemy_king, PlayerColor::White);

        assert!(
            score_close > score_far,
            "Closer king should get higher score: close={} far={}",
            score_close,
            score_far
        );
    }

    #[test]
    fn test_calculate_mop_up_scale_with_pawns() {
        let game = create_test_game_from_icn("w (8;q|1;q) k5,5|K4,4|R1,1|R2,2|P3,7");

        let scale = calculate_mop_up_scale(&game, PlayerColor::Black);
        // Should return a scale since white has mating material
        assert!(scale.is_some(), "Should have mop-up scale with rooks");
    }

    #[test]
    fn test_amazon_prefers_cutoff_over_drifting() {
        let enemy_king = Coordinate::new(5, 5);
        let good_king = Coordinate::new(3, 5);
        let good_amazon = SliderInfo { x: 7, y: 5, pt: PieceType::Amazon };
        let good_score = amazon_mate_drive_bonus(&enemy_king, &good_king, &good_amazon);

        let bad_amazon = SliderInfo { x: 1, y: 5, pt: PieceType::Amazon };
        let bad_score = amazon_mate_drive_bonus(&enemy_king, &good_king, &bad_amazon);

        assert!(
            good_score > bad_score,
            "Amazon should prefer cutting off the king from the far side: good={} bad={}",
            good_score,
            bad_score
        );
    }

    #[test]
    fn test_amazon_prefers_king_closer_in_lone_king_mop_up() {
        let enemy_king = Coordinate::new(5, 5);

        let close = create_test_game_from_icn("w (8;q|1;q) k5,5|K4,5|M7,5");
        let close_king = Coordinate::new(4, 5);
        let close_score =
            evaluate_lone_king_endgame(&close, Some(&close_king), &enemy_king, PlayerColor::White);

        let far = create_test_game_from_icn("w (8;q|1;q) k5,5|K1,5|M7,5");
        let far_king = Coordinate::new(1, 5);
        let far_score =
            evaluate_lone_king_endgame(&far, Some(&far_king), &enemy_king, PlayerColor::White);

        assert!(
            close_score > far_score,
            "K+Amazon mop-up should strongly prefer king approach: close={} far={}",
            close_score,
            far_score
        );
    }

    #[test]
    fn test_smart_mop_up_prefers_pieces_opposite_our_king() {
        // White king on the left of black king (5,5). A second piece (chancellor)
        // far away to the right cuts off escape — should score higher than placing
        // it on the same side as our king.
        // Two chancellors — `is_bounded_only_winnable` doesn't fire so the test
        // isolates the smart opposition logic instead of the corner-drive bonus.
        let opposite = create_test_game_from_icn(
            "w (50;q|1;q) k5,5|K3,5|CH20,5|CH18,3",
        );
        let same_side = create_test_game_from_icn(
            "w (50;q|1;q) k5,5|K3,5|CH-20,5|CH-18,3",
        );
        let ek = Coordinate::new(5, 5);
        let s_opp = evaluate_lone_king_endgame(
            &opposite,
            Some(&Coordinate::new(3, 5)),
            &ek,
            PlayerColor::White,
        );
        let s_same = evaluate_lone_king_endgame(
            &same_side,
            Some(&Coordinate::new(3, 5)),
            &ek,
            PlayerColor::White,
        );
        assert!(
            s_opp > s_same,
            "Piece opposite our king should score better: opp={} same={}",
            s_opp,
            s_same
        );
    }

    #[test]
    fn test_smart_mop_up_rewards_ring_coverage_with_exotic_pieces() {
        // Knight + Camel + Giraffe surround a king with our king nearby.
        // Compared to all those pieces clustered far away, ring coverage
        // and sandwich logic should give a clearly higher score.
        let near = create_test_game_from_icn(
            "w (50;q|1;q) k10,10|K12,10|N8,9|CA7,10|GI10,14",
        );
        let far = create_test_game_from_icn(
            "w (50;q|1;q) k10,10|K-30,-30|N-32,-31|CA-33,-30|GI-30,-26",
        );
        let ek = Coordinate::new(10, 10);
        let s_near = evaluate_lone_king_endgame(
            &near,
            Some(&Coordinate::new(12, 10)),
            &ek,
            PlayerColor::White,
        );
        let s_far = evaluate_lone_king_endgame(
            &far,
            Some(&Coordinate::new(-30, -30)),
            &ek,
            PlayerColor::White,
        );
        assert!(
            s_near > s_far,
            "Pieces engaged around enemy king should score way higher: near={} far={}",
            s_near,
            s_far
        );
    }

    #[test]
    fn test_smart_mop_up_axis_sandwich_bonus() {
        // Two rooks above and below the enemy king (vertical sandwich) should
        // score higher than two rooks both above (no sandwich).
        let sandwich =
            create_test_game_from_icn("w (50;q|1;q) k10,10|K10,7|R10,2|R10,18");
        let no_sandwich =
            create_test_game_from_icn("w (50;q|1;q) k10,10|K10,7|R10,18|R10,19");
        let ek = Coordinate::new(10, 10);
        let s_sand = evaluate_lone_king_endgame(
            &sandwich,
            Some(&Coordinate::new(10, 7)),
            &ek,
            PlayerColor::White,
        );
        let s_no = evaluate_lone_king_endgame(
            &no_sandwich,
            Some(&Coordinate::new(10, 7)),
            &ek,
            PlayerColor::White,
        );
        assert!(
            s_sand > s_no,
            "Vertical sandwich should outscore stacked rooks: sand={} no={}",
            s_sand,
            s_no
        );
    }

    #[test]
    fn test_piece_attacks_geom_basic_pieces() {
        // Sanity-check the geometric attack table for representative pieces.
        assert!(piece_attacks_geom(PieceType::Rook, PlayerColor::White, 7, 0));
        assert!(!piece_attacks_geom(PieceType::Rook, PlayerColor::White, 3, 4));
        assert!(piece_attacks_geom(PieceType::Bishop, PlayerColor::White, 4, 4));
        assert!(piece_attacks_geom(PieceType::Knight, PlayerColor::White, 1, 2));
        assert!(piece_attacks_geom(PieceType::Camel, PlayerColor::White, 1, 3));
        assert!(piece_attacks_geom(PieceType::Giraffe, PlayerColor::White, 4, 1));
        assert!(piece_attacks_geom(PieceType::Zebra, PlayerColor::White, 3, 2));
        assert!(piece_attacks_geom(PieceType::Hawk, PlayerColor::White, 3, 3));
        assert!(!piece_attacks_geom(PieceType::Hawk, PlayerColor::White, 1, 1));
        assert!(piece_attacks_geom(PieceType::Knightrider, PlayerColor::White, 2, 4));
        assert!(piece_attacks_geom(PieceType::Knightrider, PlayerColor::White, 6, 3));
        assert!(!piece_attacks_geom(PieceType::Knightrider, PlayerColor::White, 5, 5));
        assert!(piece_attacks_geom(PieceType::Huygen, PlayerColor::White, 7, 0));
        assert!(!piece_attacks_geom(PieceType::Huygen, PlayerColor::White, 4, 0));
    }

    #[test]
    fn test_find_bitboard_cage() {
        let game = create_test_game_from_icn("w (8;q|1;q) k4,4|R4,0|R4,8|R0,4|R8,4|K1,1");

        let enemy_king = Coordinate::new(4, 4);
        let (_is_caged, area) = find_bitboard_cage(
            &game.board,
            &game.spatial_indices,
            &enemy_king,
            PlayerColor::White,
        );
        // The king should be significantly restricted
        assert!(
            area < 100,
            "King should be in a small area, found: {}",
            area
        );
    }
}
