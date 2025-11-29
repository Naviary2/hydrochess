use hydrochess_wasm::game::GameState;
use hydrochess_wasm::board::{Board, Piece, PieceType, PlayerColor, Coordinate};
use std::collections::HashSet;
use hydrochess_wasm::moves::{get_legal_moves, Move};

#[test]
fn test_repro_crash() {
    let mut board = Board::new();
    
    // Setup pieces as per the log
    // White Pawns at y=2, x=1..8
    for x in 1..=8 {
        board.set_piece(x, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
    }
    // Black Pawns at y=7, x=1..8
    for x in 1..=8 {
        board.set_piece(x, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
    }
    
    // Rooks
    board.set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
    board.set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White));
    board.set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
    board.set_piece(8, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
    
    // Knights
    board.set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(2, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
    board.set_piece(7, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
    
    // Bishops
    board.set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
    board.set_piece(6, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
    board.set_piece(3, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
    board.set_piece(6, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
    
    // Queens
    board.set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
    board.set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
    
    // Kings
    board.set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));

    // Castling rights (includes pawns as per log)
    let mut castling_rights = HashSet::new();
    // White Pawns
    for x in 1..=8 { castling_rights.insert(Coordinate::new(x, 2)); }
    // Black Pawns
    for x in 1..=8 { castling_rights.insert(Coordinate::new(x, 7)); }
    // Rooks and Kings
    castling_rights.insert(Coordinate::new(1, 1));
    castling_rights.insert(Coordinate::new(8, 1));
    castling_rights.insert(Coordinate::new(5, 1));
    
    castling_rights.insert(Coordinate::new(1, 8));
    castling_rights.insert(Coordinate::new(8, 8));
    castling_rights.insert(Coordinate::new(5, 8));

    let mut game = GameState {
        board,
        turn: PlayerColor::White,
        castling_rights,
        en_passant: None,
        halfmove_clock: 0,
        fullmove_number: 1,
        material_score: 0,
        hash_stack: Vec::new(),
        null_moves: 0,
    };

    println!("Starting search...");
    // Updated signature: get_best_move(&mut GameState, max_depth)
    let best_move = hydrochess_wasm::search::get_best_move(&mut game, 4); // Depth 4 to trigger some search
    println!("Best move: {:?}", best_move);
}


#[test]
fn replay_bug_move_sequence() {
    println!("\n================================================================");
    println!("Replaying suspected bug move sequence");
    println!("================================================================");

    let mut game = GameState::new();
    game.setup_standard_chess();

    // First: make a simple white opening move so that the following
    // sequence starts with Black to move, matching the JS logs.
    {
        let fx = 1;
        let fy = 2;
        let tx = 1;
        let ty = 3;
        let piece = game
            .board
            .get_piece(&fx, &fy)
            .cloned()
            .expect("Expected white pawn at (1,2) for opening move");
        let mv = Move::new(Coordinate::new(fx, fy), Coordinate::new(tx, ty), piece);
        let _undo = game.make_move(&mv);
        assert!(
            !game.is_move_illegal(),
            "Opening white move (1,2)->(1,3) is illegal"
        );
    }

    // Sequence captured from JS engine logs (in board coordinates)
    // This now correctly starts with Black to move after White's pawn move.
    let moves: &[(i64, i64, i64, i64)] = &[
        // Move 1 - black
        (1, 8, 0, 8),
        // Move 2 - white
        (8, 1, 9, 1),
        // Move 3 - black
        (8, 8, 10, 8),
        // Move 4 - white
        (1, 1, -1, 1),
        // Move 5 - black
        (7, 8, 6, 6),
        // Move 6 - white
        (2, 1, 3, 3),
        // Move 7 - black
        (2, 8, 3, 6),
        // Move 8 - white
        (-1, 1, -1, 8),
        // Move 9 - black
        (0, 8, -1, 8),
        // Move 10 - white
        (6, 1, -1, 8),
        // Move 11 - black
        (10, 8, 10, 2),
        // Move 12 - white
        (-1, 8, 2, 11),
    ];

    for (idx, (fx, fy, tx, ty)) in moves.iter().cloned().enumerate() {
        println!(
            "Applying move {}: from ({}, {}) to ({}, {})",
            idx + 1,
            fx,
            fy,
            tx,
            ty
        );

        // Before move 10 (idx=9), check if the bishop move is actually legal
        if idx == 9 {
            println!("\n>>> Checking legal moves BEFORE move 10 <<<");
            let legal_moves = get_legal_moves(&game.board, game.turn, &game.castling_rights, &game.en_passant);

            // Find bishop moves from (6,1)
            let bishop_moves: Vec<_> = legal_moves
                .iter()
                .filter(|m| m.from.x == 6 && m.from.y == 1)
                .collect();

            println!("Pieces at (6,1): {:?}", game.board.get_piece(&6, &1));
            println!("Pieces at (5,2): {:?}", game.board.get_piece(&5, &2));
            println!("Bishop at (6,1) has {} legal moves:", bishop_moves.len());
            for m in &bishop_moves {
                println!("  -> ({}, {})", m.to.x, m.to.y);
            }

            // Check if the illegal move (-1,8) is in the list
            let has_illegal = bishop_moves.iter().any(|m| m.to.x == -1 && m.to.y == 8);
            if has_illegal {
                println!("!!! BUG: Move to (-1,8) is in legal moves but should be blocked!");
            } else {
                println!("OK: Move to (-1,8) is NOT in legal moves (correctly blocked)");
            }
            println!("<<<\n");
        }

        // Fetch the moving piece from the current board position
        let piece = match game.board.get_piece(&fx, &fy) {
            Some(p) => p.clone(),
            None => {
                panic!(
                    "No piece found at from-square ({}, {}) before move {}",
                    fx,
                    fy,
                    idx + 1
                );
            }
        };

        let mv = Move::new(
            Coordinate::new(fx, fy),
            Coordinate::new(tx, ty),
            piece,
        );

        // Apply move directly through GameState
        let _undo = game.make_move(&mv);

        // If this triggers a panic or illegal state, the test will clearly surface it
        if game.is_move_illegal() {
            panic!("Move {} left side to move in check (illegal).", idx + 1);
        }
    }

    println!("Finished replaying move sequence without immediate illegal-move detection.");
    println!("================================================================");
}
