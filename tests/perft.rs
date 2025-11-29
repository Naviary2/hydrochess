use hydrochess_wasm::game::GameState;
use std::time::Instant;
use hydrochess_wasm::search::negamax_node_count_for_depth;

#[test]
fn run_perft_suite() {
    println!("\n================================================================");
    println!("Running Perft Suite for HydroChess WASM");
    println!("================================================================");
    println!("NOTE: Move counts vary based on slider move heuristics");
    println!("================================================================");

    // Setup Standard Chess
    let mut game = GameState::new();
    game.setup_standard_chess();

    println!("Initial Board Setup: Standard Chess");
    println!("----------------------------------------------------------------");

    let max_depth = 3;
    let mut last_duration = None;

    let mut total_perft_nodes: u128 = 0;
    let mut total_perft_micros: u128 = 0;
    let mut total_search_nodes: u128 = 0;
    let mut total_search_micros: u128 = 0;

    for depth in 1..=max_depth {
        // Perft timing
        let perft_start = Instant::now();
        let perft_nodes = game.perft(depth);
        let perft_duration = perft_start.elapsed();
        let perft_micros = perft_duration.as_micros().max(1);
        let perft_nps = (perft_nodes as u128 * 1_000_000) / perft_micros;

        total_perft_nodes += perft_nodes as u128;
        total_perft_micros += perft_micros;

        // Search timing
        let search_start = Instant::now();
        let searched_nodes = negamax_node_count_for_depth(&mut game, depth);
        let search_duration = search_start.elapsed();
        let search_micros = search_duration.as_micros().max(1);
        let search_nps = (searched_nodes as u128 * 1_000_000) / search_micros;

        total_search_nodes += searched_nodes as u128;
        total_search_micros += search_micros;

        println!(
            "Depth {}: perft {:10} | searched {:10} | \
             Time P: {:?} | NPS P: {:10} | Time S: {:?} | NPS S: {:10}",
            depth,
            perft_nodes,
            searched_nodes,
            perft_duration,
            perft_nps,
            search_duration,
            search_nps
        );

        last_duration = Some(perft_duration);
    }

    println!("================================================================");
    println!("Performance Summary:");
    if total_perft_micros > 0 {
        let avg_perft_nps = (total_perft_nodes * 1_000_000) / total_perft_micros.max(1);
        println!("  Avg perft NPS over depths 1..{}: {}", max_depth, avg_perft_nps);
    }
    if total_search_micros > 0 {
        let avg_search_nps = (total_search_nodes * 1_000_000) / total_search_micros.max(1);
        println!("  Avg search NPS over depths 1..{}: {}", max_depth, avg_search_nps);
    }
    if let Some(duration_d_max) = last_duration {
        println!("  Depth {} perft completed in {:?}", max_depth, duration_d_max);
    }
    println!("================================================================");
}

#[test]
fn run_search_only_suite() {
    println!("\n================================================================");
    println!("Running Search-Only Suite for HydroChess WASM");
    println!("================================================================");

    // Setup Standard Chess
    let mut game = GameState::new();
    game.setup_standard_chess();

    println!("Initial Board Setup: Standard Chess");
    println!("----------------------------------------------------------------");

    let max_depth = 5;
    let mut last_duration = None;

    let mut total_search_nodes: u128 = 0;
    let mut total_search_micros: u128 = 0;

    for depth in 1..=max_depth {
        let search_start = Instant::now();
        let searched_nodes = negamax_node_count_for_depth(&mut game, depth);
        let search_duration = search_start.elapsed();
        let search_micros = search_duration.as_micros().max(1);
        let search_nps = (searched_nodes as u128 * 1_000_000) / search_micros;

        total_search_nodes += searched_nodes as u128;
        total_search_micros += search_micros;

        println!(
            "Depth {}: searched {:10} | Time: {:?} | NPS: {:10}",
            depth,
            searched_nodes,
            search_duration,
            search_nps
        );

        last_duration = Some(search_duration);
    }

    println!("================================================================");
    println!("Search-Only Performance Summary:");
    if total_search_micros > 0 {
        let avg_search_nps = (total_search_nodes * 1_000_000) / total_search_micros.max(1);
        println!("  Avg search NPS over depths 1..{}: {}", max_depth, avg_search_nps);
    }
    if let Some(duration_d_max) = last_duration {
        println!("  Depth {} search completed in {:?}", max_depth, duration_d_max);
    }
    println!("================================================================");
}