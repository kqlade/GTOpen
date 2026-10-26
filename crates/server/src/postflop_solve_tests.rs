//! The one-shot postflop route: build, solve and read back in one request.
use super::*;

const BOARD: &str = "Ks7h2d";

fn request(read: serde_json::Value) -> PostflopSolveRequest {
    let sizes = serde_json::json!([
        {"bet": "50", "raise": "60", "donk": ""},
        {"bet": "50", "raise": "60", "donk": ""},
        {"bet": "50", "raise": "60", "donk": ""}
    ]);
    serde_json::from_value(serde_json::json!({
        "spot": {
            "board": BOARD,
            "range_oop": "QQ,JJ,TT,99,AhKh,AsKs,87s",
            "range_ip": "AA,KK,AQs,JTs,T9s",
            "starting_pot": 10, "effective_stack": 100,
            "max_raises": 1, "oop": sizes, "ip": sizes
        },
        "solve": {"max_iterations": 20, "target_exploit_pct": 0.0, "check_every": 10},
        "read": read
    }))
    .unwrap()
}

/// A solved tiny spot to read from directly.
fn solved() -> Solver {
    let config = request(serde_json::json!({})).spot.to_spot_config().unwrap();
    let mut solver = Solver::with_storage(Arc::new(Spot::new(config).unwrap()), Storage::F32);
    for _ in 0..20 {
        solver.iterate();
    }
    solver.ensure_symmetric();
    solver
}

fn solve(req: &PostflopSolveRequest) -> Result<PostflopSolveResult, String> {
    let config = req.spot.to_spot_config()?;
    solve_postflop(config, Storage::F32, 1e9, 1_000_000, &req.solve, &req.read)
}

fn has(nodes: &[SolvedNode], path: &[PathStep]) -> bool {
    nodes.iter().any(|n| n.path == path)
}

fn dealt(path: &[PathStep]) -> usize {
    path.iter().filter(|s| matches!(s, PathStep::Card { .. })).count()
}

/// Every action node read has every child read: a street is never half there.
fn assert_streets_whole(nodes: &[SolvedNode]) {
    for n in nodes.iter().filter(|n| n.node.node_type == "action") {
        for index in 0..n.node.actions.len() {
            let mut child = n.path.clone();
            child.push(PathStep::Action { index });
            assert!(has(nodes, &child), "missing child {}", path_text(&child));
        }
    }
}

#[test]
fn the_request_has_defaults_for_everything_but_the_spot() {
    let req: PostflopSolveRequest = serde_json::from_value(serde_json::json!({
        "spot": serde_json::to_value(&request(serde_json::json!({})).spot).unwrap()
    }))
    .unwrap();
    assert_eq!(req.solve.max_iterations, 2000);
    assert_eq!(req.read.lines, 4);
    assert!(req.read.cards.is_empty() && req.read.paths.is_empty());
    assert_eq!(req.read.min_reach, 0.0);
}

#[test]
fn the_flop_street_comes_whole_and_nothing_else_without_cards() {
    let res = solve(&request(serde_json::json!({}))).unwrap();
    assert_eq!(res.iterations, 20);
    assert_eq!(res.engine, "cpu");
    assert!(res.exploit_pct >= 0.0 && res.seconds > 0.0);
    assert_eq!(res.tree.board, BOARD);
    assert!(has(&res.nodes, &[]));
    assert!(res.nodes.iter().all(|n| dealt(&n.path) == 0));
    assert!(res.nodes.iter().any(|n| n.node.node_type == "chance"));
    assert_streets_whole(&res.nodes);
}

#[test]
fn line_frequency_is_a_probability_over_a_nodes_actions() {
    let solver = solved();
    assert_eq!(solver.line_frequency(&[]).unwrap(), 1.0);
    let root = solver.node_view(&[]).unwrap();
    let total: f64 = (0..root.actions.len())
        .map(|index| solver.line_frequency(&[PathStep::Action { index }]).unwrap())
        .sum();
    assert!((total - 1.0).abs() < 1e-4, "root actions sum to {total}");
    let check = solver.line_frequency(&[PathStep::Action { index: 0 }]).unwrap();
    let check_check = solver
        .line_frequency(&[PathStep::Action { index: 0 }, PathStep::Action { index: 0 }])
        .unwrap();
    assert!(check_check <= check + 1e-9);
    assert!(solver.line_frequency(&[PathStep::Action { index: 9 }]).is_err());
}

#[test]
fn cards_follow_the_most_taken_lines_and_the_lines_asked_for() {
    let solver = solved();
    let flop = read_tree(&solver, &ReadRequest::default()).unwrap();
    let mut deals: Vec<(Vec<PathStep>, f64)> = flop
        .iter()
        .filter(|n| n.node.node_type == "chance")
        .map(|n| (n.path.clone(), solver.line_frequency(&n.path).unwrap()))
        .collect();
    deals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert!(deals.len() >= 3, "{} closing lines", deals.len());
    let rarest = deals.last().unwrap().0.clone();
    let mut asked = rarest.clone();
    asked.push(PathStep::Card { card: "Qh".into() });
    asked.push(PathStep::Action { index: 0 });

    let read: ReadRequest = serde_json::from_value(serde_json::json!({
        "cards": ["Qh"], "lines": 2, "paths": [asked]
    }))
    .unwrap();
    let nodes = read_tree(&solver, &read).unwrap();
    assert_streets_whole(&nodes);
    let mut expected: Vec<Vec<PathStep>> = deals.iter().take(2).map(|(p, _)| p.clone()).collect();
    expected.push(rarest);
    for line in &deals {
        let mut turn = line.0.clone();
        turn.push(PathStep::Card { card: "Qh".into() });
        assert_eq!(has(&nodes, &turn), expected.contains(&line.0), "{}", path_text(&turn));
    }
    for n in nodes.iter().filter(|n| dealt(&n.path) == 1) {
        assert_eq!(n.node.board, ["Ks", "7h", "2d", "Qh"]);
    }
    // the flop street is there in full as well
    for n in &flop {
        assert!(has(&nodes, &n.path));
    }
}

#[test]
fn a_line_that_does_not_end_at_a_deal_is_refused() {
    let solver = solved();
    let read: ReadRequest = serde_json::from_value(serde_json::json!({
        "cards": ["Qh"], "lines": 1,
        "paths": [[{"type": "action", "index": 0}, {"type": "card", "card": "Qh"}]]
    }))
    .unwrap();
    let err = read_tree(&solver, &read).unwrap_err();
    assert!(err.contains("does not end at a deal"), "{err}");
}

#[test]
fn a_card_already_on_the_board_is_refused() {
    let err = solve(&request(serde_json::json!({"cards": ["Ks"]}))).unwrap_err();
    assert!(err.contains("Ks"), "{err}");
    let too_many = request(serde_json::json!({"cards": ["Qh", "Qc", "Qd"]}));
    assert_eq!(too_many.read.cards.len(), 3); // the handler refuses it before building
}

#[test]
fn min_reach_leaves_out_the_hands_that_hardly_get_there() {
    let solver = solved();
    let all = read_tree(&solver, &ReadRequest::default()).unwrap();
    let read: ReadRequest = serde_json::from_value(serde_json::json!({"min_reach": 0.5})).unwrap();
    let some = read_tree(&solver, &read).unwrap();
    assert_eq!(all.len(), some.len());
    let hands = |nodes: &[SolvedNode]| -> usize {
        nodes.iter().map(|n| n.node.players.iter().map(|p| p.hands.len()).sum::<usize>()).sum()
    };
    assert!(hands(&some) < hands(&all));
    for n in &some {
        for p in &n.node.players {
            assert!(p.hands.iter().all(|h| h.reach >= 0.5));
        }
    }
}
