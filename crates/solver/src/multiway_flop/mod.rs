//! Flop solver for three or more seats, built to measure the preflop
//! engine's multiway terminals (see `docs/multiway_flop_solver.md`).
//!
//! One street from the flop (or the flop and a sampled turn), every seat's
//! strategy a vector over the 1,326 combos, leaves checked down by exact
//! showdown over a fixed sample of runouts. CPU only, one solve per call.
//! Besides the per-class values the preflop engine consumes, a solve can
//! read back the tree node by node (`read`, view.rs) for a coach to walk.

pub mod showdown;
pub mod solve;
pub mod tree;
pub mod view;

use crate::cards::{cards_to_string, combo_from_index, parse_cards, rank, suit, Card, NUM_COMBOS};
use crate::preflop::equity::{class_index, NUM_CLASSES};
use crate::query::PathStep;
use crate::range::Range;
use serde::{Deserialize, Serialize};
use showdown::{sample_cards, sample_cards_with, sample_runouts, Board};
use solve::Solver;
use tree::{Builder, TreeConfig, KIND_CHANCE};
pub use view::{NodeView, SolvedNode};

#[derive(Debug, Clone, Deserialize)]
pub struct SeatInput {
    pub position: String,
    /// Chips this seat put in before the flop.
    pub invested: f64,
    /// Per hand class (engine order): the probability each combo of the
    /// class takes the line to this flop, in [0, 1]. One of `weights` and
    /// `range`.
    #[serde(default)]
    pub weights: Option<Vec<f32>>,
    /// The range in solver notation ("22+,A2s+,KhQh:0.5"): per combo, so a
    /// suit-specific read keeps its suits.
    #[serde(default)]
    pub range: Option<String>,
}

/// Which nodes of the solved tree come back. The flop street always comes
/// whole, root to every deal; a turn card in `cards` is then dealt at the
/// `lines` most-taken closing lines plus every line in `paths` that reaches
/// a deal, and the turn street below it comes whole as well.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadRequest {
    /// The turn to deal (one card at most: the tree ends on the turn). The
    /// card joins the sampled turn cards as if named in `turn_cards_fixed`.
    #[serde(default)]
    pub cards: Vec<String>,
    #[serde(default = "default_read_lines")]
    pub lines: usize,
    #[serde(default)]
    pub paths: Vec<Vec<PathStep>>,
    /// Hands reaching a node with less than this probability are left out.
    #[serde(default)]
    pub min_reach: f32,
}

fn default_read_lines() -> usize {
    4
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveRequest {
    /// Three cards, e.g. "Ks7h2d".
    pub board: String,
    /// Chips in the pot on the flop, folded seats' money included.
    pub pot: f64,
    /// Chips every live seat has behind.
    pub stack: f64,
    /// Rake as a percentage of the pot (5 = 5%), like the rest of the API
    /// and the preflop config.
    #[serde(default)]
    pub rake_pct: f64,
    /// Rake cap in chips (0: uncapped).
    #[serde(default)]
    pub rake_cap: f64,
    /// Live seats in postflop acting order.
    pub seats: Vec<SeatInput>,
    /// Bet sizes as fractions of the pot.
    #[serde(default = "default_bet_sizes")]
    pub bet_sizes: Vec<f64>,
    /// Raise sizes as multiples of the bet faced.
    #[serde(default = "default_raise_sizes")]
    pub raise_sizes: Vec<f64>,
    /// Raise sizes as fractions of the pot after the call (tree.rs).
    #[serde(default)]
    pub raise_pot_fractions: Vec<f64>,
    #[serde(default = "default_max_raises")]
    pub max_raises: u8,
    #[serde(default = "default_allin_threshold")]
    pub allin_threshold: f64,
    /// Sampled turn cards to deal (0: the tree ends on the flop).
    #[serde(default)]
    pub turn_cards: usize,
    /// Turn cards that must be among the sampled ones ("Td"), each standing
    /// in for a sampled card of its texture class.
    #[serde(default)]
    pub turn_cards_fixed: Vec<String>,
    /// Turn bet sizes, raise sizes and raise cap; the flop's when absent
    /// (an empty list: check or call only).
    #[serde(default)]
    pub turn_bet_sizes: Option<Vec<f64>>,
    #[serde(default)]
    pub turn_raise_sizes: Option<Vec<f64>>,
    #[serde(default)]
    pub turn_raise_pot_fractions: Option<Vec<f64>>,
    #[serde(default)]
    pub turn_max_raises: Option<u8>,
    /// Board completions each showdown averages over.
    #[serde(default = "default_runouts")]
    pub runouts: usize,
    #[serde(default = "default_iterations")]
    pub iterations: u32,
    #[serde(default = "default_check_every")]
    pub check_every: u32,
    /// Stop once the summed best-response gap is under this share of the pot.
    #[serde(default = "default_target_gap_pct")]
    pub target_gap_pct: f64,
    #[serde(default = "default_max_arena_mb")]
    pub max_arena_mb: f64,
    /// Nodes to read back after the solve (none: values only).
    #[serde(default)]
    pub read: Option<ReadRequest>,
}

fn default_bet_sizes() -> Vec<f64> {
    vec![0.33, 0.75]
}
fn default_raise_sizes() -> Vec<f64> {
    vec![2.5]
}
fn default_max_raises() -> u8 {
    2
}
fn default_allin_threshold() -> f64 {
    0.65
}
fn default_runouts() -> usize {
    40
}
fn default_iterations() -> u32 {
    300
}
fn default_check_every() -> u32 {
    25
}
fn default_target_gap_pct() -> f64 {
    1.0
}
fn default_max_arena_mb() -> f64 {
    1500.0
}

#[derive(Debug, Clone, Serialize)]
pub struct SeatResult {
    pub position: String,
    /// Per class: chips the seat owns at the end of the hand (preflop
    /// investment included), averaged over the class's combos weighted by
    /// reach × the other seats' compatible mass.
    pub values: Vec<f32>,
    /// Per class: that weight.
    pub masses: Vec<f32>,
    /// The whole range: chips owned at the end under the average strategies.
    pub range_ev: f64,
    /// … and when best-responding to the others' average strategies.
    pub best_response_ev: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RootAction {
    pub action: String,
    pub frequency: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SolveResult {
    pub board: String,
    pub nodes: usize,
    pub action_nodes: usize,
    pub arena_mb: f64,
    pub runouts: usize,
    pub turn_cards: Vec<String>,
    pub iterations: u32,
    /// Summed best-response gap as a percentage of the flop pot.
    pub gap_pct: f64,
    pub gap_history: Vec<(u32, f64)>,
    pub seconds: f64,
    pub seats: Vec<SeatResult>,
    /// The first seat's action split at the root.
    pub root_actions: Vec<RootAction>,
    /// The nodes `read` asked for, a parent before its children (empty
    /// without a `read`).
    pub views: Vec<SolvedNode>,
}

/// Root reach of a seat: its class weights spread over the combos, minus the
/// ones the board blocks.
pub fn root_reach(weights: &[f32], board: &[Card]) -> Vec<f32> {
    let mut per_combo = vec![0f32; NUM_COMBOS];
    for (idx, slot) in per_combo.iter_mut().enumerate() {
        let (a, b) = combo_from_index(idx);
        *slot = weights[class_index(rank(a), rank(b), suit(a) == suit(b))].clamp(0.0, 1.0);
    }
    root_reach_combos(&per_combo, board)
}

/// Root reach of a seat from per-combo weights, minus the combos the board blocks.
pub fn root_reach_combos(weights: &[f32], board: &[Card]) -> Vec<f32> {
    let mut blocked = 0u64;
    for &c in board {
        blocked |= 1 << c;
    }
    let mut out = vec![0f32; NUM_COMBOS];
    for (idx, slot) in out.iter_mut().enumerate() {
        let (a, b) = combo_from_index(idx);
        if blocked & (1 << a) != 0 || blocked & (1 << b) != 0 {
            continue;
        }
        *slot = weights[idx].clamp(0.0, 1.0);
    }
    out
}

/// A seat's root reach from whichever of `weights` and `range` it gives.
fn seat_reach(seat: &SeatInput, flop: &[Card]) -> Result<Vec<f32>, String> {
    let reach = match (&seat.weights, &seat.range) {
        (Some(weights), None) => {
            if weights.len() != NUM_CLASSES {
                return Err(format!("{}: weights must have {NUM_CLASSES} entries", seat.position));
            }
            if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return Err(format!("{}: weights must be finite and not negative", seat.position));
            }
            root_reach(weights, flop)
        }
        (None, Some(range)) => {
            let parsed = Range::parse(range).map_err(|e| format!("{}: {e}", seat.position))?;
            root_reach_combos(&parsed.weights, flop)
        }
        _ => return Err(format!("{}: give weights or range, one of the two", seat.position)),
    };
    if reach.iter().all(|w| *w <= 0.0) {
        return Err(format!("{}: the range is empty on this flop", seat.position));
    }
    Ok(reach)
}

pub fn solve(req: &SolveRequest) -> Result<SolveResult, String> {
    let started = std::time::Instant::now();
    let flop = parse_cards(&req.board)?;
    if flop.len() != 3 {
        return Err("board must be a flop: three cards".into());
    }
    if !(2..=tree::MAX_SEATS).contains(&req.seats.len()) {
        return Err(format!("seats must number 2 to {}", tree::MAX_SEATS));
    }
    let reach: Vec<Vec<f32>> = req.seats.iter().map(|s| seat_reach(s, &flop)).collect::<Result<_, _>>()?;
    let mut fixed: Vec<Card> = Vec::new();
    for text in req.turn_cards_fixed.iter().chain(req.read.iter().flat_map(|r| r.cards.iter())) {
        let cards = parse_cards(text)?;
        let [c] = cards[..] else {
            return Err(format!("{text:?} is not one turn card"));
        };
        if flop.contains(&c) {
            return Err(format!("the turn {text} is on the flop"));
        }
        if !fixed.contains(&c) {
            fixed.push(c);
        }
    }
    if let Some(read) = &req.read {
        if read.cards.len() > 1 {
            return Err("read.cards is the turn alone: the multiway tree ends on the turn".into());
        }
        if !read.cards.is_empty() && req.turn_cards == 0 {
            return Err("read.cards names a turn but turn_cards is 0: no turn is dealt".into());
        }
        if read.lines == 0 && read.paths.is_empty() && !read.cards.is_empty() {
            return Err("read.lines is 0 and read.paths is empty: no line would reach the turn".into());
        }
    }
    if req.runouts == 0 {
        return Err("runouts must be at least 1".into());
    }
    if req.iterations == 0 || req.check_every == 0 {
        return Err("iterations and check_every must be at least 1".into());
    }
    if !(0.0..100.0).contains(&req.rake_pct) || !(req.rake_cap >= 0.0 && req.rake_cap.is_finite()) {
        return Err("rake_pct is a percentage of the pot in [0, 100) and rake_cap chips ≥ 0".into());
    }
    let cfg = TreeConfig {
        pot: req.pot,
        stack: req.stack,
        invested: req.seats.iter().map(|s| s.invested).collect(),
        rake_pct: req.rake_pct / 100.0,
        rake_cap: req.rake_cap,
        bet_sizes: req.bet_sizes.clone(),
        raise_sizes: req.raise_sizes.clone(),
        raise_pot_fractions: req.raise_pot_fractions.clone(),
        max_raises: req.max_raises,
        allin_threshold: req.allin_threshold,
        turn_cards: req.turn_cards,
        turn_bet_sizes: req.turn_bet_sizes.clone(),
        turn_raise_sizes: req.turn_raise_sizes.clone(),
        turn_raise_pot_fractions: req.turn_raise_pot_fractions.clone(),
        turn_max_raises: req.turn_max_raises,
    };
    let (turn_cards, turn_weights) =
        if req.turn_cards > 0 { sample_cards_with(&flop, req.turn_cards, &fixed) } else { (Vec::new(), Vec::new()) };
    let max_nodes = (req.max_arena_mb * 1e6 / (8.0 * NUM_COMBOS as f64)) as usize;
    let mut tree = Builder::build(&cfg, turn_cards.clone(), NUM_COMBOS, max_nodes.max(1000))?;
    tree.turn_weights = turn_weights;
    let arena_mb = tree.data_size.iter().sum::<usize>() as f64 * 8.0 / 1e6;
    if arena_mb > req.max_arena_mb {
        return Err(format!(
            "tree too large: {arena_mb:.0} MB of arenas over the {:.0} MB limit; fewer sizes, raises or turn cards",
            req.max_arena_mb
        ));
    }
    // Set 0 runs the flop out for all-in leaves: about `runouts` class-weighted
    // turn × river boards; set 1 + i runs turn card i out over `runouts` rivers.
    let mut boards: Vec<Vec<Board>> = Vec::with_capacity(1 + turn_cards.len());
    let turns = (req.runouts as f64).sqrt().ceil().max(1.0) as usize;
    let rivers = req.runouts.div_ceil(turns).max(1);
    boards.push(sample_runouts(&flop, turns, rivers).into_iter().map(|(cards, w)| Board::weighted(cards, w)).collect());
    for &t in &turn_cards {
        let four = [flop[0], flop[1], flop[2], t];
        let (cards, weights) = sample_cards(&four, req.runouts);
        boards.push(
            cards
                .into_iter()
                .zip(weights)
                .map(|(r, w)| Board::weighted([flop[0], flop[1], flop[2], t, r], w))
                .collect(),
        );
    }
    let mut solver = Solver::new(tree, boards, reach);
    let mut gap_history = Vec::new();
    let mut gap_pct = f64::INFINITY;
    let mut evs: Vec<(f64, f64)> = Vec::new();
    let mut done = 0;
    while done < req.iterations {
        solver.iterate();
        done += 1;
        if done % req.check_every == 0 || done == req.iterations {
            evs = solver.gap();
            let gap: f64 = evs.iter().map(|(ev, br)| (br - ev).max(0.0)).sum();
            gap_pct = gap / req.pot * 100.0;
            gap_history.push((done, gap_pct));
            if gap_pct <= req.target_gap_pct {
                break;
            }
        }
    }
    let mut seats = Vec::with_capacity(req.seats.len());
    for (p, seat) in req.seats.iter().enumerate() {
        let values = solver.root_values(p);
        let mass = solver.root_mass(p);
        let w = &solver.root_reach[p];
        let mut num = vec![0f64; NUM_CLASSES];
        let mut den = vec![0f64; NUM_CLASSES];
        for h in 0..NUM_COMBOS {
            if w[h] <= 0.0 {
                continue;
            }
            let (a, b) = combo_from_index(h);
            let class = class_index(rank(a), rank(b), suit(a) == suit(b));
            num[class] += w[h] as f64 * values[h] as f64;
            den[class] += w[h] as f64 * mass[h] as f64;
        }
        seats.push(SeatResult {
            position: seat.position.clone(),
            values: (0..NUM_CLASSES)
                .map(|c| if den[c] > 0.0 { (seat.invested + num[c] / den[c]) as f32 } else { 0.0 })
                .collect(),
            masses: den.iter().map(|d| *d as f32).collect(),
            range_ev: seat.invested + evs.get(p).map_or(0.0, |(ev, _)| *ev),
            best_response_ev: seat.invested + evs.get(p).map_or(0.0, |(_, br)| *br),
        });
    }
    let root = &solver.tree.nodes[0];
    let sigma = solver.average_strategy(0);
    let w = &solver.root_reach[root.seat as usize];
    let total: f64 = w.iter().map(|x| *x as f64).sum();
    let root_actions = root
        .actions
        .iter()
        .enumerate()
        .map(|(i, a)| RootAction {
            action: Solver::label(a, req.pot),
            frequency: if total > 0.0 {
                (0..NUM_COMBOS).map(|h| w[h] as f64 * sigma[i * NUM_COMBOS + h] as f64).sum::<f64>() / total
            } else {
                0.0
            },
        })
        .collect();
    let views = match &req.read {
        Some(read) => read_tree(&mut solver, &flop, read)?,
        None => Vec::new(),
    };
    Ok(SolveResult {
        board: cards_to_string(&flop),
        nodes: solver.tree.nodes.len(),
        action_nodes: solver.tree.action_nodes(),
        arena_mb,
        runouts: req.runouts,
        turn_cards: turn_cards.iter().map(|c| cards_to_string(&[*c])).collect(),
        iterations: done,
        gap_pct,
        gap_history,
        seconds: started.elapsed().as_secs_f64(),
        seats,
        root_actions,
        views,
    })
}

/// The nodes a read request names, a parent before its children: the flop
/// street whole, then the turn street under the card at the chosen lines.
fn read_tree(solver: &mut Solver, flop: &[Card], read: &ReadRequest) -> Result<Vec<SolvedNode>, String> {
    let mut nodes = Vec::new();
    let mut deals = Vec::new();
    read_street(solver, flop, Vec::new(), read.min_reach, &mut nodes, &mut deals)?;
    if let Some(card) = read.cards.first() {
        let mut lines: Vec<(Vec<PathStep>, f64)> = deals
            .drain(..)
            .map(|path| solver.walk(&path).map(|w| (path, w.frequency)))
            .collect::<Result<_, _>>()?;
        lines.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut chosen: Vec<Vec<PathStep>> = lines.into_iter().take(read.lines).map(|(path, _)| path).collect();
        for path in &read.paths {
            let line: Vec<PathStep> = path.iter().take_while(|s| matches!(s, PathStep::Action { .. })).cloned().collect();
            if line.len() == path.len() || chosen.contains(&line) {
                continue;
            }
            let walk = solver.walk(&line)?;
            if solver.tree.nodes[walk.node].kind != KIND_CHANCE {
                return Err(format!("read.paths: {} does not end at the deal", path_text(&line)));
            }
            chosen.push(line);
        }
        for line in chosen {
            let mut path = line;
            path.push(PathStep::Card { card: card.clone() });
            read_street(solver, flop, path, read.min_reach, &mut nodes, &mut deals)?;
        }
    }
    Ok(nodes)
}

/// Every node of the street below `path` down to the next deal, the deals
/// met collected for the turn.
fn read_street(
    solver: &mut Solver,
    flop: &[Card],
    path: Vec<PathStep>,
    min_reach: f32,
    nodes: &mut Vec<SolvedNode>,
    deals: &mut Vec<Vec<PathStep>>,
) -> Result<(), String> {
    let node = solver.node_view(flop, &path, min_reach)?;
    let (kind, actions) = (node.node_type.clone(), node.actions.len());
    nodes.push(SolvedNode { path: path.clone(), node });
    match kind.as_str() {
        "chance" => deals.push(path),
        "action" => {
            for index in 0..actions {
                let mut child = path.clone();
                child.push(PathStep::Action { index });
                read_street(solver, flop, child, min_reach, nodes, deals)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn path_text(path: &[PathStep]) -> String {
    path.iter()
        .map(|step| match step {
            PathStep::Action { index } => index.to_string(),
            PathStep::Card { card } => card.clone(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::equity::class_label;

    fn uniform(w: f32) -> Vec<f32> {
        vec![w; NUM_CLASSES]
    }

    fn request(seats: Vec<SeatInput>, pot: f64) -> SolveRequest {
        SolveRequest {
            board: "Ks7h2d".into(),
            pot,
            stack: 97.5,
            rake_pct: 0.0,
            rake_cap: 0.0,
            seats,
            bet_sizes: vec![0.5],
            raise_sizes: vec![2.5],
            raise_pot_fractions: Vec::new(),
            max_raises: 1,
            allin_threshold: 0.65,
            turn_cards: 0,
            turn_cards_fixed: Vec::new(),
            turn_bet_sizes: None,
            turn_raise_sizes: None,
            turn_raise_pot_fractions: None,
            turn_max_raises: None,
            runouts: 12,
            iterations: 40,
            check_every: 20,
            target_gap_pct: 0.0,
            max_arena_mb: 1500.0,
            read: None,
        }
    }

    fn seat(position: &str, weights: Vec<f32>) -> SeatInput {
        SeatInput { position: position.into(), invested: 2.5, weights: Some(weights), range: None }
    }

    fn ranged(position: &str, range: &str) -> SeatInput {
        SeatInput { position: position.into(), invested: 2.5, weights: None, range: Some(range.into()) }
    }

    fn read(cards: &[&str], paths: Vec<Vec<PathStep>>) -> Option<ReadRequest> {
        Some(ReadRequest { cards: cards.iter().map(|c| c.to_string()).collect(), lines: 2, paths, min_reach: 0.0 })
    }

    /// The read-back is the solve seen node by node: at every action node
    /// the actor's strategies sum to one and its EV is the strategy-weighted
    /// EV of its actions; the live seats' chips owned, weighted by
    /// reach × valid, add up to the raked pot; a seat given as a range
    /// string reads the same as the class weights it spells out.
    #[test]
    fn reads_the_tree_back_node_by_node() {
        let mut req = request(
            vec![seat("SB", uniform(1.0)), seat("BB", uniform(1.0)), ranged("BTN", "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s,A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+,42o+,32o")],
            7.5,
        );
        req.iterations = 20;
        req.check_every = 20;
        req.read = read(&[], Vec::new());
        let res = solve(&req).unwrap();
        assert!(res.views.len() > 5, "{} nodes read", res.views.len());
        let root = &res.views[0];
        assert!(root.path.is_empty());
        assert_eq!(root.node.node_type, "action");
        assert_eq!(root.node.player, Some(0));
        assert_eq!(root.node.put, vec![2.5; 3]);
        assert_eq!(root.node.board, vec!["Ks", "7h", "2d"]);
        assert_eq!(root.node.actions.iter().map(|a| a.kind.as_str()).collect::<Vec<_>>(), ["check", "bet"]);
        assert_eq!(root.node.actions[1].label, "Bet 3.75 (50%)");
        // BTN's range string covers every hand: as many combos reach as SB's uniform weights
        assert_eq!(root.node.players[2].hands.len(), root.node.players[0].hands.len());
        assert_eq!(root.node.players[0].hands.len(), 1176);
        let mut owned = 0f64;
        let mut weight = 0f64;
        for (p, player) in root.node.players.iter().enumerate() {
            for h in &player.hands {
                let w = h.reach as f64 * h.valid as f64;
                owned += w * h.ev.unwrap() as f64;
                weight += w;
                if p == 0 {
                    let s = h.strategy.as_ref().unwrap();
                    assert!((s.iter().sum::<f32>() - 1.0).abs() < 1e-4, "{s:?}");
                    let evs = h.evs.as_ref().unwrap();
                    let mixed: f32 = s.iter().zip(evs).map(|(p, e)| p * e.unwrap()).sum();
                    assert!((mixed - h.ev.unwrap()).abs() < 1e-3, "{}: ev {} vs Σσ·evs {mixed}", h.combo, h.ev.unwrap());
                } else {
                    assert!(h.strategy.is_none() && h.evs.is_none());
                }
                assert!(h.eq.is_none());
            }
        }
        // Three seats share the same reach × valid weight, so the sum over
        // all of them is three times the pot.
        assert!((owned / weight * 3.0 - 7.5).abs() < 0.2, "chips owned {}", owned / weight * 3.0);
        // After SB bets, BB faces fold / call / raise and SB's reach is the betting part of its range.
        let after_bet = res.views.iter().find(|n| n.path == vec![PathStep::Action { index: 1 }]).unwrap();
        assert_eq!(after_bet.node.player, Some(1));
        assert_eq!(after_bet.node.actions.iter().map(|a| a.kind.as_str()).collect::<Vec<_>>(), ["fold", "call", "raise"]);
        assert_eq!(after_bet.node.put, vec![6.25, 2.5, 2.5]);
        let sb_bets: f64 = after_bet.node.players[0].hands.iter().map(|h| h.reach as f64).sum();
        let sb_all: f64 = root.node.players[0].hands.iter().map(|h| h.reach as f64).sum();
        let bet_freq = root.node.players[0].hands.iter().map(|h| h.reach as f64 * h.strategy.as_ref().unwrap()[1] as f64).sum::<f64>() / sb_all;
        assert!((sb_bets / sb_all - bet_freq).abs() < 1e-4);
        // A folded seat shows no hands; the fold-out node names the winner's chips.
        let sb_folds = res.views.iter().find(|n| n.path == vec![PathStep::Action { index: 1 }, PathStep::Action { index: 0 }, PathStep::Action { index: 0 }]).unwrap();
        assert_eq!(sb_folds.node.node_type, "fold");
        assert!(sb_folds.node.players[1].hands.is_empty() && sb_folds.node.players[2].hands.is_empty());
        let won = sb_folds.node.players[0].hands[0].ev.unwrap();
        assert!((won - 11.25).abs() < 1e-3, "SB owns {won} after the others fold");
    }

    /// A turn card asked for is dealt even when the sample would not have
    /// picked it, and the line asked about is read under it.
    #[test]
    fn reads_the_turn_asked_for_along_the_line_asked_for() {
        let mut req = request(vec![seat("SB", uniform(1.0)), seat("BB", uniform(1.0)), seat("BTN", uniform(1.0))], 7.5);
        req.turn_cards = 3;
        req.turn_bet_sizes = Some(vec![]);
        req.runouts = 3;
        req.iterations = 2;
        req.check_every = 2;
        let check_check_check = vec![PathStep::Action { index: 0 }, PathStep::Action { index: 0 }, PathStep::Action { index: 0 }];
        let mut line = check_check_check.clone();
        line.push(PathStep::Card { card: "Qc".into() });
        req.read = read(&["Qc"], vec![line.clone()]);
        let res = solve(&req).unwrap();
        assert!(res.turn_cards.contains(&"Qc".to_string()), "{:?}", res.turn_cards);
        assert_eq!(res.turn_cards.len(), 3);
        let deal = res.views.iter().find(|n| n.path == check_check_check).unwrap();
        assert_eq!(deal.node.node_type, "chance");
        assert_eq!(deal.node.available_cards.as_ref().unwrap().len(), 3);
        let turn = res.views.iter().find(|n| n.path == line).unwrap();
        assert_eq!(turn.node.street, 1);
        assert_eq!(turn.node.board, vec!["Ks", "7h", "2d", "Qc"]);
        assert_eq!(turn.node.player, Some(0));
        // Queens (and every hand with a club queen) cannot be held on this turn.
        assert!(turn.node.players[1].hands.iter().all(|h| !h.combo.contains("Qc")));
        // Only `lines` deals come back beyond the one asked for.
        let turns = res.views.iter().filter(|n| matches!(n.path.last(), Some(PathStep::Card { .. }))).count();
        assert!((1..=3).contains(&turns), "{turns} turn roots");
        // Without a card in read.cards the flop street alone comes back, paths or not.
        req.read = read(&[], vec![line]);
        let res = solve(&req).unwrap();
        assert!(res.views.iter().all(|n| !n.path.iter().any(|s| matches!(s, PathStep::Card { .. }))));
        assert!(res.views.iter().any(|n| n.node.node_type == "chance"));
    }

    #[test]
    fn a_range_string_and_its_class_weights_solve_alike() {
        let mut classes = uniform(0.0);
        for (hi, lo, suited) in [(12, 12, false), (11, 11, false), (12, 11, true), (12, 11, false)] {
            classes[class_index(hi, lo, suited)] = 1.0;
        }
        let by_class = request(vec![seat("SB", classes), seat("BB", uniform(1.0)), seat("BTN", uniform(1.0))], 7.5);
        let mut by_range = by_class.clone();
        by_range.seats[0] = ranged("SB", "AA,KK,AKs,AKo");
        let (a, b) = (solve(&by_class).unwrap(), solve(&by_range).unwrap());
        assert!((a.seats[0].range_ev - b.seats[0].range_ev).abs() < 1e-6);
        assert_eq!(a.seats[0].values, b.seats[0].values);
        let mut bad = by_class.clone();
        bad.seats[0] = SeatInput { position: "SB".into(), invested: 2.5, weights: None, range: None };
        assert!(solve(&bad).unwrap_err().contains("one of the two"));
        bad.seats[0] = ranged("SB", "KsKd,Ks7h");
        assert!(solve(&bad).unwrap_err().contains("empty on this flop"));
    }

    #[test]
    fn three_identical_ranges_split_the_pot_and_the_gap_falls() {
        let req = request(
            vec![seat("SB", uniform(1.0)), seat("BB", uniform(1.0)), seat("BTN", uniform(1.0))],
            7.5,
        );
        let res = solve(&req).unwrap();
        assert_eq!(res.seats.len(), 3);
        let total: f64 = res.seats.iter().map(|s| s.range_ev).sum();
        // Chips owned add up to the pot: nobody wins chips from nowhere.
        assert!((total - 7.5).abs() < 0.15, "range EVs {:?}", res.seats.iter().map(|s| s.range_ev).collect::<Vec<_>>());
        // Position is worth something, but not the pot.
        for s in &res.seats {
            assert!(s.range_ev > 1.5 && s.range_ev < 3.5, "{}: {}", s.position, s.range_ev);
        }
        // The gap shrinks over the checks.
        let first = res.gap_history.first().unwrap().1;
        let last = res.gap_history.last().unwrap().1;
        assert!(last < first, "gap {first} -> {last}");
        // Sets of kings are worth far more than seven-deuce.
        let btn = &res.seats[2];
        let kk = btn.values[class_index(11, 11, false)];
        let seven_deuce = btn.values[class_index(5, 0, false)];
        assert!(kk > 6.0, "KK owns {kk}");
        assert!(seven_deuce < kk, "72o {seven_deuce} vs KK {kk}");
        assert!(btn.masses.iter().all(|m| *m > 0.0));
        assert_eq!(class_label(class_index(11, 11, false)), "KK");
    }

    #[test]
    fn a_folded_seat_is_dead_money_for_the_others() {
        // SB folded preflop: pot 8.0 with 0.5 nobody owns.
        let mut req = request(vec![seat("BB", uniform(1.0)), seat("CO", uniform(1.0)), seat("BTN", uniform(1.0))], 8.0);
        req.iterations = 20;
        let res = solve(&req).unwrap();
        let total: f64 = res.seats.iter().map(|s| s.range_ev).sum();
        assert!((total - 8.0).abs() < 0.15, "range EVs sum to {total}");
    }

    #[test]
    fn a_turn_street_builds_and_prices() {
        let mut req = request(vec![seat("SB", uniform(1.0)), seat("BB", uniform(1.0)), seat("BTN", uniform(1.0))], 7.5);
        req.turn_cards = 6;
        req.runouts = 12;
        req.iterations = 4;
        req.check_every = 4;
        let res = solve(&req).unwrap();
        assert_eq!(res.turn_cards.len(), 6);
        assert!(res.nodes > 100);
        let total: f64 = res.seats.iter().map(|s| s.range_ev).sum();
        assert!((total - 7.5).abs() < 0.3, "range EVs sum to {total}");
    }

    /// Three seats checking down every runout own a third of the pot each:
    /// with two other seats the deck is exact (showdown.rs `Pair`), so the
    /// shares sum to the whole pot.
    #[test]
    fn checking_down_every_runout_splits_the_pot_three_ways() {
        let mut req = request(vec![seat("SB", uniform(1.0)), seat("BB", uniform(1.0)), seat("BTN", uniform(1.0))], 7.5);
        req.turn_cards = 49;
        req.runouts = 48;
        req.iterations = 1;
        req.check_every = 1;
        req.bet_sizes = vec![];
        let res = solve(&req).unwrap();
        for s in &res.seats {
            assert!((s.range_ev - 2.5).abs() < 0.01, "{}: {}", s.position, s.range_ev);
        }
    }

    /// Heads-up there is no deck approximation, so a check-down over every
    /// runout must own exactly the pot — dead money included. Without the
    /// sample weighting (showdown.rs `sample_weight`) the opponent's hands
    /// are counted only under the runouts they leave possible, which shrinks
    /// every net result by a twelfth and loses a twelfth of the dead money.
    #[test]
    fn heads_up_check_down_owns_the_dead_money_too() {
        let mut req = request(vec![seat("BB", uniform(1.0)), seat("BTN", uniform(1.0))], 6.0);
        req.turn_cards = 49;
        req.runouts = 48;
        req.iterations = 1;
        req.check_every = 1;
        req.bet_sizes = vec![];
        let res = solve(&req).unwrap();
        let total: f64 = res.seats.iter().map(|s| s.range_ev).sum();
        assert!((total - 6.0).abs() < 0.02, "range EVs {:?}", res.seats.iter().map(|s| s.range_ev).collect::<Vec<_>>());
        for s in &res.seats {
            assert!((s.range_ev - 3.0).abs() < 0.02, "{}: {}", s.position, s.range_ev);
        }
        // Top set owns most of the pot; queen-high far less than its stake.
        let kk = res.seats[1].values[class_index(11, 11, false)];
        let queen_three = res.seats[1].values[class_index(10, 1, false)];
        assert!(kk > 5.4, "KK owns {kk}");
        assert!(queen_three < 2.4, "Q3o owns {queen_three}");
    }

    /// A hand's value is conditional on holding it: the weight the hero's
    /// own range gives it must not change it. And the rake is a percentage.
    #[test]
    fn a_hands_value_does_not_depend_on_its_own_weight_and_rake_is_a_percentage() {
        let mut owns = Vec::new();
        for w_hero in [1.0f32, 0.001] {
            let mut sb = uniform(0.3);
            sb[class_index(11, 11, false)] = w_hero;
            let mut req = request(vec![seat("SB", sb), seat("BB", uniform(0.3)), seat("BTN", uniform(0.3))], 7.5);
            req.runouts = 24;
            req.iterations = 1;
            req.check_every = 1;
            req.bet_sizes = vec![];
            owns.push(solve(&req).unwrap().seats[0].values[class_index(11, 11, false)]);
        }
        assert!((owns[0] - owns[1]).abs() < 1e-3, "{owns:?}");
        assert!(owns[0] > 6.0, "top set owns {}", owns[0]);
        let mut req = request(vec![seat("SB", uniform(0.3)), seat("BB", uniform(0.3)), seat("BTN", uniform(0.3))], 7.5);
        req.runouts = 24;
        req.iterations = 1;
        req.check_every = 1;
        req.bet_sizes = vec![];
        req.rake_pct = 5.0;
        req.rake_cap = 8.0;
        let total: f64 = solve(&req).unwrap().seats.iter().map(|s| s.range_ev).sum();
        // 5% of 7.5 leaves 7.125 (plus the deck approximation's percent or two).
        assert!((total - 7.5 * 0.95).abs() < 0.2, "raked chips owned {total}");
        req.rake_pct = 500.0;
        assert!(solve(&req).is_err());
    }

    #[test]
    fn rejects_bad_requests() {
        let mut req = request(vec![seat("SB", uniform(1.0)), seat("BB", uniform(1.0))], 5.0);
        req.board = "Ks7h".into();
        assert!(solve(&req).is_err());
        let mut req = request(vec![seat("SB", uniform(1.0)), seat("BB", vec![0.0; NUM_CLASSES])], 5.0);
        req.iterations = 1;
        assert!(solve(&req).unwrap_err().contains("empty"));
        let mut req = request(vec![seat("SB", uniform(1.0)), seat("BB", vec![1.0; 10])], 5.0);
        req.iterations = 1;
        assert!(solve(&req).unwrap_err().contains("169"));
    }
}
