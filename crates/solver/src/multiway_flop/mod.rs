//! Flop solver for three or more seats, built to measure the preflop
//! engine's multiway terminals (see `docs/multiway_flop_solver.md`).
//!
//! One street from the flop (or the flop and a sampled turn), every seat's
//! strategy a vector over the 1,326 combos, leaves checked down by exact
//! showdown over a fixed sample of runouts. CPU only, one solve per call.

pub mod showdown;
pub mod solve;
pub mod tree;

use crate::cards::{cards_to_string, combo_from_index, parse_cards, rank, suit, Card, NUM_COMBOS};
use crate::preflop::equity::{class_index, NUM_CLASSES};
use serde::{Deserialize, Serialize};
use showdown::{flop_runouts, sampled_cards, Board};
use solve::Solver;
use tree::{Builder, TreeConfig};

#[derive(Debug, Clone, Deserialize)]
pub struct SeatInput {
    pub position: String,
    /// Chips this seat put in before the flop.
    pub invested: f64,
    /// Per hand class (engine order): the probability each combo of the
    /// class takes the line to this flop, in [0, 1].
    pub weights: Vec<f32>,
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
    #[serde(default = "default_max_raises")]
    pub max_raises: u8,
    #[serde(default = "default_allin_threshold")]
    pub allin_threshold: f64,
    /// Sampled turn cards to deal (0: the tree ends on the flop).
    #[serde(default)]
    pub turn_cards: usize,
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
}

/// Root reach of a seat: its class weights spread over the combos, minus the
/// ones the board blocks.
pub fn root_reach(weights: &[f32], board: &[Card]) -> Vec<f32> {
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
        *slot = weights[class_index(rank(a), rank(b), suit(a) == suit(b))].clamp(0.0, 1.0);
    }
    out
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
    for seat in &req.seats {
        if seat.weights.len() != NUM_CLASSES {
            return Err(format!("{}: weights must have {NUM_CLASSES} entries", seat.position));
        }
        if seat.weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(format!("{}: weights must be finite and not negative", seat.position));
        }
        if seat.weights.iter().all(|w| *w <= 0.0) {
            return Err(format!("{}: the range is empty", seat.position));
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
        max_raises: req.max_raises,
        allin_threshold: req.allin_threshold,
        turn_cards: req.turn_cards,
    };
    let turn_cards = if req.turn_cards > 0 { sampled_cards(&flop, req.turn_cards) } else { Vec::new() };
    let max_nodes = (req.max_arena_mb * 1e6 / (8.0 * NUM_COMBOS as f64)) as usize;
    let tree = Builder::build(&cfg, turn_cards.clone(), NUM_COMBOS, max_nodes.max(1000))?;
    let arena_mb = tree.data_size.iter().sum::<usize>() as f64 * 8.0 / 1e6;
    if arena_mb > req.max_arena_mb {
        return Err(format!(
            "tree too large: {arena_mb:.0} MB of arenas over the {:.0} MB limit; fewer sizes, raises or turn cards",
            req.max_arena_mb
        ));
    }
    let mut boards: Vec<Vec<Board>> = Vec::with_capacity(1 + turn_cards.len());
    boards.push(flop_runouts(&flop, req.runouts).into_iter().map(Board::new).collect());
    for &t in &turn_cards {
        let four = [flop[0], flop[1], flop[2], t];
        boards.push(
            sampled_cards(&four, req.runouts)
                .into_iter()
                .map(|r| Board::new([flop[0], flop[1], flop[2], t, r]))
                .collect(),
        );
    }
    let reach: Vec<Vec<f32>> = req.seats.iter().map(|s| root_reach(&s.weights, &flop)).collect();
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
    })
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
            max_raises: 1,
            allin_threshold: 0.65,
            turn_cards: 0,
            runouts: 12,
            iterations: 40,
            check_every: 20,
            target_gap_pct: 0.0,
            max_arena_mb: 1500.0,
        }
    }

    fn seat(position: &str, weights: Vec<f32>) -> SeatInput {
        SeatInput { position: position.into(), invested: 2.5, weights }
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

    /// Three seats checking down every runout own a third of the pot each,
    /// up to the deck approximation (opponents independent of each other):
    /// the overlapping opponent hands it counts are correlated, which lifts
    /// every seat's share by about one percent of the pot.
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
            assert!((s.range_ev - 2.5).abs() < 0.04, "{}: {}", s.position, s.range_ev);
        }
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
