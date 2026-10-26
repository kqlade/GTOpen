//! Vector-form discounted CFR over the multiway tree, with the best-response
//! gap and the root values the preflop engine consumes.

use super::showdown::{compatible_mass, compatible_pair_mass, sample_weight, showdown, Board};
use super::tree::{Action, Tree, KIND_ACTION, KIND_CHANCE, KIND_FOLD_OUT, KIND_SHOWDOWN};
use crate::cards::{combo_from_index, NUM_COMBOS};

/// DCFR(α 1.5, β 0, γ 2).
const ALPHA: f64 = 1.5;
const GAMMA: f64 = 2.0;

pub struct Solver {
    pub tree: Tree,
    /// Runout sets: index 0 for flop showdowns, 1 + i under turn card i.
    pub boards: Vec<Vec<Board>>,
    pub(super) by_card: Vec<Vec<u16>>,
    /// Each seat's reach at the root (combos the flop blocks are zero).
    pub root_reach: Vec<Vec<f32>>,
    regrets: Vec<Vec<f32>>,
    strategy_sum: Vec<Vec<f32>>,
    pub iteration: u32,
    scratch: Vec<f32>,
}

/// Which strategy a traversal plays and whether it learns.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Mode {
    /// Current strategies from regrets; the hero's regrets and strategy sums update.
    Train,
    /// Everyone plays the average strategy.
    Average,
    /// Opponents play the average strategy; the hero picks the best action per combo.
    BestResponse,
}

impl Solver {
    pub fn new(tree: Tree, boards: Vec<Vec<Board>>, root_reach: Vec<Vec<f32>>) -> Solver {
        let regrets = tree.data_size.iter().map(|&n| vec![0f32; n]).collect();
        let strategy_sum = tree.data_size.iter().map(|&n| vec![0f32; n]).collect();
        Solver {
            tree,
            boards,
            by_card: super::showdown::combos_by_card(),
            root_reach,
            regrets,
            strategy_sum,
            iteration: 0,
            scratch: Vec::new(),
        }
    }

    pub fn arena_bytes(&self) -> usize {
        self.tree.data_size.iter().sum::<usize>() * 8
    }

    /// One iteration: every seat traverses once, then the DCFR discounts.
    pub fn iterate(&mut self) {
        let seats = self.tree.seats;
        for hero in 0..seats {
            let mut reach = self.root_reach.clone();
            let mut out = vec![0f32; NUM_COMBOS];
            self.traverse(0, hero, &mut reach, &mut out, Mode::Train);
        }
        self.iteration += 1;
        let t = self.iteration as f64;
        let pos = (t.powf(ALPHA) / (t.powf(ALPHA) + 1.0)) as f32;
        let neg = 0.5f32;
        let avg = ((t / (t + 1.0)).powf(GAMMA)) as f32;
        for arena in &mut self.regrets {
            for r in arena.iter_mut() {
                *r *= if *r > 0.0 { pos } else { neg };
            }
        }
        for arena in &mut self.strategy_sum {
            for s in arena.iter_mut() {
                *s *= avg;
            }
        }
    }

    /// Mass-weighted root values of every hero combo under the average
    /// strategies (net chips from the flop on, times the other seats'
    /// compatible reach).
    pub fn root_values(&mut self, hero: usize) -> Vec<f32> {
        let mut reach = self.root_reach.clone();
        let mut out = vec![0f32; NUM_COMBOS];
        self.traverse(0, hero, &mut reach, &mut out, Mode::Average);
        out
    }

    /// Per seat: (average-strategy EV, best-response EV) in chips per hand.
    pub fn gap(&mut self) -> Vec<(f64, f64)> {
        let seats = self.tree.seats;
        let mut out = Vec::with_capacity(seats);
        for hero in 0..seats {
            let mass = self.root_mass(hero);
            let norm: f64 = (0..NUM_COMBOS).map(|h| self.root_reach[hero][h] as f64 * mass[h] as f64).sum();
            let mut reach = self.root_reach.clone();
            let mut avg = vec![0f32; NUM_COMBOS];
            self.traverse(0, hero, &mut reach, &mut avg, Mode::Average);
            let mut reach = self.root_reach.clone();
            let mut br = vec![0f32; NUM_COMBOS];
            self.traverse(0, hero, &mut reach, &mut br, Mode::BestResponse);
            let ev: f64 = (0..NUM_COMBOS).map(|h| self.root_reach[hero][h] as f64 * avg[h] as f64).sum();
            let best: f64 = (0..NUM_COMBOS).map(|h| self.root_reach[hero][h] as f64 * br[h] as f64).sum();
            out.push(if norm > 0.0 { (ev / norm, best / norm) } else { (0.0, 0.0) });
        }
        out
    }

    /// The other seats' compatible reach mass at the root, per hero combo.
    pub fn root_mass(&self, hero: usize) -> Vec<f32> {
        let mut mass = vec![0f64; NUM_COMBOS];
        self.others_mass(hero, &self.root_reach, &mut mass);
        mass.iter().map(|m| *m as f32).collect()
    }

    /// `out[h]` = mass of the other seats' hands compatible with hero combo
    /// `h` and (with two other seats) with each other; more seats are taken
    /// as independent of each other.
    pub(super) fn others_mass(&self, hero: usize, reach: &[Vec<f32>], out: &mut [f64]) {
        let others: Vec<usize> = (0..self.tree.seats).filter(|&q| q != hero).collect();
        if let [q1, q2] = others[..] {
            compatible_pair_mass(&reach[q1], &reach[q2], &self.by_card, out);
            return;
        }
        out.iter_mut().for_each(|v| *v = 1.0);
        let mut one = vec![0f32; NUM_COMBOS];
        for q in others {
            compatible_mass(&reach[q], &self.by_card, &mut one);
            for h in 0..NUM_COMBOS {
                out[h] *= one[h] as f64;
            }
        }
    }

    /// Average strategy at an action node: `[action][combo]`.
    pub fn average_strategy(&self, node: usize) -> Vec<f32> {
        let nd = &self.tree.nodes[node];
        let na = nd.actions.len();
        let seat = nd.seat as usize;
        let sums = &self.strategy_sum[seat][nd.data_offset..nd.data_offset + na * NUM_COMBOS];
        let mut out = vec![0f32; na * NUM_COMBOS];
        for h in 0..NUM_COMBOS {
            let total: f32 = (0..na).map(|i| sums[i * NUM_COMBOS + h]).sum();
            for i in 0..na {
                out[i * NUM_COMBOS + h] = if total > 0.0 { sums[i * NUM_COMBOS + h] / total } else { 1.0 / na as f32 };
            }
        }
        out
    }

    fn current_strategy(regrets: &[f32], na: usize, out: &mut [f32]) {
        for h in 0..NUM_COMBOS {
            let mut total = 0f32;
            for i in 0..na {
                total += regrets[i * NUM_COMBOS + h].max(0.0);
            }
            for i in 0..na {
                out[i * NUM_COMBOS + h] = if total > 0.0 {
                    regrets[i * NUM_COMBOS + h].max(0.0) / total
                } else {
                    1.0 / na as f32
                };
            }
        }
    }

    pub(super) fn traverse(&mut self, node: usize, hero: usize, reach: &mut Vec<Vec<f32>>, out: &mut [f32], mode: Mode) {
        let nd = &self.tree.nodes[node];
        let seats = self.tree.seats;
        let (kind, actor, live, put_hero, offset, na) =
            (nd.kind, nd.seat as usize, nd.live, nd.put[hero], nd.data_offset, nd.actions.len());
        let children = nd.children.clone();
        // The hero folded earlier: it loses what it put in whatever happens
        // next, weighted by the others' compatible mass here (their later
        // actions sum to one).
        if kind != KIND_FOLD_OUT && live & (1 << hero) == 0 {
            self.fold_payoff(hero, reach, -put_hero, out);
            return;
        }
        match kind {
            KIND_FOLD_OUT => {
                let delta = if actor == hero { self.tree.fold_out_winnings(nd) - put_hero } else { -put_hero };
                self.fold_payoff(hero, reach, delta, out);
            }
            KIND_SHOWDOWN => {
                let pot = self.tree.pot_at(nd);
                let pot_eff = pot - self.tree.rake(pot);
                let set = nd.runout_set as usize;
                // set 0 runs the flop out (two cards); the others run a turn card out (one)
                let drawn = if set == 0 { 2 } else { 1 };
                let live_opps: Vec<usize> = (0..seats).filter(|&q| q != hero && live & (1 << q) != 0).collect();
                let folded: Vec<usize> = (0..seats).filter(|&q| q != hero && live & (1 << q) == 0).collect();
                let boards = std::mem::take(&mut self.boards[set]);
                let mut scratch = std::mem::take(&mut self.scratch);
                showdown(&boards, drawn, hero, &live_opps, &folded, reach, pot_eff, put_hero, out, &mut scratch);
                self.boards[set] = boards;
                self.scratch = scratch;
            }
            KIND_CHANCE => {
                // the turn card comes from the deck every seat's hand leaves (showdown.rs `sample_weight`)
                let weight = sample_weight(47, seats - 1, 1) as f32;
                let mut norm = vec![0f32; NUM_COMBOS];
                let mut child_out = vec![0f32; NUM_COMBOS];
                for (i, &child) in children.iter().enumerate() {
                    let card = self.tree.turn_cards[i];
                    let w = self.tree.turn_weights[i];
                    let mut masked = reach.clone();
                    for r in masked.iter_mut() {
                        for &idx in &self.by_card[card as usize] {
                            r[idx as usize] = 0.0;
                        }
                    }
                    child_out.iter_mut().for_each(|v| *v = 0.0);
                    self.traverse(child as usize, hero, &mut masked, &mut child_out, mode);
                    for h in 0..NUM_COMBOS {
                        let (a, b) = combo_from_index(h);
                        if a != card && b != card {
                            out[h] += w * child_out[h];
                            norm[h] += w;
                        }
                    }
                }
                for h in 0..NUM_COMBOS {
                    if norm[h] > 0.0 {
                        out[h] *= weight / norm[h];
                    }
                }
            }
            KIND_ACTION => {
                let sigma = match mode {
                    Mode::Train => {
                        let mut sigma = vec![0f32; na * NUM_COMBOS];
                        Self::current_strategy(&self.regrets[actor][offset..offset + na * NUM_COMBOS], na, &mut sigma);
                        sigma
                    }
                    Mode::Average | Mode::BestResponse => self.average_strategy(node),
                };
                let saved = reach[actor].clone();
                let mut values = vec![0f32; na * NUM_COMBOS];
                for i in 0..na {
                    for h in 0..NUM_COMBOS {
                        reach[actor][h] = saved[h] * sigma[i * NUM_COMBOS + h];
                    }
                    let mut child_out = vec![0f32; NUM_COMBOS];
                    self.traverse(children[i] as usize, hero, reach, &mut child_out, mode);
                    values[i * NUM_COMBOS..(i + 1) * NUM_COMBOS].copy_from_slice(&child_out);
                }
                reach[actor].copy_from_slice(&saved);
                if actor == hero {
                    if mode == Mode::BestResponse {
                        for h in 0..NUM_COMBOS {
                            let mut best = f32::NEG_INFINITY;
                            for i in 0..na {
                                best = best.max(values[i * NUM_COMBOS + h]);
                            }
                            out[h] = best;
                        }
                        return;
                    }
                    for h in 0..NUM_COMBOS {
                        let mut v = 0f32;
                        for i in 0..na {
                            v += sigma[i * NUM_COMBOS + h] * values[i * NUM_COMBOS + h];
                        }
                        out[h] = v;
                    }
                    if mode == Mode::Train {
                        let regrets = &mut self.regrets[actor][offset..offset + na * NUM_COMBOS];
                        let sums = &mut self.strategy_sum[actor][offset..offset + na * NUM_COMBOS];
                        for i in 0..na {
                            for h in 0..NUM_COMBOS {
                                regrets[i * NUM_COMBOS + h] += values[i * NUM_COMBOS + h] - out[h];
                                sums[i * NUM_COMBOS + h] += saved[h] * sigma[i * NUM_COMBOS + h];
                            }
                        }
                    }
                } else {
                    for h in 0..NUM_COMBOS {
                        let mut v = 0f32;
                        for i in 0..na {
                            v += values[i * NUM_COMBOS + h];
                        }
                        out[h] = v;
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    /// `out[h] = delta × mass of the other seats' hands compatible with h`.
    fn fold_payoff(&self, hero: usize, reach: &[Vec<f32>], delta: f64, out: &mut [f32]) {
        let mut mass = vec![0f64; NUM_COMBOS];
        self.others_mass(hero, reach, &mut mass);
        for h in 0..NUM_COMBOS {
            out[h] = (mass[h] * delta) as f32;
        }
    }

    /// Label of an action for reports.
    pub fn label(action: &Action, pot: f64) -> String {
        match action {
            Action::Fold => "fold".into(),
            Action::Check => "check".into(),
            Action::Call(_) => "call".into(),
            Action::Bet(to) => format!("bet {:.1}% ({to:.2})", to / pot * 100.0),
            Action::Raise(to) => format!("raise to {to:.2}"),
        }
    }
}
