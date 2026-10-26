//! Reading a solved multiway tree back node by node, the way the heads-up
//! solver's `node_view` does: every seat's hands with their reach, and for
//! the seat to act its strategy and the EV of each action, per combo.
//!
//! Values follow the convention of `solve()`'s root values: a hand's EV is
//! the chips it owns at the end of the hand, averaged over the other seats'
//! compatible hands (`valid`, the normaliser) and offset by what the seat
//! has put in at the node, so `ev == Σ strategy × evs` at every action node
//! and range averages must weight by `reach × valid`.

use super::solve::{Mode, Solver};
use super::tree::{Action, KIND_ACTION, KIND_CHANCE, KIND_FOLD_OUT, KIND_SHOWDOWN};
use crate::cards::{card_to_string, combo_from_index, combo_to_string, parse_cards, Card, NUM_COMBOS};
use crate::query::PathStep;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ActionView {
    pub label: String,
    pub kind: String,
    /// Street total the action is to (0 for check and fold).
    pub amount: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HandView {
    pub combo: String,
    /// Range weight at the root.
    pub weight: f32,
    /// Reach: range weight × the strategy along the path.
    pub reach: f32,
    /// The other seats' compatible reach mass here (the EV normaliser).
    pub valid: f32,
    /// The multiway solver does not price equity per hand: always null.
    pub eq: Option<f32>,
    /// Chips owned at the end of the hand; null when no other hand is compatible.
    pub ev: Option<f32>,
    /// Seat to act only: average strategy per action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<Vec<f32>>,
    /// Seat to act only: EV of each action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evs: Option<Vec<Option<f32>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlayerView {
    /// Empty for a seat that has folded.
    pub hands: Vec<HandView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeView {
    /// "action", "chance", "fold" (one seat left) or "showdown".
    pub node_type: String,
    /// 0 flop, 1 turn.
    pub street: u8,
    pub board: Vec<String>,
    pub pot: f64,
    /// Every seat's contribution so far, preflop included, in acting order.
    pub put: Vec<f64>,
    pub player: Option<u8>,
    pub actions: Vec<ActionView>,
    /// Chance node: the sampled turn cards that can come.
    pub available_cards: Option<Vec<String>>,
    pub players: Vec<PlayerView>,
    pub locked: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SolvedNode {
    pub path: Vec<PathStep>,
    pub node: NodeView,
}

/// Where a path leads: the node, every seat's reach there, the cards dealt
/// on the way and how often the line is taken (the actors' range-weighted
/// action frequencies multiplied).
pub struct Walk {
    pub node: usize,
    pub reach: Vec<Vec<f32>>,
    pub dealt: Vec<Card>,
    pub frequency: f64,
}

impl Solver {
    pub fn walk(&self, path: &[PathStep]) -> Result<Walk, String> {
        let mut node = 0usize;
        let mut reach = self.root_reach.clone();
        let mut dealt = Vec::new();
        let mut frequency = 1.0f64;
        for (at, step) in path.iter().enumerate() {
            let nd = &self.tree.nodes[node];
            match (nd.kind, step) {
                (KIND_ACTION, PathStep::Action { index }) => {
                    if *index >= nd.actions.len() {
                        return Err(format!("step {at}: action {index} of {} at this node", nd.actions.len()));
                    }
                    let sigma = self.average_strategy(node);
                    let actor = nd.seat as usize;
                    let (mut total, mut taken) = (0f64, 0f64);
                    for h in 0..NUM_COMBOS {
                        total += reach[actor][h] as f64;
                        taken += (reach[actor][h] * sigma[index * NUM_COMBOS + h]) as f64;
                        reach[actor][h] *= sigma[index * NUM_COMBOS + h];
                    }
                    frequency *= if total > 0.0 { taken / total } else { 0.0 };
                    node = nd.children[*index] as usize;
                }
                (KIND_CHANCE, PathStep::Card { card }) => {
                    let cards = parse_cards(card)?;
                    let [c] = cards[..] else {
                        return Err(format!("step {at}: {card:?} is not one card"));
                    };
                    let Some(i) = self.tree.turn_cards.iter().position(|&t| t == c) else {
                        return Err(format!(
                            "step {at}: the turn {card} is not among the sampled turn cards ({}); ask for it in turn_cards_fixed",
                            self.tree.turn_cards.iter().map(|&t| card_to_string(t)).collect::<Vec<_>>().join(" ")
                        ));
                    };
                    for r in reach.iter_mut() {
                        for &idx in &self.by_card[c as usize] {
                            r[idx as usize] = 0.0;
                        }
                    }
                    dealt.push(c);
                    node = nd.children[i] as usize;
                }
                (KIND_ACTION, PathStep::Card { .. }) => {
                    return Err(format!("step {at}: a card where an action is due"));
                }
                (KIND_CHANCE, PathStep::Action { .. }) => {
                    return Err(format!("step {at}: an action where the turn is dealt"));
                }
                _ => return Err(format!("step {at}: the hand is over here")),
            }
        }
        Ok(Walk { node, reach, dealt, frequency })
    }

    /// The view of the node a path leads to; hands reaching it with less
    /// than `min_reach` are left out.
    pub fn node_view(&mut self, flop: &[Card], path: &[PathStep], min_reach: f32) -> Result<NodeView, String> {
        let walk = self.walk(path)?;
        let nd = self.tree.nodes[walk.node].clone();
        let seats = self.tree.seats;
        let pot = self.tree.pot_at(&nd);
        let actor = (nd.kind == KIND_ACTION).then_some(nd.seat as usize);
        let actions: Vec<ActionView> = nd
            .actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let me = nd.seat as usize;
                let child = &self.tree.nodes[nd.children[i] as usize];
                let behind_after = self.tree.cfg.stack - (child.put[me] - self.tree.cfg.invested[me]);
                let all_in = a.is_aggressive() && behind_after.abs() < 1e-6;
                action_view(a, pot, all_in)
            })
            .collect();
        let sigma = actor.map(|_| self.average_strategy(walk.node));
        let mut players = Vec::with_capacity(seats);
        for p in 0..seats {
            if nd.live & (1 << p) == 0 {
                players.push(PlayerView { hands: Vec::new() });
                continue;
            }
            let mut mass = vec![0f64; NUM_COMBOS];
            self.others_mass(p, &walk.reach, &mut mass);
            // The seat to act: one traversal per action, its own value being
            // their strategy-weighted sum. Anyone else: one traversal from here.
            let per_action: Option<Vec<Vec<f32>>> = (actor == Some(p)).then(|| {
                nd.children
                    .iter()
                    .map(|&child| {
                        let mut reach = walk.reach.clone();
                        let mut out = vec![0f32; NUM_COMBOS];
                        self.traverse(child as usize, p, &mut reach, &mut out, Mode::Average);
                        out
                    })
                    .collect()
            });
            let mut out = vec![0f32; NUM_COMBOS];
            match (&per_action, &sigma) {
                (Some(values), Some(s)) => {
                    for (i, v) in values.iter().enumerate() {
                        for h in 0..NUM_COMBOS {
                            out[h] += s[i * NUM_COMBOS + h] * v[h];
                        }
                    }
                }
                _ => {
                    let mut reach = walk.reach.clone();
                    self.traverse(walk.node, p, &mut reach, &mut out, Mode::Average);
                }
            }
            let put = nd.put[p];
            let value = |v: f32, m: f64| if m > 1e-9 { Some((v as f64 / m + put) as f32) } else { None };
            let mut hands = Vec::new();
            for h in 0..NUM_COMBOS {
                let r = walk.reach[p][h];
                if r <= 0.0 || r < min_reach {
                    continue;
                }
                let (a, b) = combo_from_index(h);
                hands.push(HandView {
                    combo: combo_to_string(a, b),
                    weight: self.root_reach[p][h],
                    reach: r,
                    valid: mass[h] as f32,
                    eq: None,
                    ev: value(out[h], mass[h]),
                    strategy: sigma
                        .as_ref()
                        .filter(|_| actor == Some(p))
                        .map(|s| (0..nd.actions.len()).map(|i| s[i * NUM_COMBOS + h]).collect()),
                    evs: per_action.as_ref().map(|vals| vals.iter().map(|v| value(v[h], mass[h])).collect()),
                });
            }
            players.push(PlayerView { hands });
        }
        Ok(NodeView {
            node_type: match nd.kind {
                KIND_ACTION => "action",
                KIND_CHANCE => "chance",
                KIND_FOLD_OUT => "fold",
                KIND_SHOWDOWN => "showdown",
                _ => unreachable!(),
            }
            .into(),
            street: nd.street,
            board: flop.iter().chain(&walk.dealt).map(|&c| card_to_string(c)).collect(),
            pot,
            put: nd.put[..seats].to_vec(),
            player: actor.map(|p| p as u8),
            actions,
            available_cards: (nd.kind == KIND_CHANCE)
                .then(|| self.tree.turn_cards.iter().map(|&c| card_to_string(c)).collect()),
            players,
            locked: false,
        })
    }
}

/// Labels like the heads-up tree's: "Check", "Bet 2.48 (33%)", "Raise 9.95", "All-in 97.5".
fn action_view(action: &Action, pot: f64, all_in: bool) -> ActionView {
    let trim = |x: f64| {
        let s = format!("{x:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    let (label, kind, amount) = match action {
        Action::Fold => ("Fold".to_string(), "fold", 0.0),
        Action::Check => ("Check".to_string(), "check", 0.0),
        Action::Call(to) => ("Call".to_string(), "call", *to),
        Action::Bet(to) => (
            if all_in {
                format!("All-in {}", trim(*to))
            } else {
                format!("Bet {} ({}%)", trim(*to), (to / pot * 100.0).round())
            },
            "bet",
            *to,
        ),
        Action::Raise(to) => (
            if all_in { format!("All-in {}", trim(*to)) } else { format!("Raise {}", trim(*to)) },
            "raise",
            *to,
        ),
    };
    ActionView { label, kind: kind.into(), amount }
}
