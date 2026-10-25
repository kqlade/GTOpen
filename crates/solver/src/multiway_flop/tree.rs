//! Betting tree for three or more seats from the flop: one street, or the
//! flop and a turn dealt on sampled cards. Everyone live matched the last
//! preflop bet, so all seats have the same stack behind and no side pots
//! arise: a seat that is all-in has every other live seat all-in too once
//! they call.

use serde::{Deserialize, Serialize};

pub const MAX_SEATS: usize = 6;

pub const KIND_ACTION: u8 = 0;
/// The turn is dealt: one child per sampled card.
pub const KIND_CHANCE: u8 = 1;
/// One seat left; it takes the pot.
pub const KIND_FOLD_OUT: u8 = 2;
/// Two or more seats live with no betting left to model: checked down.
pub const KIND_SHOWDOWN: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "kind", content = "to")]
pub enum Action {
    Fold,
    Check,
    /// Call up to this street total.
    Call(f64),
    /// Bet to this street total.
    Bet(f64),
    /// Raise to this street total.
    Raise(f64),
}

impl Action {
    pub fn is_aggressive(&self) -> bool {
        matches!(self, Action::Bet(_) | Action::Raise(_))
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: u8,
    /// Action node: the seat to act. Fold-out: the winner.
    pub seat: u8,
    /// 0 flop, 1 turn.
    pub street: u8,
    /// Bitmask of the seats still in the hand.
    pub live: u8,
    /// Every seat's total contribution (preflop included) at this node.
    pub put: [f64; MAX_SEATS],
    /// Action node: one child per action. Chance node: one per sampled turn
    /// card, in the order of `Tree::turn_cards`.
    pub children: Vec<u32>,
    pub actions: Vec<Action>,
    /// Action node: offset (in f32 elements) into the actor's arenas.
    pub data_offset: usize,
    /// Showdown: which runout set prices it (0 = the flop sample, 1 + i =
    /// the rivers sampled after turn card i).
    pub runout_set: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TreeConfig {
    /// Chips in the pot when the flop is dealt: the live seats' investments
    /// plus whatever the folded seats left behind.
    pub pot: f64,
    /// Chips every live seat has behind.
    pub stack: f64,
    /// Each seat's preflop contribution, in postflop acting order.
    pub invested: Vec<f64>,
    #[serde(default)]
    pub rake_pct: f64,
    #[serde(default)]
    pub rake_cap: f64,
    /// Bet sizes as fractions of the pot.
    pub bet_sizes: Vec<f64>,
    /// Raise sizes as multiples of the bet faced (street total).
    pub raise_sizes: Vec<f64>,
    #[serde(default = "default_max_raises")]
    pub max_raises: u8,
    /// A bet or raise committing more than this fraction of the stack
    /// becomes all-in.
    #[serde(default = "default_allin_threshold")]
    pub allin_threshold: f64,
    /// How many turn cards to deal (0: the tree stops at the flop).
    #[serde(default)]
    pub turn_cards: usize,
}

fn default_max_raises() -> u8 {
    2
}
fn default_allin_threshold() -> f64 {
    0.65
}

pub struct Tree {
    pub cfg: TreeConfig,
    pub seats: usize,
    /// Chips in the pot that belong to no live seat (folded seats' money).
    pub dead: f64,
    pub nodes: Vec<Node>,
    /// f32 elements each seat's regret and strategy arenas need.
    pub data_size: Vec<usize>,
    /// Sampled turn cards (empty for a one-street tree).
    pub turn_cards: Vec<u8>,
}

impl Tree {
    pub fn rake(&self, pot: f64) -> f64 {
        let r = self.cfg.rake_pct * pot;
        if self.cfg.rake_cap > 0.0 {
            r.min(self.cfg.rake_cap)
        } else {
            r
        }
    }

    /// Chips in the middle at this node.
    pub fn pot_at(&self, node: &Node) -> f64 {
        self.dead + node.put[..self.seats].iter().sum::<f64>()
    }

    /// Fold-out: the raked chips the winner owns at the end. Only the matched
    /// part of the pot is raked; the uncalled bet goes back unraked.
    pub fn fold_out_winnings(&self, node: &Node) -> f64 {
        let w = node.seat as usize;
        let matched_by_others = (0..self.seats)
            .filter(|&q| q != w)
            .map(|q| node.put[q])
            .fold(0.0, f64::max);
        let uncalled = (node.put[w] - matched_by_others).max(0.0);
        let pot = self.pot_at(node);
        pot - self.rake(pot - uncalled)
    }

    pub fn action_nodes(&self) -> usize {
        self.nodes.iter().filter(|n| n.kind == KIND_ACTION).count()
    }
}

#[derive(Clone)]
struct State {
    street: u8,
    live: u8,
    /// Live seats that still have to act this street.
    pending: u8,
    to_act: usize,
    put: [f64; MAX_SEATS],
    street_bet: [f64; MAX_SEATS],
    current_bet: f64,
    last_increment: f64,
    raises: u8,
}

pub struct Builder<'a> {
    cfg: &'a TreeConfig,
    seats: usize,
    dead: f64,
    num_hands: usize,
    nodes: Vec<Node>,
    data_size: Vec<usize>,
    turn_cards: Vec<u8>,
    max_nodes: usize,
}

impl<'a> Builder<'a> {
    /// `turn_cards` are the sampled turn cards (ignored when the config has
    /// `turn_cards == 0`); `num_hands` sizes the arenas.
    pub fn build(
        cfg: &'a TreeConfig,
        turn_cards: Vec<u8>,
        num_hands: usize,
        max_nodes: usize,
    ) -> Result<Tree, String> {
        let seats = cfg.invested.len();
        if !(2..=MAX_SEATS).contains(&seats) {
            return Err(format!("the tree needs 2 to {MAX_SEATS} seats, got {seats}"));
        }
        if !(cfg.pot > 0.0 && cfg.pot.is_finite()) || !(cfg.stack > 0.0 && cfg.stack.is_finite()) {
            return Err("pot and stack must be positive finite numbers".into());
        }
        if cfg.invested.iter().any(|v| !(*v >= 0.0 && v.is_finite())) {
            return Err("invested amounts must be finite and not negative".into());
        }
        let posted: f64 = cfg.invested.iter().sum();
        let dead = cfg.pot - posted;
        if dead < -1e-6 * cfg.pot.max(1.0) {
            return Err(format!(
                "the pot ({}) is smaller than the live seats' investments ({posted})",
                cfg.pot
            ));
        }
        let dead = dead.max(0.0);
        if cfg.bet_sizes.iter().chain(&cfg.raise_sizes).any(|s| !(*s > 0.0 && s.is_finite())) {
            return Err("bet and raise sizes must be positive finite numbers".into());
        }
        if !(cfg.allin_threshold > 0.0 && cfg.allin_threshold <= 1.0) {
            return Err("allin_threshold must be in (0, 1]".into());
        }
        if cfg.turn_cards > 0 && turn_cards.is_empty() {
            return Err("turn_cards > 0 needs sampled turn cards".into());
        }
        let turn_cards = if cfg.turn_cards > 0 { turn_cards } else { Vec::new() };
        let mut b = Builder {
            cfg,
            seats,
            dead,
            num_hands,
            nodes: Vec::new(),
            data_size: vec![0; seats],
            turn_cards,
            max_nodes,
        };
        let mut put = [0f64; MAX_SEATS];
        put[..seats].copy_from_slice(&cfg.invested);
        let live = ((1u16 << seats) - 1) as u8;
        let st = State {
            street: 0,
            live,
            pending: live,
            to_act: 0,
            put,
            street_bet: [0.0; MAX_SEATS],
            current_bet: 0.0,
            last_increment: 0.0,
            raises: 0,
        };
        b.action(st)?;
        let mut tree = Tree {
            cfg: cfg.clone(),
            seats,
            dead,
            nodes: b.nodes,
            data_size: b.data_size,
            turn_cards: b.turn_cards,
        };
        assign_runout_sets(&mut tree);
        Ok(tree)
    }

    fn push(&mut self, node: Node) -> Result<u32, String> {
        if self.nodes.len() >= self.max_nodes {
            return Err(format!(
                "tree too large: over {} nodes; fewer sizes, raises or turn cards",
                self.max_nodes
            ));
        }
        self.nodes.push(node);
        Ok((self.nodes.len() - 1) as u32)
    }

    fn pot(&self, st: &State) -> f64 {
        self.dead + st.put[..self.seats].iter().sum::<f64>()
    }

    fn behind(&self, st: &State, seat: usize) -> f64 {
        self.cfg.stack - (st.put[seat] - self.cfg.invested[seat])
    }

    fn legal(&self, st: &State) -> Vec<Action> {
        let me = st.to_act;
        let behind = self.behind(st, me);
        let facing = st.current_bet - st.street_bet[me];
        let mut out = Vec::new();
        if facing > 1e-9 {
            out.push(Action::Fold);
            out.push(Action::Call(st.current_bet.min(st.street_bet[me] + behind)));
            if behind > facing + 1e-9 && st.raises < self.cfg.max_raises {
                let max_to = st.street_bet[me] + behind;
                let min_to = st.current_bet + st.last_increment.max(1e-9);
                let mut tos: Vec<f64> = Vec::new();
                for m in &self.cfg.raise_sizes {
                    let mut to = st.current_bet * m;
                    if to < min_to {
                        to = min_to;
                    }
                    let committed = st.put[me] - st.street_bet[me] + to - self.cfg.invested[me];
                    if to >= max_to - 1e-9 || committed >= self.cfg.allin_threshold * self.cfg.stack - 1e-9 {
                        to = max_to;
                    }
                    tos.push(to);
                }
                dedupe(&mut tos);
                out.extend(tos.into_iter().map(Action::Raise));
            }
        } else {
            out.push(Action::Check);
            if behind > 1e-9 {
                let pot = self.pot(st);
                let max_to = st.street_bet[me] + behind;
                let mut tos: Vec<f64> = Vec::new();
                for f in &self.cfg.bet_sizes {
                    let mut to = st.street_bet[me] + f * pot;
                    let committed = st.put[me] - st.street_bet[me] + to - self.cfg.invested[me];
                    if to >= max_to - 1e-9 || committed >= self.cfg.allin_threshold * self.cfg.stack - 1e-9 {
                        to = max_to;
                    }
                    tos.push(to);
                }
                dedupe(&mut tos);
                out.extend(tos.into_iter().map(Action::Bet));
            }
        }
        out
    }

    fn action(&mut self, st: State) -> Result<u32, String> {
        let actions = self.legal(&st);
        let me = st.to_act;
        let data_offset = self.data_size[me];
        self.data_size[me] += actions.len() * self.num_hands;
        let idx = self.push(Node {
            kind: KIND_ACTION,
            seat: me as u8,
            street: st.street,
            live: st.live,
            put: st.put,
            children: Vec::new(),
            actions: actions.clone(),
            data_offset,
            runout_set: 0,
        })?;
        let mut children = Vec::with_capacity(actions.len());
        for action in &actions {
            children.push(self.apply(&st, *action)?);
        }
        self.nodes[idx as usize].children = children;
        Ok(idx)
    }

    fn apply(&mut self, st: &State, action: Action) -> Result<u32, String> {
        let me = st.to_act;
        let mut next = st.clone();
        next.pending &= !(1 << me);
        match action {
            Action::Fold => {
                next.live &= !(1 << me);
            }
            Action::Check => {}
            Action::Call(to) => {
                next.put[me] += to - st.street_bet[me];
                next.street_bet[me] = to;
            }
            Action::Bet(to) | Action::Raise(to) => {
                next.put[me] += to - st.street_bet[me];
                next.street_bet[me] = to;
                next.last_increment = to - st.current_bet;
                next.current_bet = to;
                if matches!(action, Action::Raise(_)) {
                    next.raises += 1;
                }
                // Everyone else live and not all-in has to act again.
                next.pending = 0;
                for q in 0..self.seats {
                    if q != me && next.live & (1 << q) != 0 && self.behind(&next, q) > 1e-9 {
                        next.pending |= 1 << q;
                    }
                }
            }
        }
        if next.live.count_ones() == 1 {
            let winner = next.live.trailing_zeros() as u8;
            return self.push(Node {
                kind: KIND_FOLD_OUT,
                seat: winner,
                street: st.street,
                live: next.live,
                put: next.put,
                children: Vec::new(),
                actions: Vec::new(),
                data_offset: 0,
                runout_set: 0,
            });
        }
        // Seats that cannot act any more (all-in) leave the pending set.
        for q in 0..self.seats {
            if next.pending & (1 << q) != 0 && self.behind(&next, q) <= 1e-9 {
                next.pending &= !(1 << q);
            }
        }
        if next.pending != 0 {
            // Next live seat after me that still has to act.
            let mut q = (me + 1) % self.seats;
            while next.pending & (1 << q) == 0 {
                q = (q + 1) % self.seats;
            }
            next.to_act = q;
            return self.action(next);
        }
        // The street closed. Chips behind for two or more seats and a turn to
        // deal: chance node; otherwise checked down.
        let can_bet = (0..self.seats)
            .filter(|&q| next.live & (1 << q) != 0 && self.behind(&next, q) > 1e-9)
            .count();
        if st.street == 0 && !self.turn_cards.is_empty() && can_bet >= 2 {
            let idx = self.push(Node {
                kind: KIND_CHANCE,
                seat: 0,
                street: 1,
                live: next.live,
                put: next.put,
                children: Vec::new(),
                actions: Vec::new(),
                data_offset: 0,
                runout_set: 0,
            })?;
            let mut children = Vec::with_capacity(self.turn_cards.len());
            for _ in 0..self.turn_cards.len() {
                let mut turn = next.clone();
                turn.street = 1;
                turn.street_bet = [0.0; MAX_SEATS];
                turn.current_bet = 0.0;
                turn.last_increment = 0.0;
                turn.raises = 0;
                turn.pending = turn.live;
                turn.to_act = turn.live.trailing_zeros() as usize;
                children.push(self.action(turn)?);
            }
            self.nodes[idx as usize].children = children;
            return Ok(idx);
        }
        self.push(Node {
            kind: KIND_SHOWDOWN,
            seat: 0,
            street: st.street,
            live: next.live,
            put: next.put,
            children: Vec::new(),
            actions: Vec::new(),
            data_offset: 0,
            runout_set: 0,
        })
    }
}

fn dedupe(tos: &mut Vec<f64>) {
    tos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tos.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
}

/// Label the runout set of every showdown under each sampled turn card.
fn assign_runout_sets(tree: &mut Tree) {
    fn mark(tree: &mut Tree, node: usize, set: u32) {
        let kids = tree.nodes[node].children.clone();
        match tree.nodes[node].kind {
            KIND_SHOWDOWN => tree.nodes[node].runout_set = set,
            KIND_CHANCE => {
                for (i, k) in kids.iter().enumerate() {
                    mark(tree, *k as usize, 1 + i as u32);
                }
            }
            _ => {
                for k in kids {
                    mark(tree, k as usize, set);
                }
            }
        }
    }
    mark(tree, 0, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seats: usize) -> TreeConfig {
        TreeConfig {
            pot: 7.5,
            stack: 97.5,
            invested: vec![2.5; seats],
            rake_pct: 0.05,
            rake_cap: 8.0,
            bet_sizes: vec![0.33, 0.75],
            raise_sizes: vec![2.5],
            max_raises: 2,
            allin_threshold: 0.65,
            turn_cards: 0,
        }
    }

    #[test]
    fn three_seats_check_around_to_a_showdown() {
        let tree = Builder::build(&cfg(3), Vec::new(), 1326, 1_000_000).unwrap();
        let root = &tree.nodes[0];
        assert_eq!(root.kind, KIND_ACTION);
        assert_eq!(root.seat, 0);
        assert_eq!(root.actions[0], Action::Check);
        let after_check = &tree.nodes[root.children[0] as usize];
        assert_eq!(after_check.seat, 1);
        let after_two = &tree.nodes[after_check.children[0] as usize];
        assert_eq!(after_two.seat, 2);
        let closed = &tree.nodes[after_two.children[0] as usize];
        assert_eq!(closed.kind, KIND_SHOWDOWN);
        assert_eq!(closed.live, 0b111);
        assert_eq!(closed.put[..3].iter().sum::<f64>(), 7.5);
    }

    #[test]
    fn a_bet_reopens_the_action_and_folds_end_it() {
        let tree = Builder::build(&cfg(3), Vec::new(), 1326, 1_000_000).unwrap();
        let root = &tree.nodes[0];
        // SB bets 33%: 2.475 to.
        let bet = root.actions.iter().position(|a| matches!(a, Action::Bet(_))).unwrap();
        let Action::Bet(to) = root.actions[bet] else { unreachable!() };
        assert!((to - 0.33 * 7.5).abs() < 1e-9);
        let bb = &tree.nodes[root.children[bet] as usize];
        assert_eq!(bb.seat, 1);
        assert_eq!(bb.actions[0], Action::Fold);
        assert!(matches!(bb.actions[1], Action::Call(_)));
        // BB folds, BTN folds: SB wins.
        let btn = &tree.nodes[bb.children[0] as usize];
        assert_eq!(btn.seat, 2);
        assert_eq!(btn.live, 0b101);
        let won = &tree.nodes[btn.children[0] as usize];
        assert_eq!(won.kind, KIND_FOLD_OUT);
        assert_eq!(won.seat, 0);
        // BB raises: SB and BTN both have to act again (BTN first in order).
        let raise = bb.actions.iter().position(|a| matches!(a, Action::Raise(_))).unwrap();
        let Action::Raise(rto) = bb.actions[raise] else { unreachable!() };
        assert!((rto - 2.5 * to).abs() < 1e-9);
        let btn2 = &tree.nodes[bb.children[raise] as usize];
        assert_eq!(btn2.seat, 2);
        let sb2 = &tree.nodes[btn2.children[1] as usize]; // BTN calls
        assert_eq!(sb2.seat, 0);
        assert!(matches!(sb2.actions[1], Action::Call(_)));
        let closed = &tree.nodes[sb2.children[1] as usize];
        assert_eq!(closed.kind, KIND_SHOWDOWN);
        assert_eq!(closed.live, 0b111);
        assert!((closed.put[0] - 2.5 - rto).abs() < 1e-9);
    }

    #[test]
    fn big_sizes_become_all_in_and_all_in_seats_stop_acting() {
        let mut c = cfg(3);
        c.pot = 60.0;
        c.stack = 40.0;
        c.invested = vec![20.0; 3];
        let tree = Builder::build(&c, Vec::new(), 1326, 1_000_000).unwrap();
        let root = &tree.nodes[0];
        // 75% of 60 = 45 > 40 behind: all-in 40. 33% = 19.8 stays a bet (under 65% of 40).
        let bets: Vec<_> = root.actions.iter().filter(|a| matches!(a, Action::Bet(_))).collect();
        assert_eq!(bets, vec![&Action::Bet(19.8), &Action::Bet(40.0)]);
        let bb = &tree.nodes[root.children[2] as usize];
        assert_eq!(bb.actions.len(), 2, "no raise over an all-in: {:?}", bb.actions);
        let btn = &tree.nodes[bb.children[1] as usize];
        let done = &tree.nodes[btn.children[1] as usize];
        assert_eq!(done.kind, KIND_SHOWDOWN);
        assert_eq!(done.put[..3], [60.0, 60.0, 60.0]);
        // A raise over the 19.8 bet would commit 2.5 × 19.8 = 49.5 > 40: it is the all-in.
        let bb = &tree.nodes[root.children[1] as usize];
        assert_eq!(bb.actions.len(), 3, "{:?}", bb.actions);
        assert_eq!(bb.actions[2], Action::Raise(40.0));
    }

    #[test]
    fn the_turn_is_dealt_on_the_sampled_cards_only_with_chips_behind() {
        let mut c = cfg(3);
        c.turn_cards = 3;
        let tree = Builder::build(&c, vec![4, 9, 50], 1326, 1_000_000).unwrap();
        let root = &tree.nodes[0];
        let n1 = &tree.nodes[root.children[0] as usize];
        let n2 = &tree.nodes[n1.children[0] as usize];
        let chance = &tree.nodes[n2.children[0] as usize];
        assert_eq!(chance.kind, KIND_CHANCE);
        assert_eq!(chance.children.len(), 3);
        for (i, k) in chance.children.iter().enumerate() {
            let turn = &tree.nodes[*k as usize];
            assert_eq!((turn.kind, turn.street, turn.seat), (KIND_ACTION, 1, 0));
            let mut n = turn;
            for _ in 0..3 {
                n = &tree.nodes[n.children[0] as usize];
            }
            assert_eq!(n.kind, KIND_SHOWDOWN);
            assert_eq!(n.runout_set, 1 + i as u32);
        }
        // All-in on the flop: no turn street.
        c.pot = 60.0;
        c.stack = 40.0;
        c.invested = vec![20.0; 3];
        let tree = Builder::build(&c, vec![4, 9, 50], 1326, 1_000_000).unwrap();
        let root = &tree.nodes[0];
        let bb = &tree.nodes[root.children[2] as usize]; // the all-in bet
        let btn = &tree.nodes[bb.children[1] as usize];
        assert_eq!(tree.nodes[btn.children[1] as usize].kind, KIND_SHOWDOWN);
    }

    #[test]
    fn dead_money_stays_in_the_pot_and_a_short_pot_is_refused() {
        let mut c = cfg(3);
        c.pot = 8.0; // SB folded its 0.5
        let tree = Builder::build(&c, Vec::new(), 1326, 1_000_000).unwrap();
        assert!((tree.dead - 0.5).abs() < 1e-9);
        let root = &tree.nodes[0];
        let Action::Bet(to) = root.actions[1] else { panic!("{:?}", root.actions) };
        assert!((to - 0.33 * 8.0).abs() < 1e-9, "bets size off the whole pot: {to}");
        // SB bets, both fold: SB owns the pot less the rake on the matched part (8.0).
        let bb = &tree.nodes[root.children[1] as usize];
        let btn = &tree.nodes[bb.children[0] as usize];
        let won = &tree.nodes[btn.children[0] as usize];
        assert_eq!(won.kind, KIND_FOLD_OUT);
        assert!((tree.pot_at(won) - (8.0 + to)).abs() < 1e-9);
        assert!((tree.fold_out_winnings(won) - (8.0 + to - 0.05 * 8.0)).abs() < 1e-9);
        c.pot = 7.0;
        assert!(Builder::build(&c, Vec::new(), 1326, 1_000_000).is_err());
    }
}
