//! Showdown values with several opponents, exact per board, over a fixed
//! sample of runouts.
//!
//! For one complete board every combo has a rank. Sorting the valid combos by
//! rank once per solve lets a leaf give every hero combo the reach mass of
//! each opponent's worse hands (`L`), equal hands (`E`) and compatible hands
//! (`T`) with three prefix-sum lookups: the whole-range prefix at the hero's
//! rank, minus the prefixes of the combos holding each of the hero's two
//! cards. The hero's share of the pot against opponents `q` is then
//!
//! ```text
//! Σ over subsets S of the live opponents that tie:
//!     Π_{q∈S} E_q · Π_{q∉S} L_q / (|S| + 1)
//! ```
//!
//! times the compatible mass of every seat that folded earlier. Opponents
//! are independent of each other given the hero's cards (a combo both could
//! hold counts for both).

use crate::cards::{combo_from_index, Card, NUM_CARDS, NUM_COMBOS};
use crate::evaluator::evaluate7;

/// Per-card lists of the combos holding that card, in combo-index order.
pub fn combos_by_card() -> Vec<Vec<u16>> {
    let mut out: Vec<Vec<u16>> = (0..NUM_CARDS).map(|_| Vec::with_capacity(51)).collect();
    for idx in 0..NUM_COMBOS {
        let (a, b) = combo_from_index(idx);
        out[a as usize].push(idx as u16);
        out[b as usize].push(idx as u16);
    }
    out
}

/// One complete board, prepared for prefix-sum showdowns.
pub struct Board {
    pub cards: [Card; 5],
    /// Rank of each combo; 0 when the board blocks it.
    pub rank: Vec<u32>,
    /// Valid combos sorted by rank.
    order: Vec<u16>,
    /// Per combo: positions in `order` where its rank group starts and ends.
    lt: Vec<u16>,
    le: Vec<u16>,
    /// Per card: its valid combos sorted by rank, flattened.
    card_start: [u32; NUM_CARDS + 1],
    card_combos: Vec<u16>,
    /// Per combo and each of its two cards: positions in that card's list
    /// where the rank group starts and ends.
    clt: Vec<[u8; 2]>,
    cle: Vec<[u8; 2]>,
}

impl Board {
    pub fn new(cards: [Card; 5]) -> Board {
        let mut blocked = 0u64;
        for &c in &cards {
            blocked |= 1 << c;
        }
        let mut rank = vec![0u32; NUM_COMBOS];
        let mut order: Vec<u16> = Vec::with_capacity(1081);
        let mut seven = [0u8; 7];
        seven[..5].copy_from_slice(&cards);
        for idx in 0..NUM_COMBOS {
            let (a, b) = combo_from_index(idx);
            if blocked & (1 << a) != 0 || blocked & (1 << b) != 0 {
                continue;
            }
            seven[5] = a;
            seven[6] = b;
            rank[idx] = evaluate7(&seven).max(1);
            order.push(idx as u16);
        }
        order.sort_by_key(|&i| rank[i as usize]);
        let mut lt = vec![0u16; NUM_COMBOS];
        let mut le = vec![0u16; NUM_COMBOS];
        let mut i = 0;
        while i < order.len() {
            let r = rank[order[i] as usize];
            let mut j = i;
            while j < order.len() && rank[order[j] as usize] == r {
                j += 1;
            }
            for &idx in &order[i..j] {
                lt[idx as usize] = i as u16;
                le[idx as usize] = j as u16;
            }
            i = j;
        }
        let mut card_start = [0u32; NUM_CARDS + 1];
        let mut card_combos: Vec<u16> = Vec::with_capacity(NUM_CARDS * 46);
        let mut clt = vec![[0u8; 2]; NUM_COMBOS];
        let mut cle = vec![[0u8; 2]; NUM_COMBOS];
        for c in 0..NUM_CARDS {
            card_start[c] = card_combos.len() as u32;
            if blocked & (1 << c) != 0 {
                continue;
            }
            let start = card_combos.len();
            for &idx in &order {
                let (a, b) = combo_from_index(idx as usize);
                if a as usize == c || b as usize == c {
                    card_combos.push(idx);
                }
            }
            let list = &card_combos[start..];
            let mut i = 0;
            while i < list.len() {
                let r = rank[list[i] as usize];
                let mut j = i;
                while j < list.len() && rank[list[j] as usize] == r {
                    j += 1;
                }
                for &idx in &list[i..j] {
                    let (a, _) = combo_from_index(idx as usize);
                    let slot = if a as usize == c { 0 } else { 1 };
                    clt[idx as usize][slot] = i as u8;
                    cle[idx as usize][slot] = j as u8;
                }
                i = j;
            }
        }
        card_start[NUM_CARDS] = card_combos.len() as u32;
        Board { cards, rank, order, lt, le, card_start, card_combos, clt, cle }
    }

    pub fn valid(&self, combo: usize) -> bool {
        self.rank[combo] != 0
    }

    /// Prefix sums of `reach` over the valid combos in rank order, and over
    /// each card's list, into `scratch` (sized by [`Board::scratch_len`]).
    fn prefixes(&self, reach: &[f32], scratch: &mut [f32]) {
        let n = self.order.len();
        let (pre, cpre) = scratch.split_at_mut(n + 1);
        pre[0] = 0.0;
        for (i, &idx) in self.order.iter().enumerate() {
            pre[i + 1] = pre[i] + reach[idx as usize];
        }
        for c in 0..NUM_CARDS {
            let (s, e) = (self.card_start[c] as usize, self.card_start[c + 1] as usize);
            let base = s + c; // one extra slot per card for the running total
            cpre[base] = 0.0;
            for (k, &idx) in self.card_combos[s..e].iter().enumerate() {
                cpre[base + k + 1] = cpre[base + k] + reach[idx as usize];
            }
        }
    }

    pub fn scratch_len(&self) -> usize {
        self.order.len() + 1 + self.card_combos.len() + NUM_CARDS
    }

    /// For hero combo `h` (valid on this board) against an opponent whose
    /// prefixes are in `scratch`: (worse mass, equal mass, compatible mass).
    #[inline]
    fn against(&self, h: usize, reach_h: f32, scratch: &[f32]) -> (f32, f32, f32) {
        let n = self.order.len();
        let (pre, cpre) = scratch.split_at(n + 1);
        let (a, b) = combo_from_index(h);
        let (a, b) = (a as usize, b as usize);
        let base_a = self.card_start[a] as usize + a;
        let base_b = self.card_start[b] as usize + b;
        let len_a = (self.card_start[a + 1] - self.card_start[a]) as usize;
        let len_b = (self.card_start[b + 1] - self.card_start[b]) as usize;
        let [ca, cb] = self.clt[h];
        let [da, db] = self.cle[h];
        let lt = pre[self.lt[h] as usize] - cpre[base_a + ca as usize] - cpre[base_b + cb as usize];
        let le = pre[self.le[h] as usize] - cpre[base_a + da as usize] - cpre[base_b + db as usize] + reach_h;
        let total = pre[n] - cpre[base_a + len_a] - cpre[base_b + len_b] + reach_h;
        (lt.max(0.0), (le - lt).max(0.0), total.max(0.0))
    }
}

/// Reach mass of `reach` compatible with every combo, with no board beyond
/// the one the reach vectors were already masked for: `total − Σ_{card ∈ h}
/// mass holding the card + reach[h]`.
pub fn compatible_mass(reach: &[f32], by_card: &[Vec<u16>], out: &mut [f32]) {
    let total: f32 = reach.iter().sum();
    let mut per_card = [0f32; NUM_CARDS];
    for c in 0..NUM_CARDS {
        per_card[c] = by_card[c].iter().map(|&i| reach[i as usize]).sum();
    }
    for h in 0..NUM_COMBOS {
        let (a, b) = combo_from_index(h);
        out[h] = (total - per_card[a as usize] - per_card[b as usize] + reach[h]).max(0.0);
    }
}

/// Showdown values for `hero` at a leaf checked down over `boards`.
///
/// `live` are the opponents still in the hand, `folded` the seats that left
/// it on the flop or turn (their reach still weights the hero's values). Adds
/// to `out[h]`, for every hero combo, the average over the boards that do not
/// block it of `share(h) × pot_eff − put_hero × others(h)` where `others(h)`
/// is the product of every other seat's compatible mass on that board.
#[allow(clippy::too_many_arguments)]
pub fn showdown(
    boards: &[Board],
    hero: usize,
    live: &[usize],
    folded: &[usize],
    reach: &[Vec<f32>],
    pot_eff: f64,
    put_hero: f64,
    out: &mut [f32],
    scratch: &mut Vec<f32>,
) {
    debug_assert!(hero < reach.len());
    let m = live.len();
    let subsets = 1usize << m;
    // Each board contributes to the combos it leaves valid; average per combo.
    let mut count = [0u16; NUM_COMBOS];
    let mut acc = vec![0f64; NUM_COMBOS];
    let seats = live.len() + folded.len();
    let mut lt = vec![0f32; m];
    let mut eq = vec![0f32; m];
    for board in boards {
        let stride = board.scratch_len();
        scratch.resize(stride * seats, 0.0);
        for (k, &q) in live.iter().chain(folded).enumerate() {
            board.prefixes(&reach[q], &mut scratch[k * stride..(k + 1) * stride]);
        }
        for &h in &board.order {
            let h = h as usize;
            count[h] += 1;
            let mut others = 1f64;
            for (k, &q) in live.iter().enumerate() {
                let (l, e, t) = board.against(h, reach[q][h], &scratch[k * stride..(k + 1) * stride]);
                lt[k] = l;
                eq[k] = e;
                others *= t as f64;
            }
            let mut folded_mass = 1f64;
            for (k, &q) in folded.iter().enumerate() {
                let (_, _, t) = board.against(h, reach[q][h], &scratch[(m + k) * stride..(m + k + 1) * stride]);
                folded_mass *= t as f64;
            }
            others *= folded_mass;
            let mut share = 0f64;
            for s in 0..subsets {
                let mut term = 1f64;
                for k in 0..m {
                    term *= if s & (1 << k) != 0 { eq[k] } else { lt[k] } as f64;
                }
                share += term / (s.count_ones() as f64 + 1.0);
            }
            share *= folded_mass;
            acc[h] += share * pot_eff - put_hero * others;
        }
    }
    for h in 0..NUM_COMBOS {
        if count[h] > 0 {
            out[h] += (acc[h] / count[h] as f64) as f32;
        }
    }
}

/// A fixed, seedable shuffle (splitmix64 + Fisher–Yates) so every solve and
/// every round samples the same runouts: common random numbers.
pub fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}

/// `k` turn+river completions of a flop: the first `k` of a fixed shuffle of
/// the unordered pairs of the remaining cards (nested: fewer runouts are a
/// subset of more).
pub fn flop_runouts(flop: &[Card], k: usize) -> Vec<[Card; 5]> {
    let rest: Vec<Card> = (0..NUM_CARDS as u8).filter(|c| !flop.contains(c)).collect();
    let mut pairs = Vec::with_capacity(rest.len() * (rest.len() - 1) / 2);
    for i in 0..rest.len() {
        for j in i + 1..rest.len() {
            pairs.push((rest[i], rest[j]));
        }
    }
    shuffled(pairs.len(), 7)
        .into_iter()
        .take(k.min(pairs.len()))
        .map(|i| [flop[0], flop[1], flop[2], pairs[i].0, pairs[i].1])
        .collect()
}

/// `k` of the cards not on `board`, from a fixed shuffle (nested as above).
pub fn sampled_cards(board: &[Card], k: usize) -> Vec<Card> {
    let rest: Vec<Card> = (0..NUM_CARDS as u8).filter(|c| !board.contains(c)).collect();
    shuffled(rest.len(), 11).into_iter().take(k.min(rest.len())).map(|i| rest[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_cards;

    fn board(s: &str) -> Board {
        let cards = parse_cards(s).unwrap();
        Board::new([cards[0], cards[1], cards[2], cards[3], cards[4]])
    }

    /// Brute force: hero's mass-weighted share against two independent
    /// opponents, compatible with the hero and the board.
    fn brute(b: &Board, h: usize, reach: &[Vec<f32>], live: &[usize]) -> (f64, f64) {
        let (ha, hb) = combo_from_index(h);
        let compatible = |j: usize| {
            let (a, c) = combo_from_index(j);
            b.valid(j) && a != ha && a != hb && c != ha && c != hb
        };
        let combos: Vec<Vec<usize>> = live
            .iter()
            .map(|&q| (0..NUM_COMBOS).filter(|&j| compatible(j) && reach[q][j] > 0.0).collect())
            .collect();
        let mut share = 0f64;
        let mut mass = 0f64;
        fn rec(
            b: &Board,
            h: usize,
            reach: &[Vec<f32>],
            live: &[usize],
            combos: &[Vec<usize>],
            k: usize,
            w: f64,
            ties: usize,
            beats: bool,
            share: &mut f64,
            mass: &mut f64,
        ) {
            if k == live.len() {
                *mass += w;
                if beats {
                    *share += w / (ties as f64 + 1.0);
                }
                return;
            }
            for &j in &combos[k] {
                let w2 = w * reach[live[k]][j] as f64;
                let (rj, rh) = (b.rank[j], b.rank[h]);
                rec(b, h, reach, live, combos, k + 1, w2, ties + (rj == rh) as usize, beats && rj <= rh, share, mass);
            }
        }
        rec(b, h, reach, live, &combos, 0, 1.0, 0, true, &mut share, &mut mass);
        (share, mass)
    }

    #[test]
    fn prefix_sums_match_brute_force_three_ways() {
        let b = board("Ks7h2d9c9s");
        let by_card = combos_by_card();
        // Opponents with structured ranges so ties and blockers both occur.
        let mut reach = vec![vec![0f32; NUM_COMBOS]; 3];
        for idx in 0..NUM_COMBOS {
            if !b.valid(idx) {
                continue;
            }
            let (a, c) = combo_from_index(idx);
            reach[1][idx] = if idx % 3 == 0 { 0.7 } else { 0.0 } + if a / 4 == c / 4 { 0.3 } else { 0.0 };
            reach[2][idx] = if idx % 5 < 2 { 0.5 } else { 0.1 };
        }
        let mut out = vec![0f32; NUM_COMBOS];
        let mut scratch = Vec::new();
        let boards = [b];
        showdown(&boards, 0, &[1, 2], &[], &reach, 100.0, 10.0, &mut out, &mut scratch);
        let mut checked = 0;
        for h in (0..NUM_COMBOS).step_by(37) {
            if !boards[0].valid(h) {
                continue;
            }
            let (share, mass) = brute(&boards[0], h, &reach, &[1, 2]);
            let expect = share * 100.0 - 10.0 * mass;
            assert!((out[h] as f64 - expect).abs() < 1e-2 * expect.abs().max(1.0), "combo {h}: {} vs {expect}", out[h]);
            checked += 1;
        }
        assert!(checked > 20);
        // The compatible mass matches the board-free helper when nothing is blocked beyond the hero.
        let mut mass = vec![0f32; NUM_COMBOS];
        compatible_mass(&reach[1], &by_card, &mut mass);
        let (_, m) = brute(&boards[0], 5, &[vec![], reach[1].clone()], &[1]);
        assert!((mass[5] as f64 - m).abs() < 1e-3);
    }

    #[test]
    fn a_folded_seat_only_weights_the_value() {
        let b = board("Ks7h2d9c9s");
        let mut reach = vec![vec![0f32; NUM_COMBOS]; 3];
        for idx in 0..NUM_COMBOS {
            if b.valid(idx) {
                reach[1][idx] = 0.5;
                reach[2][idx] = 1.0;
            }
        }
        let mut hu = vec![0f32; NUM_COMBOS];
        let mut with_folded = vec![0f32; NUM_COMBOS];
        let mut scratch = Vec::new();
        let boards = [b];
        showdown(&boards, 0, &[1], &[], &reach, 100.0, 10.0, &mut hu, &mut scratch);
        showdown(&boards, 0, &[1], &[2], &reach, 100.0, 10.0, &mut with_folded, &mut scratch);
        // Seat 2 holds every valid combo not sharing a card with the hero: 1081 − 2·45 + ... = C(45,2) = 990.
        for h in 0..NUM_COMBOS {
            if boards[0].valid(h) {
                assert!((with_folded[h] - hu[h] * 990.0).abs() < 1e-2 * hu[h].abs().max(1.0), "{h}: {} vs {}", with_folded[h], hu[h] * 990.0);
            }
        }
    }

    #[test]
    fn runouts_are_fixed_and_nested() {
        let flop = parse_cards("Ks7h2d").unwrap();
        let a = flop_runouts(&flop, 12);
        let b = flop_runouts(&flop, 48);
        assert_eq!(a.len(), 12);
        assert_eq!(b.len(), 48);
        assert_eq!(&b[..12], &a[..]);
        for r in &b {
            assert!(!flop.contains(&r[3]) && !flop.contains(&r[4]) && r[3] != r[4]);
        }
        assert_eq!(flop_runouts(&flop, 5000).len(), 1176);
        let turns = sampled_cards(&flop, 8);
        assert_eq!(turns.len(), 8);
        assert_eq!(&sampled_cards(&flop, 20)[..8], &turns[..]);
    }
}
