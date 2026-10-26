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
//! times the compatible mass of every seat that folded earlier.
//!
//! With two other seats the deck is exact: the mass of hand pairs that do
//! not share a card is the product of the two masses, minus the pairs that
//! share a card (a dot product over cards of the two per-card masses), plus
//! the pairs that are the same hand (counted twice in the dot product) —
//! `Pair::exclusive`. With three or more other seats the seats are taken as
//! independent of each other given the hero's cards (a combo two of them
//! could both hold counts for both), which lifts shares by a percent or two
//! of the pot; four-way pots are rare enough at the preflop terminals for
//! that to stand.

use crate::cards::{combo_from_index, rank, suit, Card, NUM_CARDS, NUM_COMBOS};
use crate::evaluator::evaluate7;
use rayon::prelude::*;

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
    /// Share of the deck this board stands for in its sample (see
    /// [`sample_cards`]); boards average with these weights.
    pub weight: f32,
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
    /// Per combo: index of its rank group (groups numbered in rank order).
    group: Vec<u16>,
    /// Per rank group and card: positions in the card's list where the
    /// group starts (`glt`) and ends (`gle`), flattened `[group][card]`.
    glt: Vec<u8>,
    gle: Vec<u8>,
}

impl Board {
    pub fn new(cards: [Card; 5]) -> Board {
        Board::weighted(cards, 1.0)
    }

    pub fn weighted(cards: [Card; 5], weight: f32) -> Board {
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
        let mut group = vec![0u16; NUM_COMBOS];
        let mut group_ranks: Vec<u32> = Vec::new();
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
                group[idx as usize] = group_ranks.len() as u16;
            }
            group_ranks.push(r);
            i = j;
        }
        let groups = group_ranks.len();
        let mut card_start = [0u32; NUM_CARDS + 1];
        let mut card_combos: Vec<u16> = Vec::with_capacity(NUM_CARDS * 46);
        let mut clt = vec![[0u8; 2]; NUM_COMBOS];
        let mut cle = vec![[0u8; 2]; NUM_COMBOS];
        let mut glt = vec![0u8; groups * NUM_CARDS];
        let mut gle = vec![0u8; groups * NUM_CARDS];
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
            // Walk the card's list and the rank groups together: every group
            // gets the position where it starts and ends in this list.
            let (mut i, mut g) = (0, 0);
            while g < groups {
                let r = group_ranks[g];
                while i < list.len() && rank[list[i] as usize] < r {
                    i += 1;
                }
                glt[g * NUM_CARDS + c] = i as u8;
                let mut j = i;
                while j < list.len() && rank[list[j] as usize] == r {
                    j += 1;
                }
                gle[g * NUM_CARDS + c] = j as u8;
                for &idx in &list[i..j] {
                    let (a, _) = combo_from_index(idx as usize);
                    let slot = if a as usize == c { 0 } else { 1 };
                    clt[idx as usize][slot] = i as u8;
                    cle[idx as usize][slot] = j as u8;
                }
                i = j;
                g += 1;
            }
        }
        card_start[NUM_CARDS] = card_combos.len() as u32;
        Board { cards, weight, rank, order, lt, le, card_start, card_combos, clt, cle, group, glt, gle }
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

/// How much a sample of chance cards under-weights the other seats' hands.
///
/// A hero combo's value is averaged over the sampled cards it does not hold;
/// every other seat's combo is then counted only under the cards it does not
/// hold either, which is fewer. In the game the cards come from the deck
/// everyone's hands leave: `unseen` cards from the hero's point of view, `2 ×
/// others` fewer from the table's. `drawn` cards (one turn, one river or
/// both) at a time, the ratio of the two counts is the correction, per
/// opponent combo in expectation over the sample.
pub fn sample_weight(unseen: usize, others: usize, drawn: usize) -> f64 {
    let choose = |n: usize, k: usize| -> f64 { (0..k).map(|i| (n - i) as f64 / (i + 1) as f64).product() };
    let left = unseen.saturating_sub(2 * others);
    if left < drawn {
        return 1.0;
    }
    choose(unseen, drawn) / choose(left, drawn)
}

/// Showdown values for `hero` at a leaf checked down over `boards`.
///
/// `live` are the opponents still in the hand, `folded` the seats that left
/// it on the flop or turn (their reach still weights the hero's values). Adds
/// to `out[h]`, for every hero combo, the weighted average over the boards
/// that do not block it of `share(h) × pot_eff − put_hero × others(h)` where
/// `others(h)` is the mass of the other seats' hands compatible with `h` on
/// that board, scaled by `sample_weight` for the `drawn` cards the boards
/// add to the leaf's.
///
/// The boards are independent given the ranges and are nearly all of a
/// solve's work (a flop street's time is linear in the runouts), so they are
/// evaluated across the cores and their sums combined.
#[allow(clippy::too_many_arguments)]
pub fn showdown(
    boards: &[Board],
    drawn: usize,
    hero: usize,
    live: &[usize],
    folded: &[usize],
    reach: &[Vec<f32>],
    pot_eff: f64,
    put_hero: f64,
    out: &mut [f32],
) {
    debug_assert!(hero < reach.len());
    let seats = live.len() + folded.len();
    let weight = sample_weight(52 - (5 - drawn) - 2, seats, drawn);
    // Each board contributes to the combos it leaves valid; weighted average per combo.
    let (acc, norm) = boards
        .par_iter()
        .fold(
            || Partial::new(seats),
            |mut partial, board| {
                partial.add(board, live, folded, reach, pot_eff, put_hero);
                partial
            },
        )
        .map(|partial| (partial.acc, partial.norm))
        .reduce(
            || (vec![0f64; NUM_COMBOS], vec![0f32; NUM_COMBOS]),
            |(mut acc, mut norm), (acc2, norm2)| {
                for h in 0..NUM_COMBOS {
                    acc[h] += acc2[h];
                    norm[h] += norm2[h];
                }
                (acc, norm)
            },
        );
    for h in 0..NUM_COMBOS {
        if norm[h] > 0.0 {
            out[h] += (acc[h] * weight / norm[h] as f64) as f32;
        }
    }
}

/// One worker's running sums over the boards it is given, with its own
/// scratch space.
struct Partial {
    /// Exact two-seat showdowns; the general loop serves three or more.
    pair: Option<Pair>,
    scratch: Vec<f32>,
    acc: Vec<f64>,
    norm: Vec<f32>,
}

impl Partial {
    fn new(seats: usize) -> Partial {
        Partial {
            pair: (seats == 2).then(Pair::new),
            scratch: Vec::new(),
            acc: vec![0f64; NUM_COMBOS],
            norm: vec![0f32; NUM_COMBOS],
        }
    }

    fn add(&mut self, board: &Board, live: &[usize], folded: &[usize], reach: &[Vec<f32>], pot_eff: f64, put_hero: f64) {
        if let Some(pair) = &mut self.pair {
            pair.showdown(board, live, folded, reach, pot_eff, put_hero, &mut self.acc, &mut self.norm, &mut self.scratch);
            return;
        }
        let m = live.len();
        let seats = m + folded.len();
        let subsets = 1usize << m;
        let mut lt = vec![0f32; m];
        let mut eq = vec![0f32; m];
        let stride = board.scratch_len();
        let scratch = &mut self.scratch;
        scratch.resize(stride * seats, 0.0);
        for (k, &q) in live.iter().chain(folded).enumerate() {
            board.prefixes(&reach[q], &mut scratch[k * stride..(k + 1) * stride]);
        }
        for &h in &board.order {
            let h = h as usize;
            self.norm[h] += board.weight;
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
            self.acc[h] += board.weight as f64 * (share * pot_eff - put_hero * others);
        }
    }
}

/// Exact showdown against two other seats (live or folded): hand pairs that
/// share a card are not counted.
///
/// For seats 1 and 2 with per-hero-combo category masses `X₁`, `Y₂` (worse,
/// equal or compatible) the mass of pairs with no card in common is
///
/// ```text
/// X₁ · Y₂  −  Σ_cards c  X₁(c) · Y₂(c)  +  Σ_{o in both} r₁(o) · r₂(o)
/// ```
///
/// where `X₁(c)` is the mass of seat 1's hands in the category holding card
/// `c`, and the last term returns the identical pairs the middle term removed
/// twice. Everything excludes hands holding the hero's cards. The per-card
/// masses come from the per-card prefix sums at the hero's rank group (shared
/// by every hero combo of the group), less the two hands `{c, hero card}`.
/// Card vectors are padded to [`LANES`] so the loops vectorise.
struct Pair {
    /// `[seat][x * LANES + c]`: seat's reach of combo {x, c} on this board.
    pairs: [Vec<f32>; 2],
    /// `[x * LANES + c]`: rank group of combo {x, c} as a float, infinite
    /// when the combo is blocked (or x == c).
    group: Vec<f32>,
    /// Per seat: per-card compatible mass (`t`), and worse (`lg`) / equal
    /// (`eg`) mass at the rank group in hand.
    t: [[f32; LANES]; 2],
    lg: [[f32; LANES]; 2],
    eg: [[f32; LANES]; 2],
    /// Reach product r₁ ⊙ r₂ (the identical-pair term's prefixes).
    product: Vec<f32>,
}

/// Cards padded to a multiple of 16 lanes.
const LANES: usize = 64;

#[inline(always)]
fn dot(x: &[f32; LANES], y: &[f32; LANES]) -> f64 {
    let mut acc = [0f32; 8];
    for (cx, cy) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0) {
        for j in 0..8 {
            acc[j] += cx[j] * cy[j];
        }
    }
    acc.iter().map(|v| *v as f64).sum()
}

/// The per-card dot products the pair formulas need, from the two seats'
/// category vectors `[[worse, equal, compatible]; 2]`: `[t₁·t₂, l₁·l₂,
/// l₁·e₂, e₁·l₂, e₁·e₂]` with two live seats, `[t₁·t₂, l₁·t₂, e₁·t₂, 0, 0]`
/// with the second folded. Kept out of line (like [`category_vectors`]) so
/// the loops vectorise.
#[inline(never)]
fn dots(v: &[[[f32; LANES]; 3]; 2], both_live: bool) -> [f64; 5] {
    let [[l1, e1, t1], [l2, e2, t2]] = v;
    if both_live {
        [dot(t1, t2), dot(l1, l2), dot(l1, e2), dot(e1, l2), dot(e1, e2)]
    } else {
        [dot(t1, t2), dot(l1, t2), dot(e1, t2), 0.0, 0.0]
    }
}

/// Per-card worse / equal / compatible masses of one seat for a hero combo
/// {a, b} at rank group `gf`: the group's per-card masses (`lg`, `eg`, `t`)
/// less the seat's hands `{a, c}` and `{b, c}` that fall in the category
/// (`groups` are the rank groups of those hands, `reaches` their reach).
/// Kept out of line: inlined into the hero loop the compiler no longer
/// vectorises it.
#[inline(never)]
fn category_vectors(
    gf: f32,
    groups: [&[f32; LANES]; 2],
    reaches: [&[f32; LANES]; 2],
    [lg, eg, t]: [&[f32; LANES]; 3],
    out: &mut [[f32; LANES]; 3],
) {
    let [ga, gb] = groups;
    let [ra, rb] = reaches;
    let [vl, ve, vt] = out;
    for c in 0..LANES {
        let (r1, r2) = (ra[c], rb[c]);
        let wa = if ga[c] < gf { r1 } else { 0.0 };
        let wb = if gb[c] < gf { r2 } else { 0.0 };
        vl[c] = lg[c] - wa - wb;
        let xa = if ga[c] == gf { r1 } else { 0.0 };
        let xb = if gb[c] == gf { r2 } else { 0.0 };
        ve[c] = eg[c] - xa - xb;
        vt[c] = t[c] - r1 - r2;
    }
}

/// Mass of pairs with no card in common from the category totals `x`, `y`,
/// their per-card dot product and the identical-pair mass.
#[inline(always)]
fn exclusive(x: f32, y: f32, dot_xy: f64, same: f32) -> f64 {
    x as f64 * y as f64 - dot_xy + same as f64
}

impl Pair {
    fn new() -> Pair {
        Pair {
            pairs: [vec![0f32; NUM_CARDS * LANES], vec![0f32; NUM_CARDS * LANES]],
            group: vec![f32::INFINITY; NUM_CARDS * LANES],
            t: [[0f32; LANES]; 2],
            lg: [[0f32; LANES]; 2],
            eg: [[0f32; LANES]; 2],
            product: vec![0f32; NUM_COMBOS],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn showdown(
        &mut self,
        board: &Board,
        live: &[usize],
        folded: &[usize],
        reach: &[Vec<f32>],
        pot_eff: f64,
        put_hero: f64,
        acc: &mut [f64],
        norm: &mut [f32],
        scratch: &mut Vec<f32>,
    ) {
        let seats: [usize; 2] = match (live, folded) {
            ([a, b], []) => [*a, *b],
            ([a], [b]) => [*a, *b],
            _ => unreachable!("Pair needs two other seats"),
        };
        let both_live = live.len() == 2;
        let stride = board.scratch_len();
        scratch.resize(stride * 3, 0.0);
        self.group.iter_mut().for_each(|v| *v = f32::INFINITY);
        for &idx in &board.order {
            let (x, c) = combo_from_index(idx as usize);
            let g = board.group[idx as usize] as f32;
            self.group[x as usize * LANES + c as usize] = g;
            self.group[c as usize * LANES + x as usize] = g;
        }
        for (k, &q) in seats.iter().enumerate() {
            board.prefixes(&reach[q], &mut scratch[k * stride..(k + 1) * stride]);
            let table = &mut self.pairs[k];
            table.iter_mut().for_each(|v| *v = 0.0);
            for &idx in &board.order {
                let (x, c) = combo_from_index(idx as usize);
                let r = reach[q][idx as usize];
                table[x as usize * LANES + c as usize] = r;
                table[c as usize * LANES + x as usize] = r;
            }
            let (_, cpre) = scratch[k * stride..(k + 1) * stride].split_at(board.order.len() + 1);
            for c in 0..NUM_CARDS {
                let base = board.card_start[c] as usize + c;
                let len = (board.card_start[c + 1] - board.card_start[c]) as usize;
                self.t[k][c] = cpre[base + len];
            }
        }
        for ((p, r1), r2) in self.product.iter_mut().zip(&reach[seats[0]]).zip(&reach[seats[1]]) {
            *p = r1 * r2;
        }
        board.prefixes(&self.product, &mut scratch[2 * stride..3 * stride]);
        let (s1, rest) = scratch.split_at(stride);
        let (s2, sp) = rest.split_at(stride);
        let mut current = u16::MAX;
        // Per seat and category (worse, equal, compatible): the per-card
        // vector for the hero combo in hand. On the stack, so the loops that
        // fill it from `self` cannot alias and vectorise.
        let mut vecs = [[[0f32; LANES]; 3]; 2];
        for &h in &board.order {
            let h = h as usize;
            norm[h] += board.weight;
            let gi = board.group[h];
            if gi != current {
                current = gi;
                let gi = gi as usize;
                let glt = &board.glt[gi * NUM_CARDS..(gi + 1) * NUM_CARDS];
                let gle = &board.gle[gi * NUM_CARDS..(gi + 1) * NUM_CARDS];
                for k in 0..2 {
                    let s = if k == 0 { s1 } else { s2 };
                    let (_, cpre) = s.split_at(board.order.len() + 1);
                    for c in 0..NUM_CARDS {
                        let base = board.card_start[c] as usize + c;
                        let lt = cpre[base + glt[c] as usize];
                        self.lg[k][c] = lt;
                        self.eg[k][c] = cpre[base + gle[c] as usize] - lt;
                    }
                }
            }
            let gf = gi as f32;
            let (a, b) = combo_from_index(h);
            let (a, b) = (a as usize, b as usize);
            let ga: &[f32; LANES] = self.group[a * LANES..(a + 1) * LANES].try_into().unwrap();
            let gb: &[f32; LANES] = self.group[b * LANES..(b + 1) * LANES].try_into().unwrap();
            let mut scalars = [(0f32, 0f32, 0f32); 2];
            for k in 0..2 {
                let s = if k == 0 { s1 } else { s2 };
                scalars[k] = board.against(h, reach[seats[k]][h], s);
                let ra: &[f32; LANES] = self.pairs[k][a * LANES..(a + 1) * LANES].try_into().unwrap();
                let rb: &[f32; LANES] = self.pairs[k][b * LANES..(b + 1) * LANES].try_into().unwrap();
                let (lg, eg, t) = (&self.lg[k], &self.eg[k], &self.t[k]);
                category_vectors(gf, [ga, gb], [ra, rb], [lg, eg, t], &mut vecs[k]);
                for v in &mut vecs[k] {
                    v[a] = 0.0;
                    v[b] = 0.0;
                }
            }
            let (p_lt, p_eq, p_t) = board.against(h, self.product[h], sp);
            let d = dots(&vecs, both_live);
            let (sl1, se1, st1) = scalars[0];
            let (sl2, se2, st2) = scalars[1];
            let others = exclusive(st1, st2, d[0], p_t);
            let share = if both_live {
                let ww = exclusive(sl1, sl2, d[1], p_lt);
                let we = exclusive(sl1, se2, d[2], 0.0);
                let ew = exclusive(se1, sl2, d[3], 0.0);
                let ee = exclusive(se1, se2, d[4], p_eq);
                ww + (we + ew) / 2.0 + ee / 3.0
            } else {
                let wt = exclusive(sl1, st2, d[1], p_lt);
                let et = exclusive(se1, st2, d[2], p_eq);
                wt + et / 2.0
            };
            acc[h] += board.weight as f64 * (share.max(0.0) * pot_eff - put_hero * others.max(0.0));
        }
    }
}

/// Reach mass of two seats' hand pairs compatible with every hero combo and
/// with each other (no shared card), with no board beyond the one the reach
/// vectors were masked for: `compatible_mass` of each, minus the pairs that
/// share a card, plus the identical pairs.
///
/// With `P_k[c]` seat k's mass holding card `c`, `R_k(x, c)` its reach of
/// combo {x, c} and `Q = R₁ R₂ᵀ`, the shared-card term for hero combo {a, b}
/// is `Σ_{c ∉ {a, b}} (P₁[c] − R₁(a, c) − R₁(b, c)) (P₂[c] − R₂(a, c) −
/// R₂(b, c))`, which expands into per-board constants, per-card vectors and
/// entries of `Q`: constant work per hero combo.
pub fn compatible_pair_mass(r1: &[f32], r2: &[f32], by_card: &[Vec<u16>], out: &mut [f64]) {
    let mut tables = [vec![0f32; NUM_CARDS * NUM_CARDS], vec![0f32; NUM_CARDS * NUM_CARDS]];
    let mut per_card = [[0f64; NUM_CARDS]; 2];
    let mut per_card_product = [0f64; NUM_CARDS];
    for (k, r) in [r1, r2].into_iter().enumerate() {
        for c in 0..NUM_CARDS {
            for &i in &by_card[c] {
                let (x, y) = combo_from_index(i as usize);
                let other = if x as usize == c { y } else { x } as usize;
                tables[k][c * NUM_CARDS + other] = r[i as usize];
                per_card[k][c] += r[i as usize] as f64;
            }
        }
    }
    let mut product_total = 0f64;
    for c in 0..NUM_CARDS {
        for &i in &by_card[c] {
            per_card_product[c] += r1[i as usize] as f64 * r2[i as usize] as f64;
        }
    }
    for h in 0..NUM_COMBOS {
        product_total += r1[h] as f64 * r2[h] as f64;
    }
    let total1: f64 = r1.iter().map(|v| *v as f64).sum();
    let total2: f64 = r2.iter().map(|v| *v as f64).sum();
    let (p1, p2) = (&per_card[0], &per_card[1]);
    // constant = Σ_c P₁[c] P₂[c]; u12[a] = Σ_c P₁[c] R₂(a, c); u21[a] = Σ_c R₁(a, c) P₂[c]; q[a][b] = Σ_c R₁(a, c) R₂(b, c)
    let constant: f64 = (0..NUM_CARDS).map(|c| p1[c] * p2[c]).sum();
    let mut u12 = [0f64; NUM_CARDS];
    let mut u21 = [0f64; NUM_CARDS];
    let mut q = vec![0f64; NUM_CARDS * NUM_CARDS];
    for a in 0..NUM_CARDS {
        let ra = &tables[0][a * NUM_CARDS..(a + 1) * NUM_CARDS];
        let sa = &tables[1][a * NUM_CARDS..(a + 1) * NUM_CARDS];
        for c in 0..NUM_CARDS {
            u12[a] += p1[c] * sa[c] as f64;
            u21[a] += ra[c] as f64 * p2[c];
        }
        for b in 0..NUM_CARDS {
            let sb = &tables[1][b * NUM_CARDS..(b + 1) * NUM_CARDS];
            q[a * NUM_CARDS + b] = (0..NUM_CARDS).map(|c| ra[c] as f64 * sb[c] as f64).sum();
        }
    }
    for h in 0..NUM_COMBOS {
        let (a, b) = combo_from_index(h);
        let (a, b) = (a as usize, b as usize);
        let (r1h, r2h) = (r1[h] as f64, r2[h] as f64);
        let mass1 = total1 - p1[a] - p1[b] + r1h;
        let mass2 = total2 - p2[a] - p2[b] + r2h;
        let at_a = (p1[a] - r1h) * (p2[a] - r2h);
        let at_b = (p1[b] - r1h) * (p2[b] - r2h);
        let shared = constant - u12[a] - u12[b] - u21[a] - u21[b]
            + q[a * NUM_CARDS + a]
            + q[a * NUM_CARDS + b]
            + q[b * NUM_CARDS + a]
            + q[b * NUM_CARDS + b]
            - at_a
            - at_b;
        let same = product_total - per_card_product[a] - per_card_product[b] + r1h * r2h;
        out[h] = (mass1 * mass2 - shared + same).max(0.0);
    }
}

/// Texture classes of a card dealt on a board, in priority order: the class
/// of a card is the first that applies.
pub const CLASS_NAMES: [&str; 5] = ["flush", "pair", "straight", "over", "blank"];

/// Class of `c` dealt on `board` (three or four cards): `flush` when its suit
/// then shows three or more times, `pair` when its rank is already there,
/// `straight` when some five-rank window holds it and two other board ranks
/// (two hole cards complete a straight through it), `over` when it is above
/// every board rank, `blank` otherwise.
pub fn card_class(board: &[Card], c: Card) -> usize {
    let (r, su) = (rank(c), suit(c));
    let suited = board.iter().filter(|&&b| suit(b) == su).count() + 1;
    if suited >= 3 {
        return 0;
    }
    if board.iter().any(|&b| rank(b) == r) {
        return 1;
    }
    // rank bits with the ace also low
    let bit = |x: u8| -> u16 { (1 << (x + 1)) | if x == 12 { 1 } else { 0 } };
    let ranks = board.iter().fold(0u16, |m, &b| m | bit(rank(b)));
    let mine = bit(r);
    for low in 0..10u16 {
        let window = 0b11111u16 << low;
        if window & mine != 0 && (window & ranks & !mine).count_ones() >= 2 {
            return 2;
        }
    }
    if board.iter().all(|&b| rank(b) < r) {
        return 3;
    }
    4
}

/// A class-weighted sample of `k` cards to deal on `board` (three or four
/// cards): cards are grouped by [`card_class`], every class present gets
/// samples in proportion to its size (at least one), spread evenly through
/// the class in rank order, and each sampled card carries the share of the
/// deck it stands for (class share ÷ samples in the class), so weighted
/// averages over the sample estimate averages over the deck while every
/// texture is represented. Returns `(cards, weights)`; weights sum to one.
/// `k` at or above the deck returns every card at equal weight.
pub fn sample_cards(board: &[Card], k: usize) -> (Vec<Card>, Vec<f32>) {
    let deck: Vec<Card> = (0..NUM_CARDS as Card).filter(|c| !board.contains(c)).collect();
    let n = deck.len();
    if k >= n {
        return (deck, vec![1.0 / n as f32; n]);
    }
    let mut classes: Vec<Vec<Card>> = vec![Vec::new(); CLASS_NAMES.len()];
    for &c in &deck {
        classes[card_class(board, c)].push(c);
    }
    let present: Vec<usize> = (0..classes.len()).filter(|&i| !classes[i].is_empty()).collect();
    let k = k.max(1);
    // Largest-remainder allocation with a floor of one per class; with fewer
    // samples than classes the biggest classes take them.
    let mut alloc = vec![0usize; classes.len()];
    if k >= present.len() {
        for &i in &present {
            alloc[i] = 1;
        }
        let spare = k - present.len();
        let shares: Vec<f64> = present.iter().map(|&i| spare as f64 * classes[i].len() as f64 / n as f64).collect();
        let mut given = 0;
        for (j, &i) in present.iter().enumerate() {
            alloc[i] += shares[j].floor() as usize;
            given += shares[j].floor() as usize;
        }
        let mut by_remainder: Vec<usize> = (0..present.len()).collect();
        by_remainder.sort_by(|&x, &y| (shares[y] - shares[y].floor()).partial_cmp(&(shares[x] - shares[x].floor())).unwrap());
        for &j in by_remainder.iter().take(spare - given) {
            alloc[present[j]] += 1;
        }
    } else {
        let mut biggest = present.clone();
        biggest.sort_by_key(|&i| std::cmp::Reverse(classes[i].len()));
        for &i in biggest.iter().take(k) {
            alloc[i] = 1;
        }
    }
    let covered: usize = (0..classes.len()).filter(|&i| alloc[i] > 0).map(|i| classes[i].len()).sum();
    let mut cards = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);
    for (i, class) in classes.iter().enumerate() {
        let m = alloc[i];
        if m == 0 {
            continue;
        }
        let share = class.len() as f32 / covered as f32;
        for j in 0..m {
            cards.push(class[(2 * j + 1) * class.len() / (2 * m)]);
            weights.push(share / m as f32);
        }
    }
    (cards, weights)
}

/// [`sample_cards`] with particular cards guaranteed a place: each `fixed`
/// card not already sampled stands in for a sampled card of its own texture
/// class (taking its weight, so the sample stays class-weighted); when its
/// class has no sampled card the card joins with its class's share of the
/// deck and the other weights shrink to make room.
pub fn sample_cards_with(board: &[Card], k: usize, fixed: &[Card]) -> (Vec<Card>, Vec<f32>) {
    let (mut cards, mut weights) = sample_cards(board, k);
    let deck: Vec<Card> = (0..NUM_CARDS as Card).filter(|c| !board.contains(c)).collect();
    let class_size = |class: usize| deck.iter().filter(|&&c| card_class(board, c) == class).count();
    for &f in fixed {
        if cards.contains(&f) || !deck.contains(&f) {
            continue;
        }
        let class = card_class(board, f);
        if let Some(i) = cards.iter().position(|&c| card_class(board, c) == class) {
            cards[i] = f;
            continue;
        }
        let covered: usize = {
            let mut seen: Vec<usize> = cards.iter().map(|&c| card_class(board, c)).collect();
            seen.sort_unstable();
            seen.dedup();
            seen.into_iter().map(class_size).sum()
        };
        let joining = class_size(class);
        let scale = covered as f32 / (covered + joining) as f32;
        for w in weights.iter_mut() {
            *w *= scale;
        }
        cards.push(f);
        weights.push(joining as f32 / (covered + joining) as f32);
    }
    (cards, weights)
}

/// Class-weighted two-card runouts of a flop: `turns` sampled turn cards,
/// each followed by `rivers` sampled rivers, weights multiplied.
pub fn sample_runouts(flop: &[Card], turns: usize, rivers: usize) -> Vec<([Card; 5], f32)> {
    let (ts, tw) = sample_cards(flop, turns);
    let mut out = Vec::with_capacity(turns * rivers);
    for (t, wt) in ts.into_iter().zip(tw) {
        let four = [flop[0], flop[1], flop[2], t];
        let (rs, rw) = sample_cards(&four, rivers);
        for (r, wr) in rs.into_iter().zip(rw) {
            out.push(([flop[0], flop[1], flop[2], t, r], wt * wr));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_cards;

    fn board(s: &str) -> Board {
        let cards = parse_cards(s).unwrap();
        Board::new([cards[0], cards[1], cards[2], cards[3], cards[4]])
    }

    /// Brute force: hero's mass-weighted share against the `live` seats,
    /// weighted by the `folded` seats' mass too. Every hand is compatible with
    /// the hero and the board; with `disjoint` the other seats' hands share
    /// no card with each other either.
    fn brute(b: &Board, h: usize, reach: &[Vec<f32>], live: &[usize], folded: &[usize], disjoint: bool) -> (f64, f64) {
        let (ha, hb) = combo_from_index(h);
        let compatible = |j: usize| {
            let (a, c) = combo_from_index(j);
            b.valid(j) && a != ha && a != hb && c != ha && c != hb
        };
        let seats: Vec<usize> = live.iter().chain(folded).copied().collect();
        let combos: Vec<Vec<usize>> = seats
            .iter()
            .map(|&q| (0..NUM_COMBOS).filter(|&j| compatible(j) && reach[q][j] > 0.0).collect())
            .collect();
        let mut share = 0f64;
        let mut mass = 0f64;
        #[allow(clippy::too_many_arguments)]
        fn rec(
            b: &Board,
            h: usize,
            reach: &[Vec<f32>],
            seats: &[usize],
            live: usize,
            combos: &[Vec<usize>],
            disjoint: bool,
            used: u64,
            k: usize,
            w: f64,
            ties: usize,
            beats: bool,
            share: &mut f64,
            mass: &mut f64,
        ) {
            if k == seats.len() {
                *mass += w;
                if beats {
                    *share += w / (ties as f64 + 1.0);
                }
                return;
            }
            for &j in &combos[k] {
                let (a, c) = combo_from_index(j);
                let bits = (1u64 << a) | (1u64 << c);
                if disjoint && used & bits != 0 {
                    continue;
                }
                let w2 = w * reach[seats[k]][j] as f64;
                let (rj, rh) = (b.rank[j], b.rank[h]);
                let (ties2, beats2) = if k < live { (ties + (rj == rh) as usize, beats && rj <= rh) } else { (ties, beats) };
                rec(b, h, reach, seats, live, combos, disjoint, used | bits, k + 1, w2, ties2, beats2, share, mass);
            }
        }
        rec(b, h, reach, &seats, live.len(), &combos, disjoint, 0, 0, 1.0, 0, true, &mut share, &mut mass);
        (share, mass)
    }

    /// Opponents with structured ranges so ties and blockers both occur.
    fn structured_reach(b: &Board) -> Vec<Vec<f32>> {
        let mut reach = vec![vec![0f32; NUM_COMBOS]; 4];
        for idx in 0..NUM_COMBOS {
            if !b.valid(idx) {
                continue;
            }
            let (a, c) = combo_from_index(idx);
            reach[1][idx] = if idx % 3 == 0 { 0.7 } else { 0.0 } + if a / 4 == c / 4 { 0.3 } else { 0.0 };
            reach[2][idx] = if idx % 5 < 2 { 0.5 } else { 0.1 };
            reach[3][idx] = if idx % 11 == 0 { 0.4 } else { 0.0 };
        }
        reach
    }

    fn check(b: &Board, out: &[f32], reach: &[Vec<f32>], live: &[usize], folded: &[usize], disjoint: bool) {
        let mut checked = 0;
        for h in (0..NUM_COMBOS).step_by(37) {
            if !b.valid(h) {
                continue;
            }
            let (share, mass) = brute(b, h, reach, live, folded, disjoint);
            let expect = share * 100.0 - 10.0 * mass;
            assert!((out[h] as f64 - expect).abs() < 1e-2 * expect.abs().max(1.0), "combo {h}: {} vs {expect}", out[h]);
            checked += 1;
        }
        assert!(checked > 20);
    }

    #[test]
    fn two_live_seats_match_brute_force_with_disjoint_hands() {
        let b = board("Ks7h2d9c9s");
        let reach = structured_reach(&b);
        let mut out = vec![0f32; NUM_COMBOS];
        showdown(std::slice::from_ref(&b), 0, 0, &[1, 2], &[], &reach, 100.0, 10.0, &mut out);
        check(&b, &out, &reach, &[1, 2], &[], true);
        // The independent model is measurably different: the test has teeth.
        let (share, mass) = brute(&b, 5 * 37, &reach, &[1, 2], &[], false);
        let independent = share * 100.0 - 10.0 * mass;
        assert!((out[5 * 37] as f64 - independent).abs() > 1e-2 * independent.abs());
    }

    #[test]
    fn a_live_and_a_folded_seat_match_brute_force_with_disjoint_hands() {
        let b = board("Ks7h2d9c9s");
        let reach = structured_reach(&b);
        let mut out = vec![0f32; NUM_COMBOS];
        showdown(std::slice::from_ref(&b), 0, 0, &[1], &[2], &reach, 100.0, 10.0, &mut out);
        check(&b, &out, &reach, &[1], &[2], true);
    }

    #[test]
    fn three_other_seats_use_the_independent_model() {
        let b = board("Ks7h2d9c9s");
        let reach = structured_reach(&b);
        let mut out = vec![0f32; NUM_COMBOS];
        showdown(std::slice::from_ref(&b), 0, 0, &[1, 3], &[2], &reach, 100.0, 10.0, &mut out);
        check(&b, &out, &reach, &[1, 3], &[2], false);
    }

    #[test]
    fn compatible_masses_match_brute_force() {
        let b = board("Ks7h2d9c9s");
        let by_card = combos_by_card();
        let reach = structured_reach(&b);
        // One seat: the board-free helper on board-masked reach.
        let mut mass = vec![0f32; NUM_COMBOS];
        compatible_mass(&reach[1], &by_card, &mut mass);
        let (_, m) = brute(&b, 5, &reach, &[1], &[], true);
        assert!((mass[5] as f64 - m).abs() < 1e-3);
        // Two seats: pairs with no shared card.
        let mut pair = vec![0f64; NUM_COMBOS];
        compatible_pair_mass(&reach[1], &reach[2], &by_card, &mut pair);
        for h in (0..NUM_COMBOS).step_by(53) {
            if !b.valid(h) {
                continue;
            }
            let (_, m) = brute(&b, h, &reach, &[1, 2], &[], true);
            assert!((pair[h] - m).abs() < 1e-2 * m.max(1.0), "combo {h}: {} vs {m}", pair[h]);
        }
    }

    #[test]
    #[ignore]
    fn bench_showdown() {
        let b = board("Ks7h2d9c9s");
        let reach = structured_reach(&b);
        let mut out = vec![0f32; NUM_COMBOS];
        for (live, folded) in [(vec![1usize, 2usize], vec![]), (vec![1], vec![2]), (vec![1], vec![]), (vec![1, 2], vec![3])] {
            let n = 200;
            let start = std::time::Instant::now();
            for _ in 0..n {
                showdown(std::slice::from_ref(&b), 0, 0, &live, &folded, &reach, 100.0, 10.0, &mut out);
            }
            let per = start.elapsed().as_secs_f64() / n as f64;
            println!("live {live:?} folded {folded:?}: {:.1} us per board-hero, {:.2} us per hero combo", per * 1e6, per * 1e6 / b.order.len() as f64);
        }
        let start = std::time::Instant::now();
        for _ in 0..200 {
            std::hint::black_box(board("Ks7h2d9c9s"));
        }
        println!("Board::new: {:.1} us", start.elapsed().as_secs_f64() / 200.0 * 1e6);
    }

    #[test]
    fn sample_weights_are_the_deck_ratios() {
        // Heads-up turn: 47 cards the hero does not see, 45 nobody holds.
        assert!((sample_weight(47, 1, 1) - 47.0 / 45.0).abs() < 1e-12);
        // Three seats running the flop out: C(47, 2) / C(43, 2).
        assert!((sample_weight(47, 2, 2) - 1081.0 / 903.0).abs() < 1e-12);
        // A river under a turn, four seats: 46 / 40.
        assert!((sample_weight(46, 3, 1) - 46.0 / 40.0).abs() < 1e-12);
        // Nothing drawn, nothing to correct.
        assert_eq!(sample_weight(45, 2, 0), 1.0);
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
        let boards = [b];
        showdown(&boards, 0, 0, &[1], &[], &reach, 100.0, 10.0, &mut hu);
        showdown(&boards, 0, 0, &[1], &[2], &reach, 100.0, 10.0, &mut with_folded);
        // Whatever seat 1 holds, seat 2 holds any two of the 43 cards left: C(43, 2) = 903.
        for h in 0..NUM_COMBOS {
            if boards[0].valid(h) {
                assert!((with_folded[h] - hu[h] * 903.0).abs() < 1e-2 * hu[h].abs().max(1.0), "{h}: {} vs {}", with_folded[h], hu[h] * 903.0);
            }
        }
    }

    #[test]
    fn cards_fall_into_texture_classes() {
        let flop = parse_cards("Ks7h2d").unwrap();
        let class = |s: &str| card_class(&flop, parse_cards(s).unwrap()[0]);
        assert_eq!(CLASS_NAMES[class("Kc")], "pair");
        assert_eq!(CLASS_NAMES[class("2s")], "pair");
        assert_eq!(CLASS_NAMES[class("Ac")], "over");
        // No window of five ranks holds two of K, 7, 2: no turn card lets two hole cards make a straight.
        assert_eq!(CLASS_NAMES[class("Qc")], "blank");
        assert_eq!(CLASS_NAMES[class("5c")], "blank");
        assert_eq!(CLASS_NAMES[class("Tc")], "blank");
        let two_tone = parse_cards("Ks7s2d").unwrap();
        assert_eq!(CLASS_NAMES[card_class(&two_tone, parse_cards("3s").unwrap()[0])], "flush");
        assert_eq!(CLASS_NAMES[card_class(&two_tone, parse_cards("2s").unwrap()[0])], "flush"); // flush before pair
        let connected = parse_cards("9h8c2d").unwrap();
        assert_eq!(CLASS_NAMES[card_class(&connected, parse_cards("7s").unwrap()[0])], "straight");
        assert_eq!(CLASS_NAMES[card_class(&connected, parse_cards("Js").unwrap()[0])], "straight");
        assert_eq!(CLASS_NAMES[card_class(&connected, parse_cards("4s").unwrap()[0])], "blank");
        let four = parse_cards("Ks7h2d9c").unwrap();
        assert_eq!(CLASS_NAMES[card_class(&four, parse_cards("8c").unwrap()[0])], "straight"); // 9 8 7 + two holes
        assert_eq!(CLASS_NAMES[card_class(&four, parse_cards("9d").unwrap()[0])], "pair");
        let wheel = parse_cards("Ah3c8d").unwrap();
        assert_eq!(CLASS_NAMES[card_class(&wheel, parse_cards("4s").unwrap()[0])], "straight"); // A 3 4 + 2 5
    }

    #[test]
    fn samples_cover_every_class_in_proportion_and_weights_sum_to_one() {
        let flop = parse_cards("Ks7s2d").unwrap();
        let (cards, weights) = sample_cards(&flop, 8);
        assert_eq!(cards.len(), 8);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for c in &cards {
            assert!(!flop.contains(c));
        }
        let mut seen = [0usize; CLASS_NAMES.len()];
        for c in &cards {
            seen[card_class(&flop, *c)] += 1;
        }
        // flush (11 spades), pair (8), over (3 aces), blank: every class present in the deck is sampled
        let present: Vec<bool> = (0..CLASS_NAMES.len()).map(|k| (0..52u8).any(|c| !flop.contains(&c) && card_class(&flop, c) == k)).collect();
        assert_eq!(present, [true, true, false, true, true]);
        for (k, &n) in seen.iter().enumerate() {
            assert_eq!(n >= 1, present[k], "{seen:?}");
        }
        // a class's weight is its share of the 49 cards
        let mut by_class = [0f32; CLASS_NAMES.len()];
        for (c, w) in cards.iter().zip(&weights) {
            by_class[card_class(&flop, *c)] += w;
        }
        let flush = (0..52u8).filter(|c| !flop.contains(c) && card_class(&flop, *c) == 0).count();
        assert!((by_class[0] - flush as f32 / 49.0).abs() < 1e-5, "{by_class:?}");
        // deterministic, and the whole deck when asked for enough
        assert_eq!(sample_cards(&flop, 8).0, cards);
        let (all, w) = sample_cards(&flop, 49);
        assert_eq!(all.len(), 49);
        assert!((w[0] - 1.0 / 49.0).abs() < 1e-7);
        // two-card runouts multiply the weights
        let runouts = sample_runouts(&flop, 4, 3);
        assert_eq!(runouts.len(), 12);
        assert!((runouts.iter().map(|r| r.1).sum::<f32>() - 1.0).abs() < 1e-5);
        for (r, _) in &runouts {
            assert!(!flop.contains(&r[3]) && !flop.contains(&r[4]) && r[3] != r[4]);
        }
    }

    /// A card asked for stands in for a sampled card of its class, weights
    /// untouched; when its class was not sampled it joins with the class's
    /// share of the deck and the rest shrink to keep the sum at one.
    #[test]
    fn fixed_cards_join_the_sample_in_their_class() {
        let flop = parse_cards("Ks7s2d").unwrap();
        let (cards, weights) = sample_cards(&flop, 8);
        let qh = parse_cards("Qh").unwrap()[0];
        assert!(!cards.contains(&qh));
        let (with, w) = sample_cards_with(&flop, 8, &[qh]);
        assert_eq!(with.len(), 8);
        assert!(with.contains(&qh));
        assert_eq!(w, weights);
        let swapped = cards.iter().zip(&with).filter(|(a, b)| a != b).count();
        assert_eq!(swapped, 1);
        let out = cards.iter().zip(&with).find(|(a, b)| a != b).unwrap().0;
        assert_eq!(card_class(&flop, *out), card_class(&flop, qh));
        // already there: nothing changes
        assert_eq!(sample_cards_with(&flop, 8, &[cards[0]]).0, cards);
        // two samples take the two biggest classes; an ace (class "over") has to join
        let (two, _) = sample_cards(&flop, 2);
        let ah = parse_cards("Ah").unwrap()[0];
        assert!(two.iter().all(|&c| card_class(&flop, c) != card_class(&flop, ah)));
        let (three, w3) = sample_cards_with(&flop, 2, &[ah]);
        assert_eq!(three.len(), 3);
        assert_eq!(three[2], ah);
        assert!((w3.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{w3:?}");
        let size = |c: Card| (0..52u8).filter(|d| !flop.contains(d) && card_class(&flop, *d) == card_class(&flop, c)).count();
        let covered: usize = three.iter().map(|&c| size(c)).sum();
        // three aces: the ace of spades is a flush card here
        assert_eq!(size(ah), 3);
        assert!((w3[2] - 3.0 / covered as f32).abs() < 1e-6, "{w3:?}: three aces over {covered}");
    }
}
