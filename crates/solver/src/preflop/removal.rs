//! Card removal between the hero's hand class and each opponent's, for the
//! 169-class preflop tree.
//!
//! The tree deals every seat a class independently: the chance that BB holds
//! AA is the same whether BTN holds A5s or 76s. In the deck it is half. So
//! without removal the solver cannot see why an ace is the card to 4-bet
//! bluff with, and it overestimates how often a premium is out against a
//! premium.
//!
//! `Removal` holds, for hero class h and opponent class j,
//!
//! ```text
//! R[h][j] = P(opponent holds j | hero holds h) / P(opponent holds j)
//!         = (1326 / 1225) · compatible(h, j) / combos(j)
//! ```
//!
//! where compatible(h, j) counts the combos of j sharing no card with a combo
//! of h (the same for every combo of h, by suit symmetry). Summed over j with
//! the class priors it is exactly 1, so an untouched range still reaches with
//! probability 1, and R is symmetric. `terminal_value()` weights every other
//! seat's reach by it from the hero's point of view: the first-order
//! correction. Removal among the opponents themselves (one holding an ace
//! makes another's AA rarer) stays unmodelled, as it is in the multiway
//! equity model.

use super::equity::{class_combos, class_parts, EquityTable, NUM_CLASSES};

pub struct Removal {
    /// r[h * 169 + j] = R[h][j] (= R[j][h]).
    r: Vec<f32>,
    /// er[j * 169 + h] = R[h][j] · eq(h, j): opponent-major, so one pass over
    /// an opponent's reach vector yields every hero class's equity mass.
    er: Vec<f32>,
}

/// The combos of a class as (rank, suit) card pairs.
fn combos(class: usize) -> Vec<[(u8, u8); 2]> {
    let (hi, lo, suited) = class_parts(class);
    let mut out = Vec::new();
    for s1 in 0..4u8 {
        for s2 in 0..4u8 {
            let ok = if hi == lo { s1 < s2 } else if suited { s1 == s2 } else { s1 != s2 };
            if ok {
                out.push([(hi, s1), (lo, s2)]);
            }
        }
    }
    debug_assert_eq!(out.len() as f32, class_combos(class));
    out
}

impl Removal {
    pub fn new(eq: &EquityTable) -> Self {
        let mut r = vec![0f32; NUM_CLASSES * NUM_CLASSES];
        for h in 0..NUM_CLASSES {
            let hero = combos(h)[0];
            for j in 0..NUM_CLASSES {
                let compatible = combos(j)
                    .iter()
                    .filter(|c| !c.iter().any(|card| hero.contains(card)))
                    .count() as f32;
                r[h * NUM_CLASSES + j] = (1326.0 / 1225.0) * compatible / class_combos(j);
            }
        }
        let mut er = vec![0f32; NUM_CLASSES * NUM_CLASSES];
        for h in 0..NUM_CLASSES {
            for j in 0..NUM_CLASSES {
                er[j * NUM_CLASSES + h] = r[h * NUM_CLASSES + j] * eq.eq(h, j);
            }
        }
        Self { r, er }
    }

    #[inline]
    pub fn factor(&self, hero: usize, opponent: usize) -> f32 {
        self.r[hero * NUM_CLASSES + opponent]
    }

    /// For one opponent's reach vector: the mass that is compatible with each
    /// hero class (`mass[h]` = Σ_j reach[j]·R[h][j]) and its equity-weighted
    /// mass (`eq_mass[h]` = Σ_j reach[j]·R[h][j]·eq(h, j)), so the hero's
    /// equity against the opponent's reaching range is eq_mass[h] / mass[h].
    pub fn masses(&self, reach: &[f32], mass: &mut [f32; NUM_CLASSES], eq_mass: &mut [f32; NUM_CLASSES]) {
        mass.fill(0.0);
        eq_mass.fill(0.0);
        for (j, &d) in reach.iter().enumerate() {
            if d == 0.0 {
                continue;
            }
            let r = &self.r[j * NUM_CLASSES..(j + 1) * NUM_CLASSES];
            let er = &self.er[j * NUM_CLASSES..(j + 1) * NUM_CLASSES];
            for h in 0..NUM_CLASSES {
                mass[h] += r[h] * d;
                eq_mass[h] += er[h] * d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::equity::{class_index, class_prob};
    use super::*;

    fn table() -> Removal {
        Removal::new(&EquityTable::build(8))
    }

    #[test]
    fn factors_count_the_deck() {
        let t = table();
        let aa = class_index(12, 12, false);
        let a5s = class_index(12, 3, true);
        let ako = class_index(12, 11, false);
        let x72 = class_index(5, 0, false);
        let scale = 1326.0 / 1225.0;
        // AA against AA: one of the six combos is left
        assert!((t.factor(aa, aa) - scale / 6.0).abs() < 1e-6);
        // an ace blocks half of AA and a quarter of AKo
        assert!((t.factor(a5s, aa) - scale * 3.0 / 6.0).abs() < 1e-6);
        assert!((t.factor(a5s, ako) - scale * 9.0 / 12.0).abs() < 1e-6);
        // 72o blocks nothing of AA: a 50-card deck makes it slightly likelier
        assert!((t.factor(x72, aa) - scale).abs() < 1e-6);
        for h in 0..NUM_CLASSES {
            let total: f32 = (0..NUM_CLASSES).map(|j| class_prob(j) * t.factor(h, j)).sum();
            assert!((total - 1.0).abs() < 1e-4, "{h}: {total}");
            for j in 0..NUM_CLASSES {
                assert!((t.factor(h, j) - t.factor(j, h)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn masses_match_the_scalar_sums() {
        let t = table();
        let eq = EquityTable::build(8);
        let reach: Vec<f32> = (0..NUM_CLASSES).map(|j| class_prob(j) * (0.2 + (j % 7) as f32 * 0.1)).collect();
        let (mut mass, mut eq_mass) = ([0f32; NUM_CLASSES], [0f32; NUM_CLASSES]);
        t.masses(&reach, &mut mass, &mut eq_mass);
        for h in [0, 40, 168] {
            let expect_mass: f32 = (0..NUM_CLASSES).map(|j| reach[j] * t.factor(h, j)).sum();
            let expect_eq: f32 = (0..NUM_CLASSES).map(|j| reach[j] * t.factor(h, j) * eq.eq(h, j)).sum();
            assert!((mass[h] - expect_mass).abs() < 1e-5);
            assert!((eq_mass[h] - expect_eq).abs() < 1e-5);
        }
    }
}
