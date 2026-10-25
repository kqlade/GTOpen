//! Flop values measured by the postflop solver, in place of the
//! equity-realization model at the heads-up flop terminals they cover.
//!
//! The preflop engine never plays the flop: where the betting closes with
//! chips behind, it hands each seat a share of the pot from its equity and a
//! per-class realization weight. That model cannot bet a hand off its equity,
//! so it over-rewards calling a raise with a medium hand (a 4-bet caller with
//! 66 keeps its 31% of the pot where a real flop costs it the pot on most
//! boards) and the solver over-defends, under-bluffs and 3-bets linear.
//!
//! A `LeafValuesFile` carries, per terminal (an action path from the root),
//! the chips each seat expects to collect from the flop on — the postflop
//! solver's root EV per hand class, averaged over sampled flops, with future
//! bets and rake inside — measured against the ranges the preflop solve sent
//! there. The engine prices a covered terminal as `value[class] − invested`,
//! and prices the heads-up terminals the table does not cover with the
//! fallback realization measured over the covered ones (equity × raked pot ×
//! the ratio for the seat's role, pot type and SPR band). Values are fixed
//! for a solve; since the ranges that reach a terminal move with the strategy,
//! the table is re-measured against the new solve and the solve repeated
//! until the two agree. See docs/measured_flop_values.md.

use super::equity::NUM_CLASSES;
use super::{PNode, PreflopSolver, KIND_ACTION, KIND_POT_SHARE};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafValuesFile {
    /// Provenance shown in status and saves (the measuring run).
    pub name: String,
    /// Free-form details of the measurement (flops, tree, iterations...).
    #[serde(default)]
    pub source: serde_json::Value,
    pub terminals: Vec<LeafTerminalFile>,
    #[serde(default)]
    pub fallback: Vec<LeafFallbackCell>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafTerminalFile {
    /// Action indices from the root to a heads-up flop terminal.
    pub path: Vec<usize>,
    /// Position name → 169 values (chips collected from the flop on, net of
    /// rake, in bb), for both live seats.
    pub values: HashMap<String, Vec<f32>>,
}

/// Realization for an uncovered heads-up flop terminal: a seat in `role`
/// ("oop" | "ip"), the preflop aggressor or not, at an SPR up to `spr_max`
/// (the cell with the smallest `spr_max` at or above the terminal's SPR
/// applies; above every edge, the largest), collects `ratio` × equity × pot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafFallbackCell {
    pub role: String,
    pub aggressor: bool,
    pub spr_max: f64,
    pub ratio: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LeafValuesSummary {
    pub name: String,
    pub terminals: usize,
    pub fallback_cells: usize,
}

pub struct LeafValues {
    pub file: LeafValuesFile,
    /// Per node: the covering terminal's values per seat (empty for dead seats).
    by_node: HashMap<u32, Vec<Vec<f32>>>,
}

impl LeafValues {
    pub fn summary(&self) -> LeafValuesSummary {
        LeafValuesSummary {
            name: self.file.name.clone(),
            terminals: self.by_node.len(),
            fallback_cells: self.file.fallback.len(),
        }
    }

    pub fn covers(&self, node: usize) -> bool {
        self.by_node.contains_key(&(node as u32))
    }

    pub(super) fn values(&self, node: usize) -> Option<&Vec<Vec<f32>>> {
        self.by_node.get(&(node as u32))
    }

    pub(super) fn fallback_ratio(&self, oop: bool, aggressor: bool, spr: f64) -> Option<f32> {
        let role = if oop { "oop" } else { "ip" };
        let cells = self
            .file
            .fallback
            .iter()
            .filter(|c| c.role == role && c.aggressor == aggressor);
        let mut best: Option<&LeafFallbackCell> = None;
        for c in cells {
            best = Some(match best {
                None => c,
                Some(b) => {
                    let (b_ok, c_ok) = (b.spr_max >= spr, c.spr_max >= spr);
                    match (b_ok, c_ok) {
                        (true, true) => if c.spr_max < b.spr_max { c } else { b },
                        (false, true) => c,
                        (true, false) => b,
                        (false, false) => if c.spr_max > b.spr_max { c } else { b },
                    }
                }
            });
        }
        best.map(|c| c.ratio)
    }
}

impl PreflopSolver {
    /// The node an action path leads to, following children without touching
    /// the strategy (walk() also multiplies reaches).
    pub(super) fn node_at(&self, path: &[usize]) -> Result<usize, String> {
        let mut node = 0usize;
        for &a in path {
            let nd = &self.nodes[node];
            if nd.kind != KIND_ACTION || a >= nd.actions.len() {
                return Err(format!("path {path:?} leaves the tree"));
            }
            node = self.children[nd.child_start as usize + a] as usize;
        }
        Ok(node)
    }

    /// The action path from the root to `target` (a depth-first search; the
    /// tree keeps no parent links).
    #[cfg(test)]
    pub(super) fn path_to(&self, target: usize) -> Vec<usize> {
        fn dfs(s: &PreflopSolver, node: usize, target: usize, path: &mut Vec<usize>) -> bool {
            if node == target {
                return true;
            }
            let nd = &s.nodes[node];
            if nd.kind != KIND_ACTION {
                return false;
            }
            for a in 0..nd.actions.len() {
                path.push(a);
                if dfs(s, s.children[nd.child_start as usize + a] as usize, target, path) {
                    return true;
                }
                path.pop();
            }
            false
        }
        let mut path = Vec::new();
        assert!(dfs(self, 0, target, &mut path), "node {target} is not in the tree");
        path
    }

    /// Price covered heads-up flop terminals from `file` (None clears). The
    /// payoffs change, so learning restarts: a solve must never continue
    /// against another terminal game.
    pub fn set_leaf_values(&mut self, file: Option<LeafValuesFile>) -> Result<Option<LeafValuesSummary>, String> {
        let Some(file) = file else {
            if self.leaf.take().is_some() {
                self.reset_learning();
            }
            return Ok(None);
        };
        if file.name.trim().is_empty() {
            return Err("leaf values need a name".into());
        }
        let mut by_node: HashMap<u32, Vec<Vec<f32>>> = HashMap::new();
        for t in &file.terminals {
            let node = self.node_at(&t.path)?;
            let nd: &PNode = &self.nodes[node];
            if nd.kind != KIND_POT_SHARE || nd.live.count_ones() != 2 {
                return Err(format!("path {:?} is not a heads-up flop terminal", t.path));
            }
            let mut per_seat = vec![Vec::new(); self.n];
            for seat in 0..self.n {
                if nd.live & (1 << seat) == 0 {
                    continue;
                }
                let pos = &self.cfg.positions[seat];
                let v = t.values.get(pos).ok_or_else(|| {
                    format!("path {:?}: no values for {pos}, who sees this flop", t.path)
                })?;
                if v.len() != NUM_CLASSES {
                    return Err(format!("path {:?}: {pos} has {} values, not {NUM_CLASSES}", t.path, v.len()));
                }
                if let Some(bad) = v.iter().find(|x| !x.is_finite()) {
                    return Err(format!("path {:?}: {pos} has a non-finite value {bad}", t.path));
                }
                per_seat[seat] = v.clone();
            }
            for pos in t.values.keys() {
                let seat = self.cfg.positions.iter().position(|p| p == pos);
                if !seat.is_some_and(|s| nd.live & (1 << s) != 0) {
                    return Err(format!("path {:?}: {pos} does not see this flop", t.path));
                }
            }
            if by_node.insert(node as u32, per_seat).is_some() {
                return Err(format!("path {:?} appears twice", t.path));
            }
        }
        for c in &file.fallback {
            if c.role != "oop" && c.role != "ip" {
                return Err(format!("fallback role {:?} is not oop or ip", c.role));
            }
            if !(c.spr_max > 0.0) || !c.ratio.is_finite() || c.ratio <= 0.0 {
                return Err("fallback cells need spr_max > 0 and a positive finite ratio".into());
            }
        }
        self.leaf = Some(LeafValues { file, by_node });
        self.reset_learning();
        Ok(self.leaf.as_ref().map(|l| l.summary()))
    }

    pub fn leaf_values(&self) -> Option<&LeafValues> {
        self.leaf.as_ref()
    }

    /// For an uncovered heads-up flop terminal: the fallback realization of
    /// seat `p` there, if the table has a cell for its role, initiative and SPR.
    pub(super) fn leaf_fallback(&self, node: usize, p: usize) -> Option<f32> {
        let leaf = self.leaf.as_ref()?;
        let nd = &self.nodes[node];
        let order = self.postflop_order();
        let first = order.iter().copied().find(|&s| nd.live & (1 << s) != 0)?;
        let mut min_left = f64::MAX;
        for i in 0..self.n {
            if nd.live & (1 << i) != 0 {
                min_left = min_left.min(self.cfg.stack - nd.invested[i] + self.cfg.ante);
            }
        }
        let spr = (min_left / nd.pot).max(0.0);
        if spr <= 1e-9 {
            return None; // all-in: no flop to play, raw equity is right
        }
        leaf.fallback_ratio(first == p, nd.aggressor as usize == p, spr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::equity::EquityTable;
    use super::super::{PreflopConfig, PreflopSolver, KIND_POT_SHARE};
    use super::*;
    use std::sync::{Arc, OnceLock};

    fn fixture() -> PreflopSolver {
        static EQ: OnceLock<Arc<EquityTable>> = OnceLock::new();
        let cfg = PreflopConfig {
            utg_straddle: false,
            positions: vec!["BTN".into(), "SB".into(), "BB".into()],
            stack: 20.0, posts: vec![0.0, 0.5, 1.0], ante: 0.0,
            limp: false, open_raises: vec![2.5], raise_mults: vec![3.0],
            max_raises: 2, add_allin: false, allin_threshold: 0.85,
            rake_pct: 5.0, rake_cap: 1.0, no_flop_no_drop: true,
            realization: "balanced".into(), call_only_seats: vec![],
            open_raises_by_seat: None, fourbet_mults: None, fourbet_mults_by_seat: None, raise_mults_by_seat: None,
        };
        PreflopSolver::new(cfg, EQ.get_or_init(|| Arc::new(EquityTable::build(8))).clone()).unwrap()
    }

    /// BTN raises, SB folds, BB calls: a heads-up flop between BB (OOP) and BTN.
    fn hu_flop_path(s: &PreflopSolver) -> Vec<usize> {
        let root = &s.nodes[0];
        let raise = root.actions.iter().position(|a| a.kind == "raise").unwrap();
        let sb = s.node_at(&[raise]).unwrap();
        let fold = s.nodes[sb].actions.iter().position(|a| a.kind == "fold").unwrap();
        let bb = s.node_at(&[raise, fold]).unwrap();
        let call = s.nodes[bb].actions.iter().position(|a| a.kind == "call").unwrap();
        let path = vec![raise, fold, call];
        let node = s.node_at(&path).unwrap();
        assert_eq!(s.nodes[node].kind, KIND_POT_SHARE);
        assert_eq!(s.nodes[node].live.count_ones(), 2);
        path
    }

    fn file(path: Vec<usize>, bb: f32, btn: f32) -> LeafValuesFile {
        LeafValuesFile {
            name: "test".into(),
            source: serde_json::Value::Null,
            terminals: vec![LeafTerminalFile {
                path,
                values: HashMap::from([("BB".to_string(), vec![bb; NUM_CLASSES]), ("BTN".to_string(), vec![btn; NUM_CLASSES])]),
            }],
            fallback: vec![
                LeafFallbackCell { role: "oop".into(), aggressor: false, spr_max: 2.5, ratio: 0.7 },
                LeafFallbackCell { role: "oop".into(), aggressor: false, spr_max: 8.0, ratio: 0.8 },
                LeafFallbackCell { role: "ip".into(), aggressor: true, spr_max: 8.0, ratio: 1.1 },
            ],
        }
    }

    #[test]
    fn covered_terminal_pays_the_measured_value_less_the_investment() {
        let mut s = fixture();
        let path = hu_flop_path(&s);
        let node = s.node_at(&path).unwrap();
        let summary = s.set_leaf_values(Some(file(path.clone(), 3.0, 2.0))).unwrap().unwrap();
        assert_eq!((summary.terminals, summary.fallback_cells), (1, 3));
        assert!(s.leaf_values().unwrap().covers(node));
        let nd = &s.nodes[node];
        let (bb, btn) = (2usize, 0usize);
        // every class of both other seats (the folded SB included) reaching
        // with weight 1: the path probability factor is 169 × 169
        let reaches: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0; NUM_CLASSES]).collect();
        let prob = (NUM_CLASSES * NUM_CLASSES) as f32;
        let mut out = vec![0f32; NUM_CLASSES];
        s.terminal_value(node, bb, &reaches, &mut out);
        let expected_bb = prob * (3.0 - nd.invested[bb]) as f32;
        assert!((out[0] - expected_bb).abs() < 1e-2, "{} vs {expected_bb}", out[0]);
        assert!(out.iter().all(|v| (v - expected_bb).abs() < 1e-2));
        s.terminal_value(node, btn, &reaches, &mut out);
        let expected_btn = prob * (2.0 - nd.invested[btn]) as f32;
        assert!((out[5] - expected_btn).abs() < 1e-2, "{} vs {expected_btn}", out[5]);
    }

    #[test]
    fn uncovered_terminals_use_the_fallback_ratio_by_role_and_spr() {
        let mut s = fixture();
        let path = hu_flop_path(&s);
        s.set_leaf_values(Some(file(path.clone(), 3.0, 2.0))).unwrap();
        // the same flop, but SB calls instead of BB: an uncovered terminal
        let root = &s.nodes[0];
        let raise = root.actions.iter().position(|a| a.kind == "raise").unwrap();
        let sb = s.node_at(&[raise]).unwrap();
        let call = s.nodes[sb].actions.iter().position(|a| a.kind == "call").unwrap();
        let bb_node = s.node_at(&[raise, call]).unwrap();
        let fold = s.nodes[bb_node].actions.iter().position(|a| a.kind == "fold").unwrap();
        let other = s.node_at(&[raise, call, fold]).unwrap();
        assert_eq!(s.nodes[other].kind, KIND_POT_SHARE);
        assert!(!s.leaf_values().unwrap().covers(other));
        // SB: OOP, not the aggressor; SPR = (20 - 2.5) / 6 ≈ 2.9 → the 8.0 cell
        assert_eq!(s.leaf_fallback(other, 1), Some(0.8));
        // BTN: IP and the aggressor
        assert_eq!(s.leaf_fallback(other, 0), Some(1.1));
        let lv = s.leaf_values().unwrap();
        assert_eq!(lv.fallback_ratio(true, false, 1.0), Some(0.7));
        assert_eq!(lv.fallback_ratio(true, false, 50.0), Some(0.8));
        assert_eq!(lv.fallback_ratio(false, false, 3.0), None);
    }

    #[test]
    fn rejects_paths_that_are_not_heads_up_flops_and_seats_that_do_not_see_them() {
        let mut s = fixture();
        let path = hu_flop_path(&s);
        let err = s.set_leaf_values(Some(file(vec![], 1.0, 1.0))).unwrap_err();
        assert!(err.contains("not a heads-up flop terminal"), "{err}");
        let mut f = file(path.clone(), 1.0, 1.0);
        f.terminals[0].values.insert("SB".into(), vec![0.0; NUM_CLASSES]);
        let err = s.set_leaf_values(Some(f)).unwrap_err();
        assert!(err.contains("SB does not see this flop"), "{err}");
        let mut f = file(path.clone(), 1.0, 1.0);
        f.terminals[0].values.remove("BTN");
        let err = s.set_leaf_values(Some(f)).unwrap_err();
        assert!(err.contains("no values for BTN"), "{err}");
        let mut f = file(path.clone(), 1.0, 1.0);
        f.terminals[0].values.get_mut("BB").unwrap()[3] = f32::NAN;
        assert!(s.set_leaf_values(Some(f)).unwrap_err().contains("non-finite"));
        let mut f = file(path, 1.0, 1.0);
        f.terminals.push(f.terminals[0].clone());
        assert!(s.set_leaf_values(Some(f)).unwrap_err().contains("appears twice"));
        assert!(s.leaf_values().is_none(), "a rejected table must not be installed");
    }

    #[test]
    fn setting_or_clearing_the_table_restarts_learning() {
        let mut s = fixture();
        let path = hu_flop_path(&s);
        s.iteration = 12;
        s.set_leaf_values(Some(file(path, 1.0, 1.0))).unwrap();
        assert_eq!(s.iteration, 0);
        s.iteration = 7;
        assert!(s.set_leaf_values(None).unwrap().is_none());
        assert_eq!(s.iteration, 0);
        assert!(s.leaf_values().is_none());
    }
}
