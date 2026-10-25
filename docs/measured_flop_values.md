# Measured flop values for the preflop solver

The preflop engine never plays the flop. Where the betting closes heads-up with
chips behind, it hands each seat a share of the pot from its equity against the
other seat's range and a per-class realization weight (`realization` =
"balanced" / "calibrated" / "static"). That model cannot bet a hand off its
equity, so it over-rewards continuing with medium hands: a 4-bet caller with 66
keeps its 31% of a 46 bb pot where a real flop costs it the pot on most boards.
Measured on the 6-max 100 bb baseline (BTN opens, BB 3-bets, BTN 4-bets, BB
calls: 46.5 bb pot, 77 behind, 60 flops with the postflop solver), the model
gave the calling range +2.35 bb/hand where the real flop gives −2.52; AKo +4.25
vs −3.64, 99 +0.26 vs −6.89. The 3-bettor therefore defends 4-bets far too often
(folds 21–37% where the 4-bet needs 61%), 4-bet ranges become value-only (no
A5s), and 3-bet ranges linear.

`crates/solver/src/preflop/leaf.rs` lets a solve price those terminals with
values the postflop solver measured.

## The table

A `LeafValuesFile` (JSON) lists terminals by action path from the root:

```json
{"name": "6max-100bb … leaves r2",
 "source": {"flops": 48, "tree": "...", "iterations": 80},
 "terminals": [
   {"path": [0,0,0,2,0,2,2,1],
    "values": {"BTN": [169 numbers], "BB": [169 numbers]}}],
 "fallback": [
   {"role": "oop", "aggressor": false, "spr_max": 2.5, "ratio": 0.71}, …]}
```

`values[POS][class]` is the chips (bb) the seat expects to collect from the
flop on with that hand class — the postflop solver's root EV in pot-share
convention (chips owned of the final pot, future bets and rake inside),
averaged over sampled flops against the ranges the preflop solve sent to the
terminal, with every class present (a floor weight for the classes the
strategy never sends there, so the value a deviation would find is measured
too). Class order is the engine's (`class_parts()` in `equity.rs`). The engine
prices a covered terminal for seat `p` as

    out[h] = reach_others × (values[p][h] − invested[p])

which is exactly where the model put `pot_eff × share[h] − invested[p]`.

Heads-up terminals the table does not cover (below the measuring run's reach
cutoff) are priced with the `fallback` realization measured over the covered
ones: equity against the opponent's arriving range × the raked pot × the cell
for the seat's role (first to act on the flop = "oop"), whether it was the
last preflop raiser, and SPR (the cell with the smallest `spr_max` at or above
the terminal's SPR; above every edge, the largest). All-in terminals keep raw
equity (no flop to play); multiway terminals keep the coupled-deck model.

## Engine, saves, API

- `PreflopSolver::set_leaf_values(Option<LeafValuesFile>)` validates every path
  (a heads-up `POT_SHARE` node, both live positions present, 169 finite
  values, no duplicates) and installs or clears the table. Payoffs change, so
  it restarts learning (`reset_learning`), like a table change does.
- `terminal_value` consults the table before the realization models;
  `continuation()` reports `model: "measured"` / `"measured_fallback"`.
- The GPU kernels implement the realization model themselves, so a game with
  a table refuses the GPU (`PreflopGpu` constructors return an error) and the
  server solves it on the CPU, with the reason in `gpu_note`.
- Saves carry the table in the header and use magic `GTOPREFLOP5`; older
  binaries refuse them rather than continue the solve against model payoffs.
  Loading installs the table before the arenas, so the learned strategy is
  kept.
- `POST /api/preflop/leaf-values` with the file (or `null` to clear) installs
  it into the live session (409 while a solve runs; the session returns to
  idle at iteration 0). `status`, `session` and `load` carry
  `leaf_values: {name, terminals, fallback_cells} | null`.

## Measuring, and why it iterates

The values depend on the ranges that reach each terminal, and those ranges
move with the strategy the values produce. So the table is a fixed point:

1. Solve preflop (round 0: the equity-realization model).
2. For every heads-up flop terminal the solve reaches, send both arriving
   ranges (with the floor) to the postflop solver on K sampled flops; average
   the root EVs per class; write the table. A fixed, systematic sample over the
   1,755 suit-isomorphic flops, shared by all terminals and rounds, keeps the
   rounds comparable (common random numbers).
3. Install the table, re-solve preflop, compare the strategy with the previous
   round (3-bet/4-bet structure, fold-to-4-bet, first-in ranges).
4. Repeat until the strategy stops moving; damp (`V_used = ½ old + ½ new`) if
   it oscillates.

The app's driver is `scripts/gpu-calibrate.py` in lucky-chances (Modal, one
postflop solve at a time per container, many containers).

## Card removal

The same solve can switch the tree's chance model from independent classes
to first-order card removal (`card_removal: true` in the config,
`removal.rs`): every other seat's reaching mass is weighted, per hero class,
by how many of its combos the hero's cards leave in the deck, so BB's AA is
half as likely against a BTN holding A5s and a sixth as likely against AA.
That is what makes an ace the card to 4-bet bluff with. The weights sum to
one over the class priors, so untouched ranges still reach with probability
one and the seats' values still add up to the pot. Removal among the
opponents themselves stays unmodelled.

## Limits

- Flop sampling noise: K flops per terminal give per-class values with a few
  percent of the pot of noise; the range-level numbers are far tighter.
- The measuring tree is smaller than a full postflop game (one bet size per
  street with raises, leads and jams); values are a touch low for the
  aggressor relative to a wide sizing menu.
- Only heads-up terminals are measured. Multiway flops keep the coupled-deck
  equity model.
