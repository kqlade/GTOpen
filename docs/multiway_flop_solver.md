# Multiway flop solver

`crates/solver/src/multiway_flop/` solves the flop with three or more players
so the preflop engine can price its multiway terminals with measured values,
the way `docs/measured_flop_values.md` prices the heads-up ones. The postflop
engine proper (`tree.rs`, `cfr.rs`, the GPU kernels) is two-player throughout;
this is a separate, smaller, CPU-only solver built for measuring, not for
browsing a game tree.

## The game it solves

- N live players (3 to 6) see a flop with the pot and each seat's preflop
  investment the preflop terminal reports. Everyone who is live matched the
  last bet, so every seat has the same stack behind: no side pots.
- Hands are the 1,326 two-card combos per seat, weighted by the preflop reach
  of the hand class (the weight every combo of the class takes the line with),
  minus the combos the board blocks. Card removal between the hero and each
  opponent is exact; the opponents' cards are treated as independent of each
  other (a hand both opponents could hold is counted for both). That is the
  one approximation in the deck model; the overlapping hands it counts are
  correlated, which lifts every seat's share by about one percent of the pot
  (three uniform ranges checking down every runout own 2.53 bb each of a
  7.5 bb pot).
- Betting follows the postflop order (blinds first, then seat order). Actions:
  check, bet (pot fractions), fold, call, raise (multiples of the bet faced),
  all-in whenever a size would commit more than `allin_threshold` of the
  stack, at most `max_raises` raises per street.
- The tree covers the flop; `turn_cards > 0` adds a turn street on that many
  sampled turn cards (0 keeps the tree one street). Every way the flop
  betting can close with chips behind grows its own turn subtree per card, so
  a turn street multiplies the work by a few hundred; it is there for
  checking the one-street values on a spot, not for measuring.
- Leaves: a fold-out pays the raked pot to the last live seat; a street that
  closes with two or more seats live is checked down, each seat paid its share
  of the raked pot by exact showdown over a fixed systematic sample of
  `runouts` board completions (ties split the pot between the tying seats).
  There is no betting after the last modelled street, so hands that want to
  see every card cheaply are a little overvalued, and the aggressor's barrels
  are missing. The flop betting itself, position, fold equity and card removal
  are real.

## Algorithm

Vector-form discounted CFR (α 1.5, β 0, γ 2), alternating updates, every seat's
strategy a 1,326-vector per action node. The showdown leaves sort each
runout's combos by rank once per solve; per leaf and runout, prefix sums over
each opponent's reach in rank order (with the two per-card corrections for the
hero's cards) give every hero combo its chance of beating, or tying, each
opponent, and the subsets of tying opponents give the share.

Every `check_every` iterations each seat's best response to the others'
average strategies is computed; the gap is the sum over seats of
(best-response EV − average-strategy EV) as a percentage of the starting pot.
With more than two players CFR is not guaranteed to converge to an
equilibrium; in practice the gap falls steadily on these trees, and the
measured values are what the preflop engine needs, not a certificate.

## Output

Per seat, per hand class: the chips the seat owns at the end of the hand —
share of the final pot, future bets and rake inside — averaged over the
class's combos weighted by reach × the opponents' compatible mass, exactly the
pot-share convention of the heads-up measurement, so the preflop engine prices
a covered multiway terminal for seat `p` as

    out[h] = reach_others[h] × (values[p][h] − invested[p])

Also returned: per-class masses, each seat's range EV, the gap, node count and
timing.

## API

`POST /api/multiway-flop/solve` (synchronous and stateless; the caller sets
its own timeout, and several can run at once, one thread each):

```json
{"board": "Ks7h2d",
 "pot": 7.5, "stack": 97.5, "rake_pct": 5, "rake_cap": 8.0,
 "seats": [{"position": "SB", "invested": 2.5, "weights": [169 numbers]},
           {"position": "BB", "invested": 2.5, "weights": [169 numbers]},
           {"position": "BTN", "invested": 2.5, "weights": [169 numbers]}],
 "bet_sizes": [0.33, 0.75], "raise_sizes": [2.5], "max_raises": 2,
 "allin_threshold": 0.65, "turn_cards": 0, "runouts": 40,
 "iterations": 300, "check_every": 25, "target_gap_pct": 1.0}
```

`seats` are in postflop acting order; `pot` includes the folded seats' money;
`rake_pct` is a percentage like everywhere else in the API; `weights` are the
per-combo weights the preflop node view reports (`reaches_all`). The reply
carries `values` and `masses` per position (169 each), `range_ev` and
`best_response_ev` per position, `gap_pct` with its history, `iterations`,
`nodes`, `seconds` and the first seat's action split at the root.

On the 6-max baseline's BTN-SB-BB single-raised pot (two bet sizes, one raise
size, two raises, 40 runouts) a flop is 370 nodes and reaches a 0.5% gap in
75 iterations, about 40 s on one core.

## Measuring multiway terminals

`scripts/gpu-calibrate.py` in lucky-chances walks the live preflop session for
`POT_SHARE` terminals with three or more live seats at or above its reach
cutoff, sends each seat's arriving range (with the floor weight) to this
solver on the same systematic flop sample the heads-up measurement uses, and
writes the per-position values into the same `LeafValuesFile`. The preflop
engine accepts any `POT_SHARE` terminal with chips behind as long as every
live position is present; uncovered multiway terminals keep the coupled-deck
equity model.
