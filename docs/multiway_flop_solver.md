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
  minus the combos the board blocks. Card removal is exact in a three-handed
  pot: no two seats hold the same card, between the hero and each opponent
  and between the two opponents (`showdown.rs`, `Pair`: the mass of two
  seats' hand pairs in a category is the product of the two masses, minus the
  pairs that share a card — a dot product over cards of the per-card masses
  — plus the pairs that are the same hand). Three uniform ranges checking
  down every runout own exactly 2.5 bb each of a 7.5 bb pot. With four or
  more seats the other seats are taken as independent of each other given
  the hero's cards, which lifts shares by a percent or two of the pot;
  four-way pots are under half a percent of deals at the preflop terminals.
- Betting follows the postflop order (blinds first, then seat order). Actions:
  check, bet (pot fractions), fold, call, raise (multiples of the bet faced),
  all-in whenever a size would commit more than `allin_threshold` of the
  stack, at most `max_raises` raises per street. The turn has its own sizes
  (`turn_bet_sizes`, `turn_raise_sizes`, `turn_max_raises`; the flop's when
  absent, an empty list for check or call only).
- The tree covers the flop; `turn_cards > 0` adds a turn street on that many
  sampled turn cards (0 keeps the tree one street). Every way the flop
  betting can close with chips behind grows its own turn subtree per card,
  so the turn street's sizes decide the cost: one turn size with one raise
  on six turn cards is about seventy times the one-street tree.
- Sampled cards are chosen by texture class, not at random. Every card that
  can come is classed (`card_class`) as `flush` (its suit then shows three or
  more times), `pair` (its rank is on the board), `straight` (a five-rank
  window holds it and two board ranks, so two hole cards make a straight
  through it), `over` (above every board rank) or `blank`; a sample of `k`
  cards gives every class present at least one card and the rest in
  proportion to class size, spread evenly through the class in rank order,
  and each sampled card carries the share of the deck it stands for
  (`sample_cards`). Averages over a sample are weighted by those shares, so
  they estimate the average over the deck while every texture is in the
  tree. The choice is deterministic: every solve and every calibration round
  sees the same cards for a flop.
- Leaves: a fold-out pays the raked pot to the last live seat; a street that
  closes with two or more seats live is checked down, each seat paid its share
  of the raked pot by exact showdown over the sampled completions (`runouts`
  class-weighted rivers under a turn card; about `runouts` class-weighted
  turn × river boards when the flop closes all-in). Ties split the pot
  between the tying seats. Sampled cards — the turn cards and the runouts —
  are averaged per hero combo over the cards it does not hold, and each
  other seat's hands are counted under the cards they do not hold either,
  which is fewer: the average is scaled by the ratio of the two deck sizes
  (`sample_weight`, C(47, 2) / C(43, 2) for three seats running the flop out)
  so that every opponent hand keeps its reach weight. A check-down over every
  runout then owns exactly the pot, heads-up and three-handed.
  There is no betting after the last modelled street, so hands that want to
  see every card cheaply are a little overvalued on the river, and the
  river barrels are missing. The flop and turn betting, position, fold
  equity and card removal are real.

## Algorithm

Vector-form discounted CFR (α 1.5, β 0, γ 2), alternating updates, every seat's
strategy a 1,326-vector per action node. The showdown leaves sort each
runout's combos by rank once per solve; per leaf and runout, prefix sums over
each opponent's reach in rank order (with the two per-card corrections for the
hero's cards) give every hero combo the mass of each opponent's worse, equal
and compatible hands. With two other seats those masses are also built per
card (the prefix sums at the hero's rank group, less the hands holding a
hero card), and the exclusive pair masses — worse-worse, worse-equal,
equal-equal and so on — come from the inclusion–exclusion above: about
0.15 µs per hero combo and board, ten times the independent model. With
three or more other seats the subsets of tying opponents give the share from
the per-seat masses.

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
 "turn_bet_sizes": [0.66], "turn_raise_sizes": [2.5], "turn_max_raises": 1,
 "allin_threshold": 0.65, "turn_cards": 6, "runouts": 8,
 "iterations": 150, "check_every": 25, "target_gap_pct": 0.5}
```

`seats` are in postflop acting order; `pot` includes the folded seats' money;
`rake_pct` is a percentage like everywhere else in the API; `weights` are the
per-combo weights the preflop node view reports (`reaches_all`). The reply
carries `values` and `masses` per position (169 each), `range_ev` and
`best_response_ev` per position, `gap_pct` with its history, `iterations`,
`nodes`, `seconds` and the first seat's action split at the root.

On the 6-max baseline's BTN-SB-BB single-raised pot the one-street tree (two
bet sizes, one raise size, two raises, 40 runouts) is 370 nodes and reaches a
0.5% gap in 75 iterations, about 40 s on one core; with the turn street above
(six turn cards, eight rivers, one turn size and raise) a flop is 25k nodes
and an iteration takes about 24 s on one core.

## Measuring multiway terminals

`scripts/gpu-calibrate.py` in lucky-chances walks the live preflop session for
`POT_SHARE` terminals with three or more live seats at or above its reach
cutoff, sends each seat's arriving range (with the floor weight) to this
solver on the same systematic flop sample the heads-up measurement uses (more
flops for the terminals that carry more deals), and writes the per-position
values into the same `LeafValuesFile`. The preflop engine accepts any
`POT_SHARE` terminal with chips behind as long as every live position is
present; uncovered multiway terminals keep the coupled-deck equity model.
