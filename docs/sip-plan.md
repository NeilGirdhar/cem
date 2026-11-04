# SIP rewrite plan

Tracks the in-flight rewrite of `cem/sip/` against chapter 5 of the thesis.
The stable design — classes, module layout, plug points, tested claims — lives
in [`sip.md`](sip.md).

## Working style

Interactive, piece by piece. Each piece ships its module, its unit tests, any
numbered experiment it unblocks, and the JSON output that the thesis consumes
via `chart()`. Order is set turn by turn, not pre-sequenced.

## Progress

Leaves first, then features, then models
([implementation gates in `sip.md`](sip.md#classes)):

- [x] `Credit` — committed, unit-tested.
- [x] `Emitter` — committed, unit-tested. Unlocks test 12.
- [x] `Mismatch` — committed, unit-tested. Experiments 4–7 still to write.
- [x] `TDError` — committed, unit-tested. Experiments 13–15 still to write.
- [x] `SIPFeature`, `TDFeature` — committed, unit-tested.
- [x] Recruiter support — `gain_observation` and `gain_instrument` accepted
  as `NormalNP | None` in `Emitter`, `Mismatch`, `Credit`, and `SIPFeature`;
  `RecruiterFeature` added as a controllable `SIPFeature` with `gain=0`.
- [ ] `ExplanatoryCoupling`, `RewardCritic`, `SIPAgent` — plain `eqx.Module`s
  (not `Model` subclasses) that wire features together per the thesis's
  architecture figures (prediction, demand, readout, dissonance, advantage).
  `RewardCritic` replaces the earlier `SIPValueModel` name: it is prospect
  `SIPFeature`s plus one `TDFeature`.

## Per-piece checklist

- Module in `cem/sip/` with its unit tests in `tests/sip/`.
- Any numbered test the piece unblocks: experiment in `cem/sip/experiments/`,
  pytest in `tests/sip/experiments/`, and JSON output readable by thesis
  `chart()`.
- `lefthook run pre-commit` and `pytest` pass; commit.
