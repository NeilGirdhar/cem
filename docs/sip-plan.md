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
- [ ] Recruiter support — accept `gain_observation: NormalNP | None` and
  `gain_instrument: NormalNP | None` in `Emitter`, `Mismatch`, `Credit`, and
  the feature wrappers; one `is None` branch per site drops gain multiplies
  and sets the norm-constraint RHS to `1.0`. Witness-map input size is set
  at create time, so recruiter `Mismatch`es still take
  `gain_instrument_features=0`. Add a `RecruiterFeature` composition
  (controllable `SIPFeature` with no gain parent) that supplies gain to
  its consumer features.
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
