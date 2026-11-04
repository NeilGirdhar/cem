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
- [ ] `Mismatch` — not started. With `Emitter + Credit`, unlocks tests 1–12.
- [ ] `TDError` — not started. Unlocks tests 13–15.
- [ ] `SIPFeature`, `TDFeature` — wire the leaves into per-node features.
- [ ] `ExplanatoryCoupling`, `SIPValueModel`, `SIPAgent` — the `Model`s that
  compose features into the test surfaces for tests 16–17.

## Per-piece checklist

- Module in `cem/sip/` with its unit tests in `tests/sip/`.
- Any numbered test the piece unblocks: experiment in `cem/sip/experiments/`,
  pytest in `tests/sip/experiments/`, and JSON output readable by thesis
  `chart()`.
- `lefthook run pre-commit` and `pytest` pass; commit.
