# SIP rewrite plan

Tracks remaining work for `cem/sip/`. The stable design — classes, module layout, plug points,
tested claims — lives in [`sip.md`](sip.md).

## Correctness foundations

- [ ] **Multi-loss gradient transform.** `Mismatch` and `TDError` stop-gradient their softplus
  loss-weight readouts because Adam under a single scalar loss would collapse them, so the
  weights behave as fixed zero. Implement the adaptive rule as a `DisGradientTransformation`.
- [ ] **JIT and vmap smoke test.** `SIPAgent.step` has not been exercised under
  `eqx.filter_jit` or vmapped over a batch of agents; expect pytree and static-field surprises.

## Thesis-constrained wiring decisions

Resolve before experiments 16 and 17 depend on them:

- [ ] **Advantage routing to intentions.** `SIPAgent` sums the full TD advantage and broadcasts
  the scalar to every intention level. The thesis says advantage reaches every level through
  advantage links; confirm whether a per-level projection is specified.
- [ ] **Sensation external predictors.** Each sensation level currently reads the matching
  intention's previous `last_message`. The thesis pins the bottom sensation's efference wiring
  explicitly; confirm whether higher levels use the same efference-style predictor.
- [ ] **Intention external predictors.** Each intention level currently reads the matching
  sensation's current `last_message`. Same confirmation against the thesis figure.

## Recruiter wiring

`SIPFeature` and `ProspectFeature` accept a `gain: InstrumentedMessage | None` but no coupling
supplies one:

- [ ] **Sensation recruiters.** Where recruiters sit in the sensation stack, what feeds their
  observation content and predictor, and which sensation levels their output gates.
- [ ] **Intention recruiters.** Same questions for the intention stack.
- [ ] **Prospect recruiters.** Whether a prospect-level recruiter stack exists at all, and how
  it couples to the sensation and intention recruiters.

## Experiments

Each of the 17 numbered claims in [`sip.md`](sip.md#tested-claims) needs:

- a `Problem` and `DataSource` under `cem/sip/experiments/` (a structural-causal-model generator
  for 1–12, the RL `Problem` from `cem/structure/rl/` for 13–17);
- a pytest in `tests/sip/experiments/` running a fast configuration and asserting the claim;
- a full-configuration CLI entry writing the JSON that the thesis `chart()` consumes.

Order of unblocked work: 1–12 (need `Emitter`, `Mismatch`, and `Credit`) then 13–15 (add
`TDError`) then 16–17 (full `SIPAgent`; needs the wiring decisions above).

- [ ] Experiment 1
- [ ] Experiment 2
- [ ] Experiment 3
- [ ] Experiment 4
- [ ] Experiment 5
- [ ] Experiment 6
- [ ] Experiment 7
- [ ] Experiment 8
- [ ] Experiment 9
- [ ] Experiment 10
- [ ] Experiment 11
- [ ] Experiment 12
- [ ] Experiment 13
- [ ] Experiment 14
- [ ] Experiment 15
- [ ] Experiment 16
- [ ] Experiment 17

## Tests and docs

- [ ] **`_predictor_for` at N = 3 layers.** Only exercised at N = 2 in both `Sensations` and
  `Intentions`; add a small test to catch an off-by-one in the layer-above lookup.
- [ ] **Agent-level feedback tests.** No test exercises the TD advantage as intention goal or
  the TD dissonance as prospect goal through the `SIPAgent` API.
- [ ] **Sharpen `sip.md`'s `SIPAgent` section.** Replace "a workable first pass" with the pinned
  choices once advantage routing and efference wiring are confirmed.

## Open design questions

Not blockers; record so they are not lost:

- [ ] **Loss scaling across reward, curiosity, and tracking.** The environment reward uses the
  environment's precision while the intrinsic channels use precision 1. Needs an adaptive rule
  so the three signals do not drown each other out.
- [ ] **Witness norm with non-trivial gain.** The unit-RHS special case `E[||w||²] ≤ 1` is
  hardcoded in `Mismatch` and `TDError`. Check whether the thesis's general `E[gain]` form
  matters once recruiters supply gain.
- [ ] **`Credit` trace-decay default.** `trace_decay_rate` defaults to 0.0; the thesis uses a
  nonzero decay. Pick a default once there's a tuning target.
- [ ] **`evidence_scale` stability at high gain.** Precision multiplies by gain, so a large gain
  drives variance toward zero and reconstruction toward a `log(variance)` singularity. Guard if
  recruiters ever drive gain above a few.
- [ ] **Prospect diffuse-readout size.** Every prospect sees `sum(sensation_sizes) +
  sum(intention_sizes)` input features. At many levels this dominates the parameter count; a
  lower-rank readout may be worth trying.
- [ ] **DAG sensation and intention topology.** Replace the stack of levels with a DAG. More
  sensation features means more curiosity and tracking channels, and the DAG dissolves the
  "level i sensation matches level i intention" assumption that currently forces the TD
  advantage to be summed and broadcast rather than routed per-feature.
- [ ] **Controllable pseudo-rewards.** Learn a projection from the raw reward features
  (extrinsic reward, curiosity, tracking) to pseudo-reward features chosen to be maximally
  controllable by specific intention features. Design notes in the thesis todo under
  "Controllable pseudo-rewards".

## Per-piece checklist

- Module in `cem/sip/` with its unit tests in `tests/sip/`.
- Any numbered experiment the piece unblocks: entry in `cem/sip/experiments/`, pytest in
  `tests/sip/experiments/`, and the JSON the thesis `chart()` consumes.
- `lefthook run pre-commit` and `pytest` pass; commit.
