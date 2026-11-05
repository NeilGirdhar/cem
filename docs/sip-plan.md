# SIP rewrite plan

Tracks remaining work for `cem/sip/`. The stable design — classes, module layout, plug points,
tested claims — lives in [`sip.md`](sip.md).

## Thesis-constrained wiring decisions

Resolve before experiments 16 and 17 depend on them:

- [ ] Confirm whether each intention level uses the matching sensation's current message.

## Correctness foundations

- [ ] Require and validate a positive `trace_decay_rate` in constructors that build credit circuits.

## Adaptive loss weighting

- [ ] Adapt the reconstruction and confounding loss weights in `Mismatch` and `TDError`, and
  the uncertainty penalty weight in `TDError`.
- [ ] Adapt the relative weights of reward, curiosity, and tracking.

## Recruiter wiring

`SIPFeature` and `ProspectFeature` accept a `gain: InstrumentedMessage | None` but no coupling
supplies one:

- [ ] Specify sensation recruiters' positions, observation content, predictors, and gain targets.
- [ ] Specify intention recruiters' positions, observation content, predictors, and gain targets.
- [ ] Decide whether to add prospect recruiters and how to couple them to the other recruiters.

## Experiments

Each of the 17 numbered claims in [`sip.md`](sip.md#tested-claims) needs:

- A problem and data source in `cem/sip/experiments/`. Use a structural causal model for 1–12
  and an RL problem for 13–17.
- A fast pytest in `tests/sip/experiments/` that checks the claim.
- A CLI entry that runs the full experiment and writes JSON for the thesis chart.

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

## Tests

- [ ] Smoke-test `SIPAgent.step` with `eqx.filter_jit` and with `vmap` over a batch of agents.
- [ ] Test `_predictor_for` with three layers in both `Sensations` and `Intentions`.
- [ ] Test that `SIPAgent` broadcasts summed TD advantage to all intention levels and sends TD
  dissonance to prospects.

## Open design questions

Not blockers; record so they are not lost:

- [ ] Verify the gain-weighted witness norm bound once recruiters supply gain.
- [ ] Check whether high recruiter gain requires a numerical guard in `evidence_scale`.
- [ ] Evaluate a low-rank diffuse readout for prospects.
- [ ] Design a DAG for sensations and intentions with per-feature TD advantage routing.
- [ ] Learn pseudo-rewards maximally controllable by specific intentions from environment reward,
  curiosity, and tracking. See “Controllable pseudo-rewards” in the thesis TODO.
