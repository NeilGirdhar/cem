# SIP rewrite plan

Tracks remaining work for `cem/sip/`. The stable design — classes, module layout, plug points,
tested claims — lives in [`sip.md`](sip.md).

## Adaptive loss weighting

- [x] Balance reconstruction and confounding in mismatch prediction maps.
- [x] Balance reconstruction, confounding, and uncertainty in TD prediction maps.
- [ ] Balance reconstruction and confounding in sensation, prospect, and recruiter observation maps.
- [ ] Balance reward, curiosity, and tracking in intention observation maps.

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
`TDError`) then 16–17 (full `SIPAgent`).

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
- [ ] Test that `SIPAgent` broadcasts each summed TD advantage source to all intention levels
  and sends both TD costs to prospects.

## Open design questions

Not blockers; record so they are not lost:

- [ ] Verify the gain-weighted witness norm bound once recruiters supply gain.
- [ ] Check whether high recruiter gain requires a numerical guard in `evidence_scale`.
- [ ] Evaluate a low-rank diffuse readout for prospects.
- [ ] Design a DAG for sensations and intentions with per-feature TD advantage routing.
- [ ] Learn pseudo-rewards maximally controllable by specific intentions from environment reward,
  curiosity, and tracking. See “Controllable pseudo-rewards” in the thesis TODO.
