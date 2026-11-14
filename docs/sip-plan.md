# SIP rewrite plan

Tracks remaining work for `cem/sip/`. The stable design — classes, module layout, plug points,
tested claims — lives in [`sip.md`](sip.md).

## Recruiter wiring

`SIPFeature` and `ProspectFeature` accept a `gain: InstrumentedMessage | None` but no coupling
supplies one:

- [ ] Specify sensation recruiters' positions, observation content, predictors, and gain targets.
- [ ] Specify intention recruiters' positions, observation content, predictors, and gain targets.
- [ ] Decide whether to add prospect recruiters and how to couple them to the other recruiters.

## Experiments

Each of the 18 numbered claims in [`sip.md`](sip.md#tested-claims) needs:

- A problem and data source in `cem/sip/experiments/`. Use a structural causal model for 1–13
  and an RL problem for 14–18.
- A fast pytest in `tests/sip/experiments/` that checks the claim.
- A CLI entry that runs the full experiment and writes JSON for the thesis figures.

Experiments 1–13 need `Emitter`, `Mismatch`, and `Credit`. Experiments 14–16 add `TDError`,
and 17–18 need the full `SIPAgent`.

- [x] Experiment 1
- [x] Experiment 2
- [x] Experiment 3
- [x] Experiment 4
- [x] Experiment 5
- [ ] Experiment 6
- [x] Experiment 7
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
- [ ] Experiment 18

### Experiment 6. Damping and variance purification

The claim, its arms, and the expected outcomes are in [`sip.md`](sip.md#tested-claims).

- **Check first.** Before fixing the training budget, run two seeds of each comparison:
    - Confirm that integral-only SGD at learning rate 0.003 visibly oscillates. If it settles, the
      result is that damping is unnecessary in this regime; report that rather than assume
      oscillation.
    - Confirm that the variance witness moves the fixed-mean curve within the budget. The variance
      residual contains the squared mean residual, whose standard deviation is about 6.
- **Ablation switches.** Allow `damping_weight = 0` in `MismatchSettings`, keeping the default
  of 1 and rejecting negative and non-finite values. Drop the integral term from the prediction's
  update with a flag on the experiment's model, so the circuit and the thesis keep a single
  integral term. In every arm, each witness keeps training on its own loss.
- **Matched runs.** Every arm builds the same modules, including witnesses whose output it
  ignores, so arms with the same seed start from identical parameters and see the same batches.
- **Effect telemetry.** Measure the effect as the difference between the predictions at the
  probes `Y = 1` and `Y = 0`, each observed at the training precision of 100. For the affine
  predictor, this is the slope. Record about 200 snapshots per run if snapshots at every episode
  slow training.
- **Variance-only witnesses.** Zero the mean column of both the residuals and the witness outputs
  before computing the witness losses, so the mean cannot consume the integral witness's norm
  budget. Give each witness a hidden layer at the existing widths so it can detect a dependence on
  `A²`.
- **Figures.**
    - Effect trajectories: one colour and one legend entry per arm, and the true effect dashed. If
      15 lines in one panel are unreadable, use one panel per arm.
    - Variance curves: both runs, with the flat 4.26 reference and the observational curve dashed.
    - Quote the late-training and grid metrics in the text from the JSON.

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
