# SIP design

The classes, module layout, and tested claims for chapter 5 of the thesis, as implemented under
`cem/sip/`. For the in-flight rewrite progress and process, see [`sip-plan.md`](sip-plan.md); for
the framework SIP plugs into, see [`architecture.md`](architecture.md).

## Classes

Naming convention: bare names for domain-specific nouns (`Emitter`, `Mismatch`, …); the `SIP`
prefix only for generic ones that would otherwise collide with common ML terms (`SIPFeature`,
`SIPAgent`). `ExplanatoryCoupling`, `Sensations`, `Intentions` and `RewardCritic` are thesis terms
and take no prefix.

Circuit sub-modules:

- `Credit` — eligibility-based parameter-update rule.
- `Mismatch` — prediction and two witnesses over predictor and gain instruments producing
  reconstruction and confounding costs.
- `IntegralWitnessMap` and `ProportionalWitnessMap` — the two witnesses shared by `Mismatch` and
  `TDError`. Both encode each instrument with bounded presence and centre their output at the
  all-certain-zero context. The integral witness returns tanh-bounded coefficients and ascends
  `integral_witness_loss` under the participation-weighted norm bound, enforced by each circuit's
  learned witness multiplier. The proportional witness is unbounded and ascends
  `proportional_witness_loss`, a gain-weighted regression on the residuals. `witness_losses`
  combines them: the confounding cost adds the proportional loss, scaled by the damping weight, to
  the integral loss, which damps purification without moving its resting point.
- `TDError` — Gaussian bootstrap value error with uncertainty penalty.
- `Emitter` — forward dynamics: observation map, instrument map, noise top-up, injection.
- `inherit_instrument`, `instrument_loss` — the first-stage instrument map and the
  gain-gated cross-entropy that fits it to the raw observation.

Circuits train only their own maps. Prediction maps, witness maps, and instrument maps take their
parents' messages with gradients stopped, and mismatch costs hold the target and the gain fixed.
Parents learn from their own goals through credit.

Features (per causal node):

- `SIPFeature` — identification node (emission plus mismatch).
- `SensationInputFeature` — sensation-stack entry point; the environment observation lands here.
- `ProspectFeature` — candidate causal factor for the reward critic.
- `TDFeature` — the reward critic's TD core.
- `RecruiterFeature` — feature that supplies gain to its consumers.

Messages (`messages.py`):

- `InstrumentedMessage` — paired observation and instrument Gaussian message, the unit exchanged
  between features.
- `concat_messages`, `concat_instrumented_messages` — component-axis concatenation.
- `variance_from_precision`, `evidence_scale` — Gaussian-message utilities. Moment conversions
  use efax's normal parametrizations.

Composition modules (plain `eqx.Module`s, not `Model` subclasses):

- `ExplanatoryCoupling` — marker base for `Sensations` and `Intentions`.
- `Sensations` — bottom-up stack of identification features that parse the environment.
- `Intentions` — top-down stack of controllable identification features; the bottom level emits the
  action.
- `RewardCritic` — prospects in parallel feeding one `TDFeature` (tests 14–16).
- `SIPAgent` — a `Sensations`, an `Intentions`, and a `RewardCritic` sharing state; the
  environment-facing module (test 18).

Implementation gates: `Emitter` alone covers test 13; `Emitter`, `Mismatch`, and `Credit` cover
tests 1–13. `TDError` adds tests 14–16. Tests 17–18 wire the full stack together and
need no new modules.

### Efference copy and intention nexting

The bottom intention's observation is the action. On the next step, `SIPAgent` appends that
stored action to the new environment observation and passes the result to the bottom sensation.
The first step uses a standard normal in place of the previous action.

For intention nexting, each sensation level uses the previous instrumented message from the
matching intention. The message comes from the previous step because `SIPAgent` runs sensation
before intention.

### Affordance readout and intention prediction

Each intention level receives the matching sensation's current observation as its readout.
Its predictor is the previous instrumented message from the intention directly below.
The bottom intention has no predictor inputs.

## Agent construction

Pass `SIPAgentSettings` and random streams to `SIPAgent.create`. The settings group environment
widths in `EnvironmentFeatures` and circuit settings in `CreditSettings`, `MismatchSettings`,
and `TDSettings`. The agent derives connection widths from the environment and feature widths.

`CreditSettings` requires a finite, positive trace decay rate shared by the agent's credit
circuits. Mismatch and TD settings select the prediction, integral witness, and proportional
witness hidden widths and the damping weight. The TD settings also supply the discount. By
default, the maps have no hidden layers, and the damping weight and the discount are one.

## Target code layout

- `cem/sip/` — the circuit.
    - `messages.py` — `InstrumentedMessage` plus SIP-specific Gaussian helpers (concatenation,
      evidence scaling, variance-from-precision).
    - `credit.py` — `Credit`.
    - `mismatch.py` — `Mismatch`.
    - `witness.py` — `IntegralWitnessMap`, `ProportionalWitnessMap`, `witness_residuals`,
      `witness_losses`.
    - `instrument.py` — `inherit_instrument`, `instrument_loss`.
    - `td_error.py` — `TDError`.
    - `emitter.py` — `Emitter`.
    - `features.py` — `SIPFeature`, `SensationInputFeature`, `ProspectFeature`, `TDFeature`,
      `RecruiterFeature`.
    - `explanatory_coupling.py` — `ExplanatoryCoupling` marker base.
    - `sensations.py` — `Sensations`.
    - `intentions.py` — `Intentions`.
    - `reward_critic.py` — `RewardCritic`.
    - `sip_agent.py` — `SIPAgent`.
    - `settings.py` — agent and circuit construction settings.
- `cem/sip/experiments/` — experiments and test fixtures.
    - `datasets.py` — `Problem` / `DataSource` for the identification tests.
    - `scenarios.py` — parametrized SCM generator (confounding, instrument strength,
      heteroskedasticity, endogenous gain, chain depth, bypass paths, nonlinearity).
    - `baselines.py` — 2SLS and OLS reference implementations.
    - `evaluation.py` — paired-sample runner, multi-seed aggregation, cross-fitting splits.
    - `instruments.py` — tests 1–3.
    - `prediction.py` — tests 4–8.
    - `emitter_credit.py` — tests 9–13.
    - `value_learning.py` — tests 14–16.
    - `system.py` — tests 17–18.
- `tests/sip/` — unit tests for SIP modules.

## SIP-specific plug points

The framework (`Problem` / `Model` / `Solver` / `Plotter` / `Experiment`) is documented in
[`architecture.md`](architecture.md). What's specific to SIP:

- Experiments 1–13 use a feed-forward `Problem` over a structural causal model.
  Value-learning and control tests (14–18) use the RL `Problem` in `cem/framework/rl/`.
- The circuit needs the precision-κ transform on `DisGradientTransformation`.
- Reuses `cem/npn/gaussian.py` (`GaussianNeuralProjection`),
  `cem/perceptron/mlp.py` (`MLP` inside the witness map, and for the witness-gain and
  inherited-variance maps), and `cem/transforms/` as needed. Gaussian natural-parameter conversions
  come from `efax` (`NormalNP`, `NormalPP`, `NormalVP`).

## Tested claims

Each claim is an experiment function in `cem/sip/experiments/` plus a pytest in
`tests/sip/experiments/` that runs a fast configuration and asserts the claim. The CLI runs the
full configuration and writes chart JSON.

### Instruments

1. **Inherited instrument.** On held-out samples, compare the learned first-stage strength with the
   oracle `E[y | parent instruments]`, and run 2SLS with it to recover the downstream effect.
   Report instrument approximation error and downstream estimation error separately. 2SLS on the
   injected noise is the reference. Cross-fit the first stage into evaluation folds (Chernozhukov
   et al.).
2. **Instruments survive depth.** Repeat the held-out first-stage and downstream measurements at
   each hop along a chain of non-controllable features.
3. **Exclusion repair.** Give the bypass path a second predictor. Choose experiments that
   distinguish the two causal paths and recover both coefficients. As a diagnostic condition, omit
   the bypass predictor and show that the single-instrument estimate drifts toward the mixed path.

### Prediction under confounding (synthetic instruments)

4. **Causal mean.** Start with the linear, unit-gain case. Reconstruction alone should reach OLS;
   with purification it should reach the causal effect. Add a two-parents, two-instruments
   condition.
5. **Causal variance.** Use a data-generating model whose interventional variance satisfies the
   circuit's second-moment restriction. The predicted variance should match the interventional
   variance.
6. **Damping and variance purification.** Use the one-parent setup from experiments 4 and 5 to
   test what each witness mechanism contributes. Match initialization, batches, and training
   settings across the runs of each comparison.

   - **Damping.** Train three purification arms that differ only in the prediction's update:
     - integral only (`damping_weight = 0`);
     - proportional only (no integral term);
     - both (`damping_weight = 1`).

     Plot each seed's signed effect estimate over training, without averaging across seeds, and
     mark the true effect of 1.5. Over the last quarter of training, report each seed's mean
     signed error and the standard deviation of its estimate across updates. Integral only should
     reach 1.5 but may keep oscillating. Proportional only acts as a finite quadratic penalty, so it
     should settle between OLS (0.61) and 1.5. Both should settle at 1.5.
   - **Variance purification.** In test 5, once the mean is causal, the best constant variance is
     already the interventional 4.26, so reconstruction recovers it without the variance witness.
     Fix the predicted mean at `1.5Y` in both runs, so only the variance update differs. Compare
     reconstruction alone with reconstruction plus variance purification.

     Use the same variance model in both runs, `softplus(b + cY²)` plus the variance floor, so it
     can learn a constant or vary with `Y²`. Reconstruction should learn the observational curve
     `2.48 + 0.79Y²`, because `Y` carries information about the confounder. Purification should
     learn the flat interventional observation variance of 4.26: 4.25 for `Z` plus 0.01 for its
     observation. Plot both curves over a uniform grid of interventions `do(Y = y)` against both
     references. Report each run's RMSE from 4.26 over the grid and its fitted `c`. Score over a
     uniform grid, because the observational curve also averages to 4.26 over the distribution
     of `Y`.

7. **Nonlinear effects.** Match a nonlinear structural function that the instruments identify.
8. **Endogenous gain.** Gain depends on the confounder, isolating the weighted-moment conditions.
   Run with and without the witness-gain map.

### Emitter and credit (oracle goals)

9. **Mean credit — direct.** Goal whose confounded correlation has the opposite sign to its causal
   effect. Compare the actual parameter update with the oracle causal gradient; report both their
   inner product and the angle. Separate goal contribution from shrinkage before checking
   direction. Include a unit-gain baseline and an endogenous-gain condition.
10. **Mean credit — inherited.** Same comparison at a non-controllable feature whose variation comes
    only from inherited instruments.
11. **Precision.** In a controlled Gaussian case, check presence against known pointwise curvature
    and against smoothed curvature at finite σ. More generally, compare presence with the circuit's
    constrained-regression optimum. Include contextual baselines, and a condition where the
    eligibility traces contain insufficient variation to distinguish coefficients.
12. **Temporal credit through shared parameters.** One parameter is reused across time steps; the
    credit must sum its contributions correctly. Include the shared-parameter counterexample from
    the thesis todo as a condition the test must either pass or reveal.
13. **Noise generation (emitter test).** The rule is instantaneous:
    `V_local = max(0, σ̃²_inh − μ̃²_inh)`. Check (i) the conditional local variance against this
    rule, (ii) that the same perturbation enters observation and instrument, and (iii) the squared
    instrument displacement before gain: `E[μ̃²_ins | inherited message] = max(σ̃²_inh, μ̃²_inh)`.
    This is an instantaneous top-up, not a guarantee that total experimental variance equals
    `1/κ`.

### TD-error circuit

14. **Return moments.** On a Markov chain with known return mean and variance, establish
    calibration without the uncertainty penalty first; then turn it on and measure its effect.
15. **Causal value.** A hidden variable confounds prospect and reward; the witness is active.
16. **Penalty escapes the variance trap.** From a broad, wrong start, compare learning with and
    without the penalty, checking the mean for bias.

### Whole system

17. **Joint learning.** In a multi-layer graph, emitters, instruments and predictions all learn at
    once. For learned observations, measure recovery through intervention responses, or align the
    learned coordinates with the generative ones before comparing coefficients.
18. **Causal control.** In an environment whose confounded correlation has the wrong sign, the
    agent learns intentions that increase reward.

## Methodology

- **Notation.** κ is precision.
- **Paired samples** for every ablation.
- **Multiple seeds**, with per-seed and aggregated reports.
- **Held-out intervention measurements** for every recovery claim.
- **Cross-fitting** of learned first stages when they feed a downstream estimate.
