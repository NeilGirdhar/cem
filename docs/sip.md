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
- `Mismatch` — witness over predictor and gain instruments producing reconstruction and confounding
  costs.
- `WitnessMap` — the confounding witness shared by `Mismatch` and `TDError`. It encodes each
  instrument with bounded presence and returns tanh-bounded coefficients centred at the
  all-certain-zero context. `witness_loss` trains it under the participation-weighted norm bound,
  enforced by each circuit's learned witness multiplier.
- `TDError` — Gaussian bootstrap value error with uncertainty penalty.
- `Emitter` — forward dynamics: observation map, instrument map, noise top-up, injection.
- `inherit_instrument`, `instrument_mismatch_loss` — the first-stage instrument map and the
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
- `mean_and_precision`, `variance_from_precision`, `evidence_scale`, `promoting_link_loss`,
  `suppressing_link_loss` — Gaussian-message utilities.

Composition modules (plain `eqx.Module`s, not `Model` subclasses):

- `ExplanatoryCoupling` — marker base for `Sensations` and `Intentions`.
- `Sensations` — bottom-up stack of identification features that parse the environment.
- `Intentions` — top-down stack of controllable identification features; the bottom level emits the
  action.
- `RewardCritic` — prospects in parallel feeding one `TDFeature` (tests 13–15).
- `SIPAgent` — a `Sensations`, an `Intentions`, and a `RewardCritic` sharing state; the
  environment-facing module (test 17).

Implementation gates: `Emitter` alone covers test 12; `Emitter`, `Mismatch`, and `Credit` cover
tests 1–12; `TDError` adds 13–15; tests 16–17 wire the full stack together and need no new modules.

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
circuits. Mismatch and TD settings select the prediction and witness hidden widths. The TD
settings also supply the discount. By default, the maps have no hidden layers and the discount
is one.

## Target code layout

- `cem/sip/` — the circuit.
    - `messages.py` — `InstrumentedMessage` plus SIP-specific Gaussian helpers (concatenation,
      evidence scaling, variance-from-precision, promoting/suppressing link losses).
    - `credit.py` — `Credit`.
    - `mismatch.py` — `Mismatch`.
    - `witness.py` — `WitnessMap`, `witness_residuals`, `witness_loss`.
    - `instrument.py` — `inherit_instrument`, `instrument_mismatch_loss`.
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
    - `prediction.py` — tests 4–7.
    - `emitter_credit.py` — tests 8–12.
    - `value_learning.py` — tests 13–15.
    - `system.py` — tests 16–17.
- `tests/sip/` — unit tests for SIP modules.

## SIP-specific plug points

The framework (`Problem` / `Model` / `Solver` / `Plotter` / `Experiment`) is documented in
[`architecture.md`](architecture.md). What's specific to SIP:

- Identification tests (1–12) use a feed-forward `Problem` over a structural causal model;
  value-learning and control tests (13–17) use the RL `Problem` in `cem/structure/rl/`.
- The circuit needs the precision-κ transform on `DisGradientTransformation`.
- Reuses `cem/npn/gaussian.py` (`GaussianNPN`, `normal_from_mean_and_presence`),
  `cem/perceptron/mlp.py` (`MLP` inside the witness map, and for the witness-gain and
  inherited-variance maps), and `cem/transforms/` as needed. Gaussian natural-parameter conversions
  come from `efax` (`NormalNP` / `NormalVP`).

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
6. **Nonlinear effects.** Match a nonlinear structural function that the instruments identify.
7. **Endogenous gain.** Gain depends on the confounder, isolating the weighted-moment conditions.
   Run with and without the witness-gain map.

### Emitter and credit (oracle goals)

8. **Mean credit — direct.** Goal whose confounded correlation has the opposite sign to its causal
   effect. Compare the actual parameter update with the oracle causal gradient; report both their
   inner product and the angle. Separate goal contribution from shrinkage before checking
   direction. Include a unit-gain baseline and an endogenous-gain condition.
9. **Mean credit — inherited.** Same comparison at a non-controllable feature whose variation comes
   only from inherited instruments.
10. **Precision.** In a controlled Gaussian case, check presence against known pointwise curvature
    and against smoothed curvature at finite σ. More generally, compare presence with the circuit's
    constrained-regression optimum. Include contextual baselines, and a condition where the
    eligibility traces contain insufficient variation to distinguish coefficients.
11. **Temporal credit through shared parameters.** One parameter is reused across time steps; the
    credit must sum its contributions correctly. Include the shared-parameter counterexample from
    the thesis todo as a condition the test must either pass or reveal.
12. **Noise generation (emitter test).** The rule is instantaneous:
    `V_local = max(0, σ̃²_inh − μ̃²_inh)`. Check (i) the conditional local variance against this
    rule, (ii) that the same perturbation enters observation and instrument, and (iii) the squared
    instrument displacement before gain: `E[μ̃²_ins | inherited message] = max(σ̃²_inh, μ̃²_inh)`.
    This is an instantaneous top-up, not a guarantee that total experimental variance equals
    `1/κ`.

### TD-error circuit

13. **Return moments.** On a Markov chain with known return mean and variance, establish
    calibration without the uncertainty penalty first; then turn it on and measure its effect, and
    compare the adaptation rules for the loss weights.
14. **Causal value.** A hidden variable confounds prospect and reward; the witness is active.
15. **Penalty escapes the variance trap.** From a broad, wrong start, compare learning with and
    without the penalty, checking the mean for bias.

### Whole system

16. **Joint learning.** In a multi-layer graph, emitters, instruments and predictions all learn at
    once. For learned observations, measure recovery through intervention responses, or align the
    learned coordinates with the generative ones before comparing coefficients.
17. **Causal control.** In an environment whose confounded correlation has the wrong sign, the
    agent learns intentions that increase reward.

## Methodology

- **Notation.** κ is precision; π is loss weights.
- **Paired samples** for every ablation.
- **Multiple seeds**, with per-seed and aggregated reports.
- **Held-out intervention measurements** for every recovery claim.
- **Cross-fitting** of learned first stages when they feed a downstream estimate.
