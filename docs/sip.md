# SIP design

The classes, module layout, and tested claims for chapter 5 of the thesis
(`~/src/typst/thesis.typ`), as implemented under `cem/sip/`. For the in-flight
rewrite progress and process, see [`sip-plan.md`](sip-plan.md); for the
framework SIP plugs into, see [`architecture.md`](architecture.md).

## Classes

Naming convention: bare names for domain-specific nouns (`Emitter`, `Mismatch`,
…); the `SIP` prefix only for generic ones that would otherwise collide with
common ML terms (`SIPFeature`, `SIPAgent`). `ExplanatoryCoupling`, `Sensations`,
`Intentions` and `RewardCritic` are thesis terms and take no prefix.

Circuit sub-modules (leaves first):

- `Credit` — eligibility-based update rule;
  `update(traces, stimulus, response, goal) → (new_traces, cotangent)`.
    - Traces live in the recurrent `state` carry, not in the module.
- `Mismatch` — reconstruction and observation errors, witness and witness-gain
  maps, loss weights, dissonance.
- `TDError` — Gaussian bootstrap target, one-step delay, advantage, uncertainty
  penalty.
    - Delay buffer lives in the recurrent `state` carry.
- `Emitter` — forward dynamics (observation map, instrument map, noise top-up,
  injection).
    - Contains `Credit`.

Features (per causal node):

- `SIPFeature` — one identification node: `Emitter` + `Mismatch`. Plays the
  "prospect feature" role when wired into `RewardCritic`. Each `step` writes
  its outgoing `InstrumentedMessage` into `state` through
  `last_message_index`, so the next step and any sibling in the same
  coupling can read the previous (or just-written) message.
- `SensationInputFeature` — the sensation-stack input. Drops `observation_map`
  (the environment observation is the raw observation) and `Credit` (a sensor
  only reports), keeping `instrument_map` and `Mismatch`; also writes its
  outgoing `InstrumentedMessage` into `state`.
- `TDFeature` — one value node: `TDError` only; reward features use fixed
  unit gain and no local noise, so they skip the emitter. Its observation is
  the pass-through reward, so it has no `last_message_index`.
- `RecruiterFeature` — one recruiter node: a controllable `SIPFeature` with
  no gain parent. Steps with `gain_observation=None, gain_instrument=None`
  (and its underlying `Mismatch` is created with `gain_features=0`
  so the witness map ignores the missing gain parent). Supplies gain to its
  consumer features.

Messages and helpers in `messages.py`:

- `InstrumentedMessage` — a Gaussian message paired with its instrument
  half; the pair exchanged between features and between a feature and its
  recruiter.
- `concat_messages`, `concat_instrumented_messages` — concatenation along
  the component axis (treating each input as an independent-component
  diagonal Gaussian); used to join readouts with demand signals and to join
  external predictors with the stored message of the layer directly above.
- `mean_and_precision`, `variance_from_precision`, `evidence_scale`,
  `promoting_link_loss`, `suppressing_link_loss` — Gaussian-message
  utilities used throughout the circuit.

Composition `Module`s (plain `eqx.Module`s, not `Model` subclasses):

- `ExplanatoryCoupling` — marker base for `Sensations` and `Intentions`.
  Each subclass declares its own feature-stack schema.
- `Sensations` — bottom-up coupling. Layer 0 is a `SensationInputFeature`;
  layers 1+ are non-controllable `SIPFeature`s. The environment observation
  is layer 0's content; each higher layer's content is the previous layer's
  `observation_mismatch`, and its Credit goal is the previous layer's
  `dissonance`. Each layer's effective predictor is the external predictor
  concatenated with the stored message of the layer directly above
  (previous-step value, since higher layers have not yet been processed
  this step).
- `Intentions` — top-down coupling of controllable `SIPFeature`s. The top
  layer takes its content from the readout alone; lower layers concatenate
  their readout with the above layer's `observation_mismatch`. Each
  layer's Credit goal is the externally supplied advantage. The effective
  predictor concat follows the same rule as `Sensations`; in top-down
  order the above-layer message is the just-written current-step value.
  Layer 0 is the bottom (processed last) and its outgoing `observation` is
  the action.
- `RewardCritic` — prospect `SIPFeature`s plus one `TDFeature`, wired for
  reward prediction and purification; tests 13–15.
- `SIPAgent` — a `Sensations`, an `Intentions` (the policy), and a
  `RewardCritic`, wired for readout, affordance, advantage, and dissonance
  links; test 17.

Implementation gates: `Emitter` alone covers test 12; `Emitter + Mismatch +
Credit` covers tests 1–12; `TDError` adds 13–15; tests 16–17 wire the full
stack together and need no new modules.

## Target code layout

- `cem/sip/` — the circuit.
    - `messages.py` — `InstrumentedMessage` plus SIP-specific Gaussian
      helpers (concatenation, evidence scaling, variance-from-precision,
      promoting/suppressing link losses).
    - `credit.py` — `Credit`.
    - `mismatch.py` — `Mismatch`.
    - `td_error.py` — `TDError`.
    - `emitter.py` — `Emitter`.
    - `features.py` — `SIPFeature`, `SensationInputFeature`, `TDFeature`,
      `RecruiterFeature`.
    - `explanatory_coupling.py` — `ExplanatoryCoupling` marker base.
    - `sensations.py` — `Sensations`.
    - `intentions.py` — `Intentions`.
    - `reward_critic.py` — `RewardCritic` (to come).
    - `sip_agent.py` — `SIPAgent` (to come).
- `cem/sip/experiments/` — experiments and test fixtures.
    - `datasets.py` — `Problem` / `DataSource` for the identification tests.
    - `scenarios.py` — parametrized SCM generator (confounding, instrument
      strength, heteroskedasticity, endogenous gain, chain depth, bypass paths,
      nonlinearity).
    - `baselines.py` — 2SLS and OLS reference implementations.
    - `evaluation.py` — paired-sample runner, multi-seed aggregation,
      cross-fitting splits.
    - `instruments.py` — tests 1–3.
    - `prediction.py` — tests 4–7.
    - `emitter_credit.py` — tests 8–12.
    - `value_learning.py` — tests 13–15.
    - `system.py` — tests 16–17.

## SIP-specific plug points

The framework (`Problem` / `Model` / `Solver` / `Plotter` / `Experiment`) is
documented in [`architecture.md`](architecture.md). What's specific to SIP:

- Identification tests (1–12) use a feed-forward `Problem` over a structural
  causal model; value-learning and control tests (13–17) use the RL `Problem`
  in `cem/structure/rl/`.
- The circuit needs the precision-κ transform on `DisGradientTransformation`.
- Reuses `cem/npn/gaussian.py` (`GaussianNPN`,
  `normal_from_mean_and_presence`), `cem/perceptron/mlp.py` (`MLP` for witness,
  witness-gain, and inherited-variance maps), and `cem/transforms/` as needed.
  Gaussian natural-parameter conversions come from `efax` (`NormalNP` /
  `NormalVP`).

## Tested claims

Each claim is an experiment function in `cem/sip/experiments/` plus a pytest in
`tests/sip/experiments/` that runs a fast configuration and asserts the claim.
The CLI runs the full configuration and writes chart JSON.

### Instruments

1. **Inherited instrument.** On held-out samples, compare the learned
   first-stage strength with the oracle `E[y | parent instruments]`, and run
   2SLS with it to recover the downstream effect. Report instrument
   approximation error and downstream estimation error separately. 2SLS on the
   injected noise is the reference. Cross-fit the first stage into evaluation
   folds (Chernozhukov et al.).
2. **Instruments survive depth.** Repeat the held-out first-stage and
   downstream measurements at each hop along a chain of non-controllable
   features.
3. **Exclusion repair.** Give the bypass path a second predictor. Choose
   experiments that distinguish the two causal paths and recover both
   coefficients. As a diagnostic condition, omit the bypass predictor and show
   that the single-instrument estimate drifts toward the mixed path.

### Prediction under confounding (synthetic instruments)

4. **Causal mean.** Start with the linear, unit-gain case. Reconstruction alone
   should reach OLS; with purification it should reach the causal effect. Add a
   two-parents, two-instruments condition.
5. **Causal variance.** Use a data-generating model whose interventional
   variance satisfies the circuit's second-moment restriction. The predicted
   variance should match the interventional variance.
6. **Nonlinear effects.** Match a nonlinear structural function that the
   instruments identify.
7. **Endogenous gain.** Gain depends on the confounder, isolating the
   weighted-moment conditions. Run with and without the witness-gain map.

### Emitter and credit (oracle goals)

8. **Mean credit — direct.** Goal whose confounded correlation has the opposite
   sign to its causal effect. Compare the actual parameter update with the
   oracle causal gradient; report both their inner product and the angle.
   Separate goal contribution from shrinkage before checking direction. Include
   a unit-gain baseline and an endogenous-gain condition.
9. **Mean credit — inherited.** Same comparison at a non-controllable feature
   whose variation comes only from inherited instruments.
10. **Precision.** In a controlled Gaussian case, check presence against known
    pointwise curvature and against smoothed curvature at finite σ. More
    generally, compare presence with the circuit's constrained-regression
    optimum. Include contextual baselines, and a condition where the
    eligibility traces contain insufficient variation to distinguish
    coefficients.
11. **Temporal credit through shared parameters.** One parameter is reused
    across time steps; the credit must sum its contributions correctly. Include
    the counterexample recorded in `~/src/typst/todo.txt` as a condition the
    test must either pass or reveal.
12. **Noise generation (emitter test).** The rule is instantaneous:
    `V_local = max(0, σ̃²_inh − μ̃²_inh)`. Check (i) the conditional local
    variance against this rule, (ii) that the same perturbation enters
    observation and instrument, and (iii) the squared instrument displacement
    before gain:
    `E[μ̃²_ins | inherited message] = max(σ̃²_inh, μ̃²_inh)`.
    This is an instantaneous top-up, not a guarantee that total experimental
    variance equals `1/κ`.

### TD-error circuit

13. **Return moments.** On a Markov chain with known return mean and variance,
    establish calibration without the uncertainty penalty first; then turn it
    on and measure its effect, and compare the adaptation rules for the loss
    weights.
14. **Causal value.** A hidden variable confounds prospect and reward; the
    witness is active.
15. **Penalty escapes the variance trap.** From a broad, wrong start, compare
    learning with and without the penalty, checking the mean for bias.

### Whole system

16. **Joint learning.** In a multi-layer graph, emitters, instruments and
    predictions all learn at once. For learned observations, measure recovery
    through intervention responses, or align the learned coordinates with the
    generative ones before comparing coefficients.
17. **Causal control.** In an environment whose confounded correlation has the
    wrong sign, the agent learns intentions that increase reward.

## Methodology

- **Notation.** κ is precision; π is loss weights.
- **Paired samples** for every ablation.
- **Multiple seeds**, with per-seed and aggregated reports.
- **Held-out intervention measurements** for every recovery claim.
- **Cross-fitting** of learned first stages when they feed a downstream
  estimate.
