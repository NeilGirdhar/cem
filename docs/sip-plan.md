# SIP rewrite and test plan

A working reference for aligning `cem/sip` with chapter 5 of the thesis
(`~/src/typst/thesis.typ`) and the 17 behavioural tests that cover it.

## Working style

Interactive, piece by piece. Each piece ships its module, its unit tests, any numbered
experiment it unblocks, and the JSON output that the thesis consumes via `chart()`. Order is
set turn by turn, not pre-sequenced.

## Classes

Naming convention: bare names for domain-specific nouns (`Emitter`, `Mismatch`, …); the `SIP`
prefix only for generic ones that would otherwise collide with common ML terms (`SIPFeature`,
`SIPValueModel`, `SIPAgent`). `ExplanatoryCoupling` is a thesis term and takes no prefix.

Circuit sub-modules (leaves first):

- `Credit` — eligibility-based update rule;
  `update(traces, stimulus, response, goal) → (new_traces, cotangent)`.
    - Traces live in the recurrent `state` carry, not in the module.
- `Mismatch` — reconstruction and observation errors, witness and witness-gain maps, loss
  weights, dissonance.
- `TDError` — Gaussian bootstrap target, one-step delay, advantage, uncertainty penalty.
    - Delay buffer lives in the recurrent `state` carry.
- `Emitter` — forward dynamics (observation map, instrument map, noise top-up, injection).
    - Contains `Credit`.

Features (per causal node):

- `SIPFeature` — one identification node: `Emitter` + `Mismatch`. Plays the "prospect feature"
  role when wired into `SIPValueModel`.
- `TDFeature` — one value node: `Emitter` + `TDError`.

`Model` subclasses:

- `ExplanatoryCoupling` — wired stack of `SIPFeature`s; identification `Model` for tests 1–12,
  16. Also reused as a sub-`Module` inside `SIPAgent`.
- `SIPValueModel` — collection of prospect `SIPFeature`s + one `TDFeature`; tests 13–15.
- `SIPAgent` — two `ExplanatoryCoupling`s (observation and intention; the intention coupling
  is the policy) + one `SIPValueModel`; test 17.

## Target code layout

- `cem/sip/` — the circuit.
    - `messages.py` — SIP-specific Gaussian helpers (evidence scaling, amplitude gating,
      mismatch-message encoding). Pure functions, no classes.
    - `credit.py` — `Credit`.
    - `mismatch.py` — `Mismatch`.
    - `td_error.py` — `TDError`.
    - `emitter.py` — `Emitter`.
    - `features.py` — `SIPFeature`, `TDFeature`.
    - `models.py` — `ExplanatoryCoupling`, `SIPValueModel`, `SIPAgent`.
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

Implementation gates: `Emitter` alone covers test 12; `Emitter + Mismatch + Credit` covers tests
1–12; `TDError` adds 13–15; tests 16–17 wire the full stack together and need no new modules.

## SIP-specific plug points

The framework (`Problem` / `Model` / `Solver` / `Plotter` / `Experiment`) is documented in
[`architecture.md`](architecture.md). What's specific to SIP:

- Identification tests (1–12) use a feed-forward `Problem` over a structural causal model;
  value-learning and control tests (13–17) use the RL `Problem` in `cem/structure/rl/`.
- The circuit needs the precision-κ transform on `DisGradientTransformation`.
- Reuses `cem/npn/gaussian.py` (`GaussianNPN`, `normal_from_mean_and_presence`),
  `cem/perceptron/mlp.py` (`MLP` for witness, witness-gain, and inherited-variance maps), and
  `cem/transforms/` as needed. Gaussian natural-parameter conversions come from `efax`
  (`NormalNP` / `NormalVP`).

## Test suite

Each test is an experiment function in `cem/sip/experiments/` plus a pytest in `tests/sip/` that
runs a fast configuration and asserts the claim. The CLI runs the full configuration and writes
chart JSON.

### Instruments

1. **Inherited instrument.** On held-out samples, compare the learned first-stage strength with
   the oracle `E[y | parent instruments]`, and run 2SLS with it to recover the downstream effect.
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
    credit must sum its contributions correctly. Include the counterexample recorded in
    `~/src/typst/todo.txt` as a condition the test must either pass or reveal.
12. **Noise generation (emitter test).** The rule is instantaneous:
    `V_local = max(0, σ̃²_inh − μ̃²_inh)`. Check (i) the conditional local variance against this
    rule, (ii) that the same perturbation enters observation and instrument, and (iii) the squared
    instrument displacement before gain:
    `E[μ̃²_ins | inherited message] = max(σ̃²_inh, μ̃²_inh)`.
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

## Per-piece checklist

- Module in `cem/sip/` with its unit tests in `tests/sip/`.
- Any numbered test the piece unblocks: experiment in `cem/sip/experiments/`, pytest in
  `tests/sip/experiments/`, and JSON output readable by thesis `chart()`.
- `lefthook run pre-commit` and `pytest` pass; commit.
