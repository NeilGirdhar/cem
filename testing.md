# SIP rewrite and test plan

A working plan for aligning `cem/sip` with chapter 5 of the thesis and building the behavioural
test suite that supports it. The thesis source lives at `~/src/typst/thesis.typ`.

## Context

Chapter 5 now specifies SIP with:

- Gaussian messages,
- an emitter circuit
- a mismatch circuit
- a TD-error circuit
- a credit circuit

The current `cem/sip` is the earlier real-valued simplification: scalar observations, gain
multiplying values, a learned `noise_logits` scale, a squared-batch-moment witness objective, no
witness-gain map, no traces, no credit, no uncertainty penalty, and no instantaneous noise rule.

Goals:

- prune any dead code that doesn't meet the below objectives and isn't structure (that may be useful
  later).
- ensure that the natural parameter message section is correctly implemented and tested.
- replace that code with an implementation of the chapter as written.
- replace the tests with the 17 behavioural tests below. Each test produces JSON that the thesis
  consumes via `chart()`.

## Target code layout

In `cem/sip/`:

| Module | Contents | Replaces |
|---|---|---|
| `messages.py` | Gaussian helpers on `n × 2` natural parameters: moment ↔ natural ↔ expectation conversions, cross-entropy, evidence scaling, amplitude gating, mismatch-message encoding | — |
| `links.py` | Promoting and suppressing links as surrogate losses `stop_gradient(cotangent) · quantity`, so autodiff delivers exactly the chapter's cotangents | — |
| `emitter.py` | Observation map, bias-free instrument map, instantaneous noise top-up, shared noise injection, evidence scaling, instrument mismatch, three eligibility traces, mean credit and shrinkage, precision credit and shrinkage | `emitter.py` |
| `mismatch.py` | Target observation, prediction map, reconstruction error, observation mismatch, witness residuals, witness map × witness-gain map, confounding errors, loss weights, witness multiplier, dissonance and disutility outputs | `score.py`, `predictor_witness.py`, `objectives.py` |
| `td_error.py` | Gaussian bootstrap target, one-step delay, advantage, uncertainty penalty, unit gain, init `(0, 1)` | `td_error.py` |
| `training.py` | Online and minibatch loops using `jax.lax.scan`; per-group optimizer routing through `DisGradientTransformation` | `training.py` |
| `scenarios.py` | Parametrized SCM generator: confounding, instrument strength, heteroskedasticity, endogenous gain, chain depth, bypass paths, nonlinearity | `benchmark.py` data builders |
| `experiments/` | One module per test group, each returning a result dataclass and chart series | `benchmark.py` |

Reuse (do not reimplement):

- `cem/npn/gaussian.py`: `GaussianNPN` for the observation and prediction maps,
  `normal_from_mean_and_presence`, `efax.NormalNP`/`NormalVP` conversions.
- `cem/perceptron/mlp.py`: `MLP` for the witness, witness-gain and (if kept) inherited-variance
  maps.
- `cem/structure/graph/parameters.py`: `LearnableParameter`/`FixedParameter` wrapping for every
  array leaf; `verify_model_has_no_free_parameters` enforces this at model-creation time.
- `cem/structure/graph/disassembled.py`: `DisGradientTransformation` for per-parameter-type
  optimizer routing, including the precision-κ transform.
- `cem/commands/sip_identification.py`: JSON format consumed by thesis `chart()`; generalize into
  `sip-experiments` that writes one JSON file per test group.

`explanatory_coupling.py` is rebuilt on the new emitter and mismatch circuit. Unit tests that
exercise removed API are deleted rather than ported.

## Test suite

Each test is an experiment function in `cem/sip/experiments/` plus a pytest in `tests/sip/` that
runs a fast configuration and asserts the claim. The CLI runs the full configuration and writes
chart JSON. Methodology that applies throughout is listed after the tests.

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

### Methodology

- **Notation.** κ is precision; π is loss weights.
- **Paired samples** for every ablation.
- **Multiple seeds**, with per-seed and aggregated reports.
- **Held-out intervention measurements** for every recovery claim.
- **Cross-fitting** of learned first stages when they feed a downstream estimate.
- **Suite order.** Unit-gain versions of tests 1, 4, 8, 10, 12 first; then endogenous gain; then
  temporal dependence (11, 13–15); then joint learning (16–17).

## Existing coverage

The current `cem/sip` benchmarks are partial versions of tests 1, 2, 4 and 13. They are reusable
starting points. None yet covers a complete numbered test as specified.

| Existing section | Corresponding tests | Current coverage |
|---|---|---|
| 5.7.1 Injected noise | 4 Causal mean | A linear effect with injection off and on. Missing: the reconstruction-only versus purification comparison and the confounding and instrument-strength sweeps. The confounder is also supplied as a predictor. |
| 5.7.2 Inherited instruments | 1 Inherited instrument, plus a single-hop case of 2 | Learns an inherited instrument and recovers a downstream effect. Missing: the oracle and 2SLS comparisons and the depth sweep. |
| 5.7.3 TD-error credit | Closest to 13 Return moments | Tests scalar prediction and redistribution of credit across time. Missing: return variance, the uncertainty penalty; the witness is inactive. |

## Verification

- Per prompt: `lefthook run pre-commit` and `pytest` in `~/src/cem` pass; the ported benchmark for
  that prompt still meets its old tolerance. Then commit.
- After prompt 6: `uv run sip-experiments --output results/` writes one JSON per group.
- After prompt 12: the thesis compiles with the new charts (the user's watcher rebuilds
  `thesis.pdf`).
