# SIP rewrite plan

Tracks what still has to be done for `cem/sip/`. The stable design — classes, module layout, plug
points, tested claims — lives in [`sip.md`](sip.md).

## What is done

All circuit leaves, features, and whole-agent wiring are in place with unit tests:

- [x] Leaves: `Credit`, `Emitter`, `Mismatch`, `TDError` (`cem/sip/`).
- [x] Messages: `InstrumentedMessage` plus `concat_messages`, `concat_instrumented_messages`,
  `evidence_scale`, `mean_and_precision`, `variance_from_precision`, `promoting_link_loss`,
  `suppressing_link_loss`.
- [x] Features: `SIPFeature`, `SensationInputFeature`, `ProspectFeature`, `TDFeature`,
  `RecruiterFeature`. Identification features write their outgoing `InstrumentedMessage` into
  state via `last_message_index`; the TD feature stores its scalar `dissonance` and vector
  `advantage`.
- [x] Couplings: `ExplanatoryCoupling` marker, `Sensations` (bottom-up), `Intentions` (top-down).
  Both concat the one layer directly above into each level's predictor.
- [x] `RewardCritic` runs prospects in parallel, concats their outputs into the TD feature's
  predictor, and feeds the TD feature's previous dissonance back as every prospect's goal. One
  prospect per sensation feature and one per intention feature; the predictor instrument is the
  matching feature's instrument, so each prospect attributes reward credit through one specific
  component.
- [x] `SIPAgent` wires one `Sensations`, one `Intentions`, and one `RewardCritic` through a shared
  `eqx.nn.State`. `SIPAgent.create` takes `level_sizes` (shared interior widths; `[]` for the
  single-level agent), `env_observation_features`, and `env_action_features`; sensation level 0
  has width `env_observation_features + env_action_features`, intention level 0 has width
  `env_action_features`. Each step: sensations first with sensation-level-0 input =
  `concat(env observation, previous action)` — the eligibility trace — and intention-efference
  external predictors; intentions top-down with sensation readouts, sensation-message external
  predictors, and the TD feature's previous advantage; reward critic last with a diffuse readout
  (every sensation and intention observation) and per-feature predictor instruments. The action
  is the bottom intention's observation.
- [x] First-step prior unified via `InstrumentedMessage.initial(n)`: observation half is a
  standard normal (precision 1, so raw-observation reads are well-posed), instrument half is
  the null probe (precision 0, no fake experimental noise). Replaces the former
  `_absent_message` helper.
- [x] Curiosity and tracking channels wired into the reward critic. For `L = 1 + len(level_sizes)`
  sensation levels the agent augments the width-`reward_features` environment reward with `L`
  curiosity channels (one per sensation feature's confounding cost) and `L` tracking channels
  (one per sensation feature's reconstruction cost), so the critic predicts a width
  `reward_features + 2 * L` reward. The intention-side Credit goal is the scalar sum of the
  augmented TD advantage, broadcast to every intention level (per-level routing was ruled out
  because high-level intentions legitimately affect low-level tracking).

## What is missing

### Agent-wiring decisions the thesis still constrains

The `SIPAgent.create` and `SIPAgent.step` choices currently in the code are a workable first pass
but are *not* fully pinned down by the thesis chapter alone. Before experiments 16–17 start
depending on them, confirm or revise:

- [ ] **Advantage per level.** `SIPAgent` broadcasts `sum(td_advantage)` as each intention level's
  Credit goal. The thesis says the TD error reaches every intention level through advantage links;
  whether every level receives the same scalar, or a per-level projection, is worth checking.
- [ ] **Sensation external predictors.** The code uses the matching intention's previous
  `last_message` as every sensation level's external predictor (efference copy). The thesis only
  pins down the bottom sensation's efference wiring explicitly; whether higher-level sensations
  also receive an efference-style predictor needs a pass.
- [ ] **Intention external predictors.** The code uses the matching sensation's current
  `last_message` at every intention level. Same question as above: the thesis's figure governs,
  not this guess.

### Recruiter wiring

`ProspectFeature` and `SIPFeature` accept a `gain: InstrumentedMessage | None`, but no coupling
supplies one. Each stack needs its own decision on whether, how, and where to attach recruiters:

- [ ] **Sensation recruiters.** Where in the sensation stack does a recruiter sit, what feeds its
  observation content and predictor, and which sensation level(s) does its output gate?
- [ ] **Intention recruiters.** Same questions for the intention stack. The top of the intention
  stack is the natural place to look, but the thesis figure is what governs.
- [ ] **Prospect recruiters.** Prospect features read their gain already; decide whether a
  prospect-level recruiter stack exists at all and, if so, how it couples to the sensation /
  intention recruiters.

### Experiments and JSON outputs

`cem/sip/experiments/` is empty. The 17 numbered experiments listed in
[`sip.md`](sip.md#tested-claims) each need:

- a `Problem` / `DataSource` under `cem/sip/experiments/`, with the structural-causal-model
  generator for identification tests (1–12) or the RL `Problem` from `cem/structure/rl/` for
  value / control tests (13–17);
- a pytest in `tests/sip/experiments/` running a fast configuration and asserting the numbered
  claim;
- the full-configuration CLI entry that writes the JSON thesis `chart()` consumes.

Implementation gates (from `sip.md`): experiment 12 needs only `Emitter`; 1–12 need `Emitter`,
`Mismatch`, and `Credit`; 13–15 add `TDError`; 16–17 need the full `SIPAgent`. All the circuit
pieces are in place, so experiments can begin in any order the research sequencing prefers.

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

### Test coverage gaps

The circuit itself is well tested. Known thinner spots:

- [ ] `_predictor_for` is only exercised at N = 2 layers. Add a small N = 3 check for both
  `Sensations` and `Intentions` to catch an off-by-one in the layer-above lookup.
- [ ] `SIPAgent` has a smoke test and a shape test; no test yet exercises the TD-advantage
  feedback as intention goal or the dissonance feedback as prospect goal through the agent-level
  API.

### Docs

- [ ] [`sip.md`](sip.md) reflects the current class layout. Once the agent-wiring decisions above
  are confirmed, the `SIPAgent` section should be sharpened (per-level advantage, prospect fan-in,
  efference policy) rather than left as "a workable first pass".

### Curiosity and tracking

Open questions and follow-ups that are not blockers but want a note so they are not lost:

- [ ] **Loss scaling across environment reward, curiosity, and tracking.** The augmented reward
  currently puts the environment reward at whatever precision the environment supplies and the
  `2 * L` intrinsic channels at precision 1. These weights need an adaptive rule so the three
  signals do not drown each other out; mention the rule in the thesis once it is worked out.
- [ ] **DAG sensation / intention topology.** Replace the stack-of-levels structure with a DAG.
  More sensation features means more curiosity and tracking channels, and the DAG dissolves the
  stack's "level i sensation matches level i intention" assumption, which is what currently
  forces the TD advantage to be summed and broadcast rather than routed per-feature.
- [ ] **Controllable pseudo-rewards.** Learn a projection from the raw reward features (extrinsic
  reward plus curiosity and tracking) to pseudo-reward features chosen to be maximally
  controllable by specific intention features. Design notes are in `~/src/typst/todo.txt` under
  "Controllable pseudo-rewards".
- [ ] **Multi-loss gradient transform.** `TDError`, `Mismatch`, and the loss-weight learnables all
  currently stop-gradient their softplus readouts because Adam under a single scalar loss would
  collapse them. The adaptive rule for the weights still has to be implemented as a
  `DisGradientTransformation`; until then the weights behave as fixed zero.
- [ ] **Witness norm constraint.** The unit-RHS special case (`E[||w||²] ≤ 1`) is hardcoded in
  `Mismatch` and `TDError`. The thesis's general form has the gain's mean on the RHS; check
  whether the general form matters once recruiters supply a non-trivial gain.
- [ ] **JIT / vmap.** `SIPAgent.step` has not been exercised under `eqx.filter_jit` or vmapped
  over a batch of agents. Expect the usual pytree / static-field surprises; worth a smoke test
  before it blocks an experiment.
- [ ] **Numerical stability of `evidence_scale` at high gain.** Precision multiplies by gain, so a
  large gain pushes variance towards zero and the downstream reconstruction term towards a
  `log(variance)` singularity. Watch for this if recruiters ever drive gain above a few.
- [ ] **Prospect diffuse-readout size.** `SIPAgent` uses `2 * sum(level_sizes)` as every
  prospect's observation-content dim (every sensation and intention observation concatenated). At
  many levels this will dominate the parameter count; a lower-rank readout may be worth trying.
- [ ] **Credit trace decay.** `trace_decay_rate` defaults to 0.0 everywhere. The eligibility-trace
  story in the thesis uses a nonzero decay; pick a default once there's a real experiment to tune
  against.

## Per-piece checklist

- Module in `cem/sip/` with its unit tests in `tests/sip/`.
- Any numbered test the piece unblocks: experiment in `cem/sip/experiments/`, pytest in
  `tests/sip/experiments/`, and JSON output readable by the thesis `chart()`.
- `lefthook run pre-commit` and `pytest` pass; commit.
