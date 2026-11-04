# SIP rewrite plan

Tracks what still has to be done for `cem/sip/`. The stable design — classes,
module layout, plug points, tested claims — lives in [`sip.md`](sip.md).

## What is done

All circuit leaves, features, and whole-agent wiring are in place with unit
tests:

- Leaves: `Credit`, `Emitter`, `Mismatch`, `TDError` (`cem/sip/`).
- Messages: `InstrumentedMessage` plus `concat_messages`,
  `concat_instrumented_messages`, `evidence_scale`, `mean_and_precision`,
  `variance_from_precision`, `promoting_link_loss`, `suppressing_link_loss`.
- Features: `SIPFeature`, `SensationInputFeature`, `ProspectFeature`,
  `TDFeature`, `RecruiterFeature`. Identification features write their
  outgoing `InstrumentedMessage` into state via `last_message_index`; the
  TD feature stores its scalar `dissonance` and vector `advantage`.
- Couplings: `ExplanatoryCoupling` marker, `Sensations` (bottom-up),
  `Intentions` (top-down). Both concat the one layer directly above into
  each level's predictor.
- `RewardCritic` runs prospects in parallel, concats their outputs into the
  TD feature's predictor, and feeds the TD feature's previous dissonance
  back as every prospect's goal.
- `SIPAgent` wires one `Sensations`, one `Intentions`, and one
  `RewardCritic` through a shared `eqx.nn.State`: sensations first with
  the environment observation and intention-efference predictors;
  intentions top-down with sensation readouts, sensation-instrument
  predictors, and the TD feature's previous advantage; reward critic last
  with a diffuse sensation + intention readout and matching-level
  predictor instruments. The action is the bottom intention's observation.

## What is missing

### Agent-wiring decisions the thesis still constrains

The `SIPAgent.create` and `SIPAgent.step` choices currently in the code
are a workable first pass but are *not* fully pinned down by the thesis
chapter alone. Before experiments 16–17 start depending on them, confirm
or revise:

- **Prospect fan-in.** The code assumes one prospect per level with its
  predictor instrument drawn from the matching-level sensation and
  intention. The thesis says "specific" instrument links without
  prescribing a layout; a prospect-per-level layout is the minimum, not
  necessarily the intended default.
- **Advantage per level.** `SIPAgent` broadcasts `sum(td_advantage)` as
  each intention level's Credit goal. The thesis says the TD error
  reaches every intention level through advantage links; whether every
  level receives the same scalar, or a per-level projection, is worth
  checking.
- **Sensation external predictors.** The code uses the matching
  intention's previous `last_message` as every sensation level's external
  predictor (efference copy). The thesis only pins down the bottom
  sensation's efference wiring explicitly; whether higher-level
  sensations also receive an efference-style predictor needs a pass.
- **Intention external predictors.** The code uses the matching
  sensation's current `last_message` at every intention level. Same
  question as above: the thesis's figure governs, not this guess.
- **Recruiter wiring.** `ProspectFeature` now accepts a gain, but neither
  `Sensations`, `Intentions`, nor `RewardCritic` ever supplies one. If
  the SIP agent uses recruiters, their stacking has to be designed and
  wired into `SIPAgent`.

### Experiments and JSON outputs

`cem/sip/experiments/` is empty. The 17 numbered experiments listed in
[`sip.md`](sip.md#tested-claims) each need:

- a `Problem` / `DataSource` under `cem/sip/experiments/`, with the
  structural-causal-model generator for identification tests (1–12) or
  the RL `Problem` from `cem/structure/rl/` for value / control tests
  (13–17);
- a pytest in `tests/sip/experiments/` running a fast configuration and
  asserting the numbered claim;
- the full-configuration CLI entry that writes the JSON thesis `chart()`
  consumes.

Implementation gates (from `sip.md`): experiment 12 needs only `Emitter`;
1–12 need `Emitter + Mismatch + Credit`; 13–15 add `TDError`; 16–17 need
the full `SIPAgent`. All the circuit pieces are in place, so experiments
can begin in any order the research sequencing prefers.

### Test coverage gaps

The circuit itself is well tested. Known thinner spots:

- `_predictor_for` is only exercised at N = 2 layers. Add a small N = 3
  check for both `Sensations` and `Intentions` to catch an off-by-one in
  the layer-above lookup.
- `SIPAgent` has a smoke test and a shape test; no test yet exercises
  the TD-advantage feedback as intention goal or the dissonance feedback
  as prospect goal through the agent-level API.

### Docs

[`sip.md`](sip.md) reflects the current class layout. Once the
agent-wiring decisions above are confirmed, the `SIPAgent` section of
`sip.md` should be sharpened (per-level advantage, prospect fan-in,
efference policy) rather than left as "a workable first pass".

## Per-piece checklist

- Module in `cem/sip/` with its unit tests in `tests/sip/`.
- Any numbered test the piece unblocks: experiment in
  `cem/sip/experiments/`, pytest in `tests/sip/experiments/`, and JSON
  output readable by the thesis `chart()`.
- `lefthook run pre-commit` and `pytest` pass; commit.
