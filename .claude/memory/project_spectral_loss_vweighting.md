---
name: Score-based v-weighting amplifies spectral loss oscillation
description: Why meta-learned frequency weighting via score magnitude fails for the spectral reconstruction loss
type: project
---

Score magnitude cannot reliably distinguish "reliably converged" phasors from "temporarily aliased" ones — both can have small scores for different reasons.

When v concentrates weight on a phasor with low score due to phase-aliasing (prediction is wrapped to wrong phase, so score ≈ 0), the amplified gradient kicks it out of the alias on the next step, producing a large spike. This raises that phasor's score, shifts v elsewhere, and kick-starts another alias. v chases the oscillation and amplifies it — positive feedback.

Observed empirically: entropy of v declines at exactly the same rate the oscillations grow, and the oscillations grow without bound. The meta-learning makes things strictly worse.

**Why:** Phase-wrapping aliasing makes score magnitudes an unreliable signal. The approach is fundamentally broken for this purpose.

**Implication:** The right fix is to eliminate wrong-direction gradients at the source (e.g. wrapped/cosine loss), not to try to detect and downweight them after the fact via score magnitudes. The cosine loss worked (no oscillation) but had other drawbacks (magnitude/presence information ignored, worse final loss than perceptron).
