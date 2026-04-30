---
name: use_spectral_loss is better-conditioned for phasor supervised regression
description: Prefer spectral-gradient training when phasor distributional target loss is unstable
type: project
---

For supervised phasor targets, `use_spectral_loss=True` is better-conditioned than optimizing
the distributional target loss directly.

Observed on `supervised-synthetic-regression`: with the default phasor learning rate `0.01`,
the distributional-loss phasor run initially improved, then jumped and plateaued around loss
`200`. The same architecture reached perceptron-level loss when either:

- `use_spectral_loss=True` at learning rate `0.01`
- distributional loss used a smaller learning rate, around `0.003` or `0.001`

**Why:** The distributional target path converts predicted phasors back through
`z_hat.to_distribution(grid)` and optimizes the reported cross-entropy through that inverse
characteristic-function reconstruction. That gradient is learning-rate-sensitive on continuous
multi-target regression. The spectral-loss path keeps the reported distributional loss but applies
a better-conditioned phasor-space gradient.

**Implication:** If phasor supervised regression looks much worse than perceptron while iris works,
suspect conditioning before model capacity. Prefer `use_spectral_loss=True` or lower the phasor
learning rate.
