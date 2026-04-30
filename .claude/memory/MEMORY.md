# Memory Index

- [Use array API commands not raw jnp/numpy](feedback_array_api.md) — prefer `jnp.vecdot` over `jnp.dot`, etc.
- [Score-based v-weighting amplifies spectral loss oscillation](project_spectral_loss_vweighting.md) — why meta-learned frequency weighting via score magnitude is fundamentally broken for phase-wrapping
- [use_spectral_loss is better-conditioned for phasor supervised regression](project_use_spectral_loss_conditioning.md) — prefer spectral-gradient training or lower phasor LR when distributional target loss is unstable
