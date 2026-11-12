"""Structural causal models that generate data for the SIP identification experiments."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, KeyArray


class InheritedInstrumentSample(eqx.Module):
    """One draw of the parents' injected noise, the confounder, and the features they drive."""

    injected_noise: JaxRealArray  # (k,), one entry for each parent
    confounder: JaxRealArray  # scalar
    oracle_first_stage: JaxRealArray  # scalar, the child's mean given the injected noise
    child: JaxRealArray  # scalar
    downstream: JaxRealArray  # scalar


class InheritedInstrumentScenario(eqx.Module):
    """A non-controllable child that inherits its experimental variation from its parents.

    Each parent injects standard normal noise. The noise reaches the child through the first-stage
    coefficients and reaches the downstream feature only through the child. A standard normal
    confounder drives both the child and the downstream feature, so regressing the downstream
    feature on the child is biased. The parents' noise identifies the child's effect on the
    downstream feature. A zero coefficient makes a parent irrelevant to the child.
    """

    first_stage: tuple[float, ...] = eqx.field(static=True, default=(1.0, 0.5, 0.0))
    child_confounding: float = eqx.field(static=True, default=1.0)
    child_noise: float = eqx.field(static=True, default=0.5)
    effect: float = eqx.field(static=True, default=1.5)
    downstream_confounding: float = eqx.field(static=True, default=-2.0)
    downstream_noise: float = eqx.field(static=True, default=0.5)

    @property
    def n_parents(self) -> int:
        return len(self.first_stage)

    def sample(self, key: KeyArray) -> InheritedInstrumentSample:
        noise_key, confounder_key, child_key, downstream_key = jr.split(key, 4)
        injected_noise = jr.normal(noise_key, (self.n_parents,))
        confounder = jr.normal(confounder_key, ())
        oracle_first_stage = jnp.vecdot(jnp.asarray(self.first_stage), injected_noise)
        child = (
            oracle_first_stage
            + self.child_confounding * confounder
            + self.child_noise * jr.normal(child_key, ())
        )
        downstream = (
            self.effect * child
            + self.downstream_confounding * confounder
            + self.downstream_noise * jr.normal(downstream_key, ())
        )
        return InheritedInstrumentSample(
            injected_noise=injected_noise,
            confounder=confounder,
            oracle_first_stage=oracle_first_stage,
            child=child,
            downstream=downstream,
        )
