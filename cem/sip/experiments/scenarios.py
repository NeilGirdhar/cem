"""Structural causal models that generate data for the SIP identification experiments."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, KeyArray


class InstrumentChainSample(eqx.Module):
    """One draw of the parents' injected noise, the confounders, and the chain they drive."""

    injected_noise: JaxRealArray  # (k,), one entry for each parent
    confounders: JaxRealArray  # (depth,), one for each hop
    oracle_first_stages: JaxRealArray  # (depth,), each feature's mean given the injected noise
    features: JaxRealArray  # (depth + 1,), the chain's features and then the final outcome


class InstrumentChainScenario(eqx.Module):
    """A chain of non-controllable features that inherit experimental variation from its parents.

    Each parent injects standard normal noise, which reaches the first feature through the
    first-stage coefficients. Each feature causes the next with the same effect, and the last
    feature causes a final outcome. Hop d is the effect of feature d on its successor. A standard
    normal confounder for each hop drives both of its features, so regressing the successor on the
    feature is biased. The noise reaches each feature's successor only through that feature, so it
    identifies every hop's effect. A zero first-stage coefficient makes a parent irrelevant.
    """

    first_stage: tuple[float, ...] = eqx.field(static=True, default=(1.0, 0.5, 0.0))
    depth: int = eqx.field(static=True, default=1)
    effect: float = eqx.field(static=True, default=1.5)
    feature_confounding: float = eqx.field(static=True, default=1.0)
    successor_confounding: float = eqx.field(static=True, default=-2.0)
    noise: float = eqx.field(static=True, default=0.5)

    @property
    def n_parents(self) -> int:
        return len(self.first_stage)

    def sample(self, key: KeyArray) -> InstrumentChainSample:
        noise_key, confounder_key, disturbance_key = jr.split(key, 3)
        injected_noise = jr.normal(noise_key, (self.n_parents,))
        confounders = jr.normal(confounder_key, (self.depth,))
        disturbances = self.noise * jr.normal(disturbance_key, (self.depth + 1,))
        first_stage = jnp.vecdot(jnp.asarray(self.first_stage), injected_noise)
        features = [first_stage + self.feature_confounding * confounders[0] + disturbances[0]]
        oracle_first_stages = [first_stage]
        for hop in range(self.depth):
            successor = (
                self.effect * features[-1]
                + self.successor_confounding * confounders[hop]
                + disturbances[hop + 1]
            )
            if hop + 1 < self.depth:
                successor += self.feature_confounding * confounders[hop + 1]
                oracle_first_stages.append(self.effect * oracle_first_stages[-1])
            features.append(successor)
        return InstrumentChainSample(
            injected_noise=injected_noise,
            confounders=confounders,
            oracle_first_stages=jnp.stack(oracle_first_stages),
            features=jnp.stack(features),
        )
