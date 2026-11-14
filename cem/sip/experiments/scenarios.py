"""Structural causal models that generate data for the SIP identification experiments."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
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

    A nonzero bypass coefficient lets a parent's noise also reach the first feature's successor
    directly, which violates the exclusion restriction for the first hop. That parent must then be
    a second regressor to identify both paths.
    """

    first_stage: tuple[float, ...] = eqx.field(static=True, default=(1.0, 0.5, 0.0))
    depth: int = eqx.field(static=True, default=1)
    effect: float = eqx.field(static=True, default=1.5)
    feature_confounding: float = eqx.field(static=True, default=1.0)
    successor_confounding: float = eqx.field(static=True, default=-2.0)
    noise: float = eqx.field(static=True, default=0.5)
    bypass: tuple[float, ...] | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:  # ruff:ignore[bad-dunder-method-name]
        if self.bypass is not None and len(self.bypass) != self.n_parents:
            msg = f"Expected one bypass coefficient for each of {self.n_parents} parents"
            raise ValueError(msg)

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
            if hop == 0 and self.bypass is not None:
                successor += jnp.vecdot(jnp.asarray(self.bypass), injected_noise)
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


class ConfoundedPredictionSample(eqx.Module):
    """One draw of the parents' injected noise, the confounder, the parents, and the target."""

    injected_noise: JaxRealArray  # (k,)
    confounder: JaxRealArray  # scalar
    parents: JaxRealArray  # (k,)
    target: JaxRealArray  # scalar


class ConfoundedPredictionScenario(eqx.Module):
    """Parents with injected noise and a target that they cause, all driven by one confounder.

    Each parent is its own standard normal noise plus the confounder and a disturbance. The target
    is a structural function of the parents plus the confounder and a disturbance. Regressing the
    target on the parents is biased; each parent's noise identifies its effect.

    The structural function weights each parent by its effect, after applying tanh when the
    scenario is nonlinear. The confounder and disturbance enter additively in either case.
    """

    effects: tuple[float, ...] = eqx.field(static=True, default=(1.5,))
    parent_confounding: float = eqx.field(static=True, default=1.0)
    target_confounding: float = eqx.field(static=True, default=-2.0)
    noise: float = eqx.field(static=True, default=0.5)
    nonlinear: bool = eqx.field(static=True, default=False)

    @property
    def n_parents(self) -> int:
        return len(self.effects)

    def structural_mean(self, parents: JaxRealArray) -> JaxRealArray:
        """Return the target's mean when the parents are set by intervention, shape (...).

        Args:
            parents: The parents' values, shape (..., k).
        """
        transformed = jnp.tanh(parents) if self.nonlinear else parents
        return transformed @ jnp.asarray(self.effects)

    def observational_mean(self, parents: JaxRealArray) -> JaxRealArray:
        """Return the target's mean given the observed parents, shape (...).

        The parents carry information about the confounder. All the noise is Gaussian, so the
        confounder's conditional mean is linear in the parents.

        Args:
            parents: The parents' values, shape (..., k).
        """
        return self.structural_mean(parents) + self.target_confounding * (
            parents @ jnp.asarray(self._confounder_weights())
        )

    def causal_residual_second_moment(self, parents: JaxRealArray) -> JaxRealArray:
        """Return the target's mean squared deviation from its causal mean given the parents.

        The deviation is the confounder's and the disturbance's contribution to the target. The
        parents carry information about the confounder, so its conditional mean adds a term that
        grows with the parents. Under intervention, the mean squared deviation is instead the
        interventional variance.

        Args:
            parents: The parents' values, shape (..., k).
        """
        weights = self._confounder_weights()
        confounder_variance = 1.0 - self.parent_confounding * float(np.sum(weights))
        confounder_mean = parents @ jnp.asarray(weights)
        return (
            self.target_confounding**2 * (confounder_variance + jnp.square(confounder_mean))
            + self.noise**2
        )

    def interventional_variance(self) -> float:
        """Return the target's variance when the parents are set by intervention.

        Setting the parents leaves only the confounder and the disturbance to vary the target.
        """
        return self.target_confounding**2 + self.noise**2

    def observational_variance(self) -> float:
        """Return the target's variance about its best linear prediction from the parents.

        The parents carry information about the confounder, so this is smaller than the
        interventional variance. The closed form requires a linear structural function.
        """
        if self.nonlinear:
            msg = "The observational variance has a closed form only for linear effects"
            raise ValueError(msg)
        effects = np.asarray(self.effects)
        ones = np.ones_like(effects)
        parent_covariance = self._parent_covariance()
        confounded = self.parent_confounding * self.target_confounding
        covariance = parent_covariance @ effects + confounded * ones
        target_variance = (
            effects @ parent_covariance @ effects
            + 2.0 * confounded * float(np.sum(effects))
            + self.interventional_variance()
        )
        return float(target_variance - covariance @ np.linalg.solve(parent_covariance, covariance))

    def sample(self, key: KeyArray) -> ConfoundedPredictionSample:
        noise_key, confounder_key, parent_key, target_key = jr.split(key, 4)
        injected_noise = jr.normal(noise_key, (self.n_parents,))
        confounder = jr.normal(confounder_key, ())
        parents = (
            injected_noise
            + self.parent_confounding * confounder
            + self.noise * jr.normal(parent_key, (self.n_parents,))
        )
        target = (
            self.structural_mean(parents)
            + self.target_confounding * confounder
            + self.noise * jr.normal(target_key, ())
        )
        return ConfoundedPredictionSample(
            injected_noise=injected_noise, confounder=confounder, parents=parents, target=target
        )

    def samples(self, key: KeyArray, examples: int) -> ConfoundedPredictionSample:
        """Return independent draws, each field with a leading axis of length `examples`."""
        return jax.vmap(self.sample)(jr.split(key, examples))

    def _confounder_weights(self) -> np.ndarray:
        """Return the coefficients of the confounder's conditional mean on the parents."""
        ones = np.ones(self.n_parents)
        return self.parent_confounding * np.linalg.solve(self._parent_covariance(), ones)

    def _parent_covariance(self) -> np.ndarray:
        ones = np.ones(self.n_parents)
        return (1.0 + self.noise**2) * np.eye(self.n_parents) + (
            self.parent_confounding**2 * np.outer(ones, ones)
        )
