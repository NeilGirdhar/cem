"""AFP IV problem: data sources and problem state for the synthetic IV DGP."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, KeyArray

from cem.structure.problem.data_source import DataSource, ProblemState
from cem.structure.problem.problem import Problem


class IVState(ProblemState):
    """Full state of the IV DGP, including the unobserved confounder.

    Causal graph: Z → T → Y, U → T, U → Y.

    Attributes:
        z: Instrument, shape (1,), float64.
        u: Unobserved confounder, shape (1,), float64.  Not observed during training;
            retained for offline evaluation only.
        t: Confounded treatment, shape (1,), float64.  t = alpha*z + beta*u.
        y: Outcome, shape (1,), float64.  y = gamma*t + delta*u.
    """

    z: JaxRealArray
    u: JaxRealArray
    t: JaxRealArray
    y: JaxRealArray


class IVDataSource(DataSource):
    """Generates samples from the linear IV data-generating process.

    Attributes:
        alpha: Z → T coefficient.
        beta: U → T coefficient.
        gamma: T → Y coefficient (the true causal effect).
        delta: U → Y coefficient (direct confounding effect).
    """

    alpha: float
    beta: float
    gamma: float
    delta: float

    def initial_problem_state(self, example_key: KeyArray) -> IVState:
        key_z, key_u = jr.split(example_key, 2)
        z = jr.normal(key_z, (1,)).astype(jnp.float64)
        u = jr.normal(key_u, (1,)).astype(jnp.float64)
        t = self.alpha * z + self.beta * u
        y = self.gamma * t + self.delta * u
        return IVState(z=z, u=u, t=t, y=y)


class IVProblem(Problem):
    """Synthetic instrumental-variable problem for testing AFP.

    Implements the causal graph::

        Z → T → Y
            ↑   ↑
            U───┘

    where Z is the instrument, U the unobserved confounder, T the confounded treatment,
    and Y the outcome.  The AFP model receives (z_exo=Z, z_endo=T, z_obs=Y) and should
    learn to separate exogenous (Z-driven) from endogenous (U-driven) variation.

    Attributes:
        alpha: Z → T coefficient.
        beta: U → T coefficient.
        gamma: T → Y coefficient — the true causal effect to be identified.
        delta: U → Y direct effect (confounding).
    """

    alpha: float
    beta: float
    gamma: float
    delta: float

    @property
    def exo_causal_weight(self) -> float:
        """True causal-path coefficient gamma * alpha (Z → T → Y)."""
        return self.gamma * self.alpha

    @property
    def endo_confound_weight(self) -> float:
        """Confounding coefficient gamma*beta + delta (the U contribution to Y)."""
        return self.gamma * self.beta + self.delta

    def create_data_source(self) -> IVDataSource:
        return IVDataSource(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
        )
