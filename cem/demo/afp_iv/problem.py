from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

import jax.numpy as jnp
import jax.random as jr
from efax import ExpectationParametrization, UnitVarianceNormalEP
from tjax import JaxArray, KeyArray, jit

from cem.structure.problem.data_source import DataSource, ProblemObservation, ProblemState
from cem.structure.problem.problem import Problem


class IVState(ProblemState):
    """Full state of the IV DGP, including the unobserved confounder.

    Attributes:
        z: Instrument, shape (1,), float64.
        u: Unobserved confounder, shape (1,), float64. Not observed by the model;
            retained in state for offline evaluation only.
        t: Confounded treatment, shape (1,), float64.  t = alpha*z + beta*u.
        y: Outcome, shape (1,), float64.  y = gamma*t + delta*u.
    """

    z: JaxArray
    u: JaxArray
    t: JaxArray
    y: JaxArray


class IVObservation(ProblemObservation):
    """What is visible during training: instrument, treatment, and outcome.

    Attributes:
        z: Instrument, shape (1,), float64.
        t: Confounded treatment, shape (1,), float64.
        y: Outcome, shape (1,), float64.
    """

    z: JaxArray
    t: JaxArray
    y: JaxArray


class IVDataSource(DataSource):
    """Generates samples from the IV DGP.

    Causal graph: Z → T → Y,  U → T,  U → Y.

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

    @override
    @jit
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

    where Z is the (true) instrument, U the unobserved confounder, T the confounded
    treatment, and Y the outcome.  All variables are scalar complex phasors (shape (1,)).

    DGP (linear in natural-parameter space)::

        Z, U ~ CN(0, 1)
        T = alpha * Z + beta * U
        Y = gamma * T + delta * U
          = gamma*alpha * Z  +  (gamma*beta + delta) * U

    AFP receives (z_exo=Z, z_endo=T, z_obs=Y) and should learn to separate exogenous
    (Z-driven) from endogenous (U-driven) variation.  At the adversarial equilibrium:

    - The purified exogenous representation Z_exo captures variation driven by Z only.
    - The purified endogenous representation Z_endo is uninformative about Z_exo.
    - The exo pathway identifies the causal path coefficient gamma*alpha.

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
    def endo_confound_weight(self) -> complex:
        """Confounding coefficient gamma*beta + delta (the U contribution to Y)."""
        return self.gamma * self.beta + self.delta

    @override
    def observation_distributions(self) -> Mapping[str, ExpectationParametrization[Any]]:
        zero = jnp.zeros(1)
        return {
            "z": UnitVarianceNormalEP(zero),
            "t": UnitVarianceNormalEP(zero),
            "y": UnitVarianceNormalEP(zero),
        }

    @override
    def extract_observation(self, state: ProblemState) -> IVObservation:
        assert isinstance(state, IVState)
        return IVObservation(z=state.z, t=state.t, y=state.y)

    @override
    def create_data_source(self) -> IVDataSource:
        return IVDataSource(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
        )
