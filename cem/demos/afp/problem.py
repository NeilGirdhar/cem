"""AFP IV problem: parameterized synthetic instrumental-variable DGP.

Generalizes the original scalar IV demo to vector-valued Z, U, C, Y with optional
elementwise nonlinearity in the structural mechanisms and optional multi-environment
splits via instrument-distribution shifts.

Causal graph (per environment e)::

    Z_e → C → Y
          ↑   ↑
          U───┘

with structural equations

    C = nonlin(alpha @ Z + beta @ U)
    Y = nonlin(gamma @ C + delta @ U)

``Z`` are candidate instruments (observed, exogenous), ``C`` are candidate confounders
(observed; jointly caused by Z and the unknown confounder U), ``U`` is the unknown
confounder (latent, never observed by the model), and ``Y`` is the outcome.

Z is sampled as ``N(z_env_mean[e], I)`` and U as ``N(0, diag(u_env_std[e]**2))``.
For ``n_environments == 1`` the env channel is absent from the observation.
"""

from enum import Enum
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, KeyArray

from cem.structure.problem.data_source import DataSource, ProblemObservation, ProblemState
from cem.structure.problem.problem import Problem
from cem.transforms import encode_flat


class NonlinearityKind(Enum):
    """Elementwise nonlinearity applied after each structural linear combination."""

    none = "none"
    tanh = "tanh"
    softplus = "softplus"


def _apply_nonlinearity(x: JaxRealArray, kind: NonlinearityKind) -> JaxRealArray:
    if kind == NonlinearityKind.none:
        return x
    if kind == NonlinearityKind.tanh:
        return jnp.tanh(x)
    if kind == NonlinearityKind.softplus:
        return jax.nn.softplus(x)
    msg = f"Unknown nonlinearity {kind!r}"
    raise ValueError(msg)


class IVState(ProblemState):
    """Full state of the IV DGP, including the unknown confounder.

    Attributes:
        z: Candidate instruments, shape (n_instruments,), float64.
        u: Unknown confounder, shape (n_confounders,), float64.  Not observed
            during training; retained for offline evaluation only.
        c: Candidate confounders, shape (n_candidate_confounders,), float64.
        y: Outcomes, shape (n_outcomes,), float64.
        e: Environment index, shape (), int32.
    """

    z: JaxRealArray
    u: JaxRealArray
    c: JaxRealArray
    y: JaxRealArray
    e: JaxRealArray


class IVObservation(ProblemObservation):
    """Encoded observed variables for the IV DGP.

    ``x`` is the flat UnitVarianceNormalNP encoding of observed
    ``concat(Z, C, env_one_hot)`` and ``y`` is the encoding of observed ``Y``.  The
    env one-hot is centred (subtract ``1/n_environments``) and is omitted entirely
    when ``n_environments == 1``.  ``U`` remains in :class:`IVState` for evaluation;
    the model only receives these observed fields.
    """

    x: JaxRealArray
    y: JaxRealArray


class IVDataSource(DataSource):
    """Generates samples from the parameterized IV data-generating process.

    Attributes:
        alpha: Z → C coefficient matrix, shape (n_candidate_confounders, n_instruments).
        beta: U → C coefficient matrix, shape (n_candidate_confounders, n_confounders).
        gamma: C → Y coefficient matrix, shape (n_outcomes, n_candidate_confounders) —
            the true causal-effect matrix to be identified.
        delta: U → Y coefficient matrix, shape (n_outcomes, n_confounders).
        z_env_mean: Per-environment mean of Z, shape (n_environments, n_instruments).
        u_env_std: Per-environment standard deviation of U, shape
            (n_environments, n_confounders).
    """

    alpha: JaxRealArray
    beta: JaxRealArray
    gamma: JaxRealArray
    delta: JaxRealArray
    z_env_mean: JaxRealArray
    u_env_std: JaxRealArray
    n_instruments: int = eqx.field(static=True)
    n_confounders: int = eqx.field(static=True)
    n_candidate_confounders: int = eqx.field(static=True)
    n_outcomes: int = eqx.field(static=True)
    n_environments: int = eqx.field(static=True)
    nonlinearity: NonlinearityKind = eqx.field(static=True)

    @override
    def initial_problem_state(self, example_key: KeyArray) -> IVState:
        key_z, key_u, key_e = jr.split(example_key, 3)
        z_base = jr.normal(key_z, (self.n_instruments,)).astype(jnp.float64)
        u_base = jr.normal(key_u, (self.n_confounders,)).astype(jnp.float64)
        if self.n_environments > 1:
            e = jr.randint(key_e, (), 0, self.n_environments)
        else:
            e = jnp.asarray(0, dtype=jnp.int32)
        z = z_base + self.z_env_mean[e]
        u = u_base * self.u_env_std[e]
        c = _apply_nonlinearity(self.alpha @ z + self.beta @ u, self.nonlinearity)
        y = _apply_nonlinearity(self.gamma @ c + self.delta @ u, self.nonlinearity)
        return IVState(z=z, u=u, c=c, y=y, e=e)


class IVProblem(Problem):
    """Parameterized synthetic instrumental-variable problem for testing AFP.

    The AFP model receives observed ``(Z, C, Y)`` and (when there is more than one
    environment) a centred env one-hot, and should learn to separate exogenous
    (Z-driven) from endogenous (U-driven) variation.

    Attributes:
        alpha: Z → C coefficient matrix.
        beta: U → C coefficient matrix.
        gamma: C → Y coefficient matrix — the true causal effect to be identified.
        delta: U → Y coefficient matrix — direct confounding.
        z_env_mean: Per-environment instrument mean shift.
        u_env_std: Per-environment confounder standard deviation.
    """

    alpha: JaxRealArray
    beta: JaxRealArray
    gamma: JaxRealArray
    delta: JaxRealArray
    z_env_mean: JaxRealArray
    u_env_std: JaxRealArray
    n_instruments: int = eqx.field(static=True)
    n_confounders: int = eqx.field(static=True)
    n_candidate_confounders: int = eqx.field(static=True)
    n_outcomes: int = eqx.field(static=True)
    n_environments: int = eqx.field(static=True)
    nonlinearity: NonlinearityKind = eqx.field(static=True)

    @property
    def env_features(self) -> int:
        """Length of the env channel in the observation; 0 for a single environment."""
        return self.n_environments if self.n_environments > 1 else 0

    @property
    def obs_x_features(self) -> int:
        """Total length of the observed input vector before flat encoding."""
        return self.n_instruments + self.n_candidate_confounders + self.env_features

    @property
    def obs_y_features(self) -> int:
        """Length of the observed outcome vector before flat encoding."""
        return self.n_outcomes

    @property
    def exo_causal_weight(self) -> JaxRealArray:
        """True causal-path weight ``gamma @ alpha`` (Z → C → Y)."""
        return self.gamma @ self.alpha

    @property
    def endo_confound_weight(self) -> JaxRealArray:
        """Confounding weight ``gamma @ beta + delta`` (the U contribution to Y)."""
        return self.gamma @ self.beta + self.delta

    @override
    def create_data_source(self) -> IVDataSource:
        return IVDataSource(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            z_env_mean=self.z_env_mean,
            u_env_std=self.u_env_std,
            n_instruments=self.n_instruments,
            n_confounders=self.n_confounders,
            n_candidate_confounders=self.n_candidate_confounders,
            n_outcomes=self.n_outcomes,
            n_environments=self.n_environments,
            nonlinearity=self.nonlinearity,
        )

    @override
    def extract_observation(self, state: ProblemState) -> IVObservation:
        assert isinstance(state, IVState)
        if self.env_features > 0:
            env_one_hot = jax.nn.one_hot(state.e, self.n_environments) - 1.0 / self.n_environments
            x_raw = jnp.concatenate((state.z, state.c, env_one_hot))
        else:
            x_raw = jnp.concatenate((state.z, state.c))
        return IVObservation(x=encode_flat(x_raw), y=encode_flat(state.y))


def build_iv_problem(
    *,
    n_instruments: int,
    n_confounders: int,
    n_candidate_confounders: int,
    n_outcomes: int,
    n_environments: int,
    nonlinearity: NonlinearityKind,
    seed: int,
    scale: float,
) -> IVProblem:
    """Deterministically construct a parameterized IV problem from a seed.

    Coefficient matrices are sampled i.i.d. ``N(0, scale**2)``.  Environment shifts:
    env 0 is the observational baseline (``z_env_mean = 0``); for ``n_environments > 1``
    instrument means are spread linearly from ``-2`` to ``+2`` across environments.
    Confounder standard deviations are 1.0 in every environment.
    """
    key = jr.key(seed)
    k_alpha, k_beta, k_gamma, k_delta = jr.split(key, 4)
    alpha = jr.normal(k_alpha, (n_candidate_confounders, n_instruments)) * scale
    beta = jr.normal(k_beta, (n_candidate_confounders, n_confounders)) * scale
    gamma = jr.normal(k_gamma, (n_outcomes, n_candidate_confounders)) * scale
    delta = jr.normal(k_delta, (n_outcomes, n_confounders)) * scale
    if n_environments > 1:
        magnitudes = jnp.linspace(-2.0, 2.0, n_environments)
        z_env_mean = jnp.broadcast_to(magnitudes[:, None], (n_environments, n_instruments))
    else:
        z_env_mean = jnp.zeros((n_environments, n_instruments))
    u_env_std = jnp.ones((n_environments, n_confounders))
    return IVProblem(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        z_env_mean=z_env_mean,
        u_env_std=u_env_std,
        n_instruments=n_instruments,
        n_confounders=n_confounders,
        n_candidate_confounders=n_candidate_confounders,
        n_outcomes=n_outcomes,
        n_environments=n_environments,
        nonlinearity=nonlinearity,
    )
