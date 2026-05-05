"""AFP IV solver: adversarial factor purification model and solver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, override

import equinox as eqx
import jax.numpy as jnp
from efax import Flattener, UnitVarianceNormalNP
from jax.lax import stop_gradient
from optuna.distributions import FloatDistribution, IntDistribution
from tjax import JaxArray, JaxRealArray, RngStream, frozendict, negate_cotangent

from cem.phasor.frequency import geometric_frequencies
from cem.phasor.gated_projection import GatedProjection
from cem.phasor.loss import decorrelation_loss, spectral_reconstruction_loss_and_score
from cem.phasor.message import phasor_from_distribution
from cem.structure.graph import FixedParameter, Model, ModelResult
from cem.structure.graph.node import NodeConfiguration
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, float_field, int_field
from cem.transforms import AffineWithDropout

from .problem import IVObservation, IVProblem


class AFPConfiguration(NodeConfiguration):
    """Per-step AFP losses, stored for telemetry.

    Attributes:
        recon_loss: Per-element von Mises reconstruction cross-entropy, shape (..., obs_features).
        exo_loss: Concordance Re(exo_critic(score)^H · z_exo_pure), shape (...).
        endo_loss: Concordance Re(endo_critic(z_exo_pure)^H · z_endo_pure), shape (...).
    """

    recon_loss: JaxArray
    exo_loss: JaxArray
    endo_loss: JaxArray


class AFPModel(Model):
    """Adversarial Factor Purification (AFP) model for the IV problem.

    Separates endogenous (confounded) and exogenous (causally valid) contributions via
    two adversarial independence constraints:

    - Exogeneity:  Z_exo ⊥ Score(Z_obs, Ẑ)  — exo latents uninformative about the residual
    - Endogeneity: Z_endo ⊥ Z_exo            — endo latents uninformative about exo latents

    Gradient routing reverses the critic prediction's cotangent so minimization pushes purified
    latents away from the critic while pushing critic parameters to maximize concordance.

    Attributes:
        endo_latent: Dimension of the purified endogenous latent space.
        exo_latent: Dimension of the purified exogenous latent space.
        obs_features: Dimension of the observation space.
        endo_purifier: Gated projection from observed inputs to purified endogenous latents.
        exo_purifier: Gated projection from observed inputs to purified exogenous latents.
        endo_predictor: Log-space map from purified endogenous latents to observation space.
        exo_predictor: Log-space map from purified exogenous latents to observation space.
        exo_critic: Log-space probe detecting concordance between Z_exo_pure and Score.
        endo_critic: Log-space probe detecting concordance between Z_endo_pure and Z_exo_pure.
    """

    endo_latent: int = eqx.field(static=True)
    exo_latent: int = eqx.field(static=True)
    obs_features: int = eqx.field(static=True)
    endo_purifier: GatedProjection
    exo_purifier: GatedProjection
    endo_predictor: AffineWithDropout
    exo_predictor: AffineWithDropout
    exo_critic: GatedProjection
    endo_critic: GatedProjection
    _x_flattener: FixedParameter[Flattener[Any]]
    _y_flattener: FixedParameter[Flattener[Any]]
    _frequencies: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        *,
        endo_features: int,
        exo_features: int,
        obs_features: int,
        n_frequencies: int,
        endo_latent: int,
        exo_latent: int,
        streams: Mapping[str, RngStream],
    ) -> Self:
        freqs = geometric_frequencies(n_frequencies, base=1)
        x_flattener, _ = Flattener.flatten(
            UnitVarianceNormalNP(jnp.zeros(endo_features)), mapped_to_plane=True
        )
        y_flattener, _ = Flattener.flatten(
            UnitVarianceNormalNP(jnp.zeros(obs_features)), mapped_to_plane=True
        )
        encoded_endo_features = endo_features * n_frequencies
        encoded_exo_features = exo_features * n_frequencies
        encoded_obs_features = obs_features * n_frequencies
        return cls(
            endo_latent=endo_latent,
            exo_latent=exo_latent,
            obs_features=encoded_obs_features,
            endo_purifier=GatedProjection.create(
                encoded_endo_features, endo_latent, streams=streams
            ),
            exo_purifier=GatedProjection.create(encoded_exo_features, exo_latent, streams=streams),
            endo_predictor=AffineWithDropout.create(
                endo_latent, encoded_obs_features, streams=streams
            ),
            exo_predictor=AffineWithDropout.create(
                exo_latent, encoded_obs_features, streams=streams
            ),
            exo_critic=GatedProjection.create(encoded_obs_features, exo_latent, streams=streams),
            endo_critic=GatedProjection.create(exo_latent, endo_latent, streams=streams),
            _x_flattener=FixedParameter(x_flattener),
            _y_flattener=FixedParameter(y_flattener),
            _frequencies=FixedParameter(freqs),
        )

    def _adversarial_loss(
        self,
        critic: GatedProjection,
        u: JaxArray,
        z: JaxArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxArray:
        """Compute one adversarial loss with reversed critic cotangents.

        The primal loss is ``decorrelation_loss(critic(stop_gradient(u)), z)``. Minimizing it
        pushes ``z`` away from the critic prediction, while ``negate_cotangent`` makes the critic
        parameters maximize the same concordance.

        Args:
            critic: The adversarial critic module.
            u: Nuisance phasor.
            z: Message phasor.
            streams: RNG streams.
            inference: Whether to run in inference mode.

        Returns:
            Scalar adversarial loss contribution.
        """
        prediction = critic.infer(stop_gradient(u), streams=streams, inference=inference)
        return jnp.sum(decorrelation_loss(negate_cotangent(prediction), z))

    @override
    def infer(
        self,
        observation: object,
        state: object,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ModelResult:
        assert isinstance(observation, IVObservation)

        # Match the supervised phasor demo: unflatten UnitVarianceNormalNP encodings and
        # evaluate their characteristic phasors on the configured frequency basis.
        x_dist = self._x_flattener.value.unflatten(observation.x, return_vector=True)
        y_dist = self._y_flattener.value.unflatten(observation.y, return_vector=True)
        z_input = phasor_from_distribution(x_dist, self._frequencies.value, raveled=True)
        z_obs = phasor_from_distribution(y_dist, self._frequencies.value, raveled=True)

        # Purify: map inputs to latent representations.
        z_endo_pure = self.endo_purifier.infer(z_input, streams=streams, inference=inference)
        z_exo_pure = self.exo_purifier.infer(z_input, streams=streams, inference=inference)

        # Predict: reconstruct the observation from both pathways.
        z_endo_hat = self.endo_predictor.infer(z_endo_pure, streams=streams, inference=inference)
        z_exo_hat = self.exo_predictor.infer(z_exo_pure, streams=streams, inference=inference)
        z_hat = z_endo_hat + z_exo_hat

        # Reconstruction loss and score (∂loss/∂ẑ).
        loss_and_score = spectral_reconstruction_loss_and_score(z_obs, z_hat)
        recon_loss = loss_and_score.loss
        score = loss_and_score.score

        # Adversarial losses double as telemetry; _adversarial_loss handles gradient routing.
        exo_loss = self._adversarial_loss(
            self.exo_critic, score, z_exo_pure, streams=streams, inference=inference
        )
        endo_loss = self._adversarial_loss(
            self.endo_critic, z_exo_pure, z_endo_pure, streams=streams, inference=inference
        )
        total_loss = jnp.sum(recon_loss) + exo_loss + endo_loss

        afp_config = AFPConfiguration(
            recon_loss=recon_loss,
            exo_loss=exo_loss,
            endo_loss=endo_loss,
        )
        return ModelResult(
            loss=total_loss,
            configurations=frozendict({"afp": afp_config}),
            state=state,
        )


class AFPSolver(Solver[IVProblem]):
    """Solver for the AFP IV demo.

    Attributes:
        alpha: Z → T coefficient.
        beta: U → T coefficient.
        gamma: T → Y coefficient (true causal effect).
        delta: U → Y coefficient (direct confounding).
        endo_latent: Dimension of the endogenous latent space.
        exo_latent: Dimension of the exogenous latent space.
    """

    training_examples: int = int_field(default=5000, domain=IntDistribution(1, 1 << 17, log=True))
    training_batch_size: int = int_field(default=32, domain=IntDistribution(1, 1 << 10, log=True))
    inference_examples: int = int_field(default=500, domain=IntDistribution(1, 1 << 12, log=True))
    inference_batch_size: int = int_field(default=32, domain=IntDistribution(1, 1 << 10, log=True))
    alpha: float = float_field(default=1.5, domain=FloatDistribution(0.1, 4.0), optimize=False)
    beta: float = float_field(default=1.2, domain=FloatDistribution(0.1, 4.0), optimize=False)
    gamma: float = float_field(default=2.0, domain=FloatDistribution(0.1, 4.0), optimize=False)
    delta: float = float_field(default=1.0, domain=FloatDistribution(0.0, 4.0), optimize=False)
    endo_latent: int = int_field(default=4, domain=IntDistribution(1, 16), optimize=True)
    exo_latent: int = int_field(default=4, domain=IntDistribution(1, 16), optimize=True)
    n_frequencies: int = int_field(default=10, domain=IntDistribution(2, 16), optimize=True)

    @override
    def problem(self) -> IVProblem:
        return IVProblem(alpha=self.alpha, beta=self.beta, gamma=self.gamma, delta=self.delta)

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        del data_source, problem
        return AFPModel.create(
            endo_features=2,
            exo_features=2,
            obs_features=1,
            n_frequencies=self.n_frequencies,
            endo_latent=self.endo_latent,
            exo_latent=self.exo_latent,
            streams=streams,
        )
