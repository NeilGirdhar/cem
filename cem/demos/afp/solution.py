"""AFP IV solver: adversarial factor purification model and solver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import Flattener, UnitVarianceNormalNP
from optuna.distributions import FloatDistribution, IntDistribution
from tjax import JaxArray, JaxRealArray, RngStream, frozendict

from cem.phasor.frequency import geometric_frequencies
from cem.phasor.gated_projection import GatedProjection, LogSpaceProjection
from cem.phasor.loss import decorrelation_loss, spectral_reconstruction_loss_and_score
from cem.phasor.message import PhasorMessage
from cem.structure.graph import FixedParameter, Model, ModelResult
from cem.structure.graph.node import NodeConfiguration
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, float_field, int_field

from .problem import IVObservation, IVProblem


class AFPConfiguration(NodeConfiguration):
    """Per-step AFP losses, stored for telemetry.

    Attributes:
        recon_loss: Per-element von Mises reconstruction cross-entropy, shape (..., obs_features).
        exo_loss: Alignment Re(exo_critic(z_exo_pure)^H · score), shape (...).
        endo_loss: Alignment Re(endo_critic(z_endo_pure)^H · z_exo_pure), shape (...).
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

    Gradient routing uses the double-computation trick to properly implement min-max training
    within a single scalar loss:

    - Purifier losses: gradient flows through z_*_pure (critic parameters are stopped).
    - Critic losses: gradient flows through critic parameters (z_*_pure is stopped),
      with a sign flip so that minimizing the total loss maximizes critic alignment.

    Attributes:
        endo_latent: Dimension of the purified endogenous latent space.
        exo_latent: Dimension of the purified exogenous latent space.
        obs_features: Dimension of the observation space.
        endo_purifier: Gated projection from observed inputs to purified endogenous latents.
        exo_purifier: Gated projection from observed inputs to purified exogenous latents.
        endo_predictor: Log-space map from purified endogenous latents to observation space.
        exo_predictor: Log-space map from purified exogenous latents to observation space.
        exo_critic: Log-space probe detecting alignment between Z_exo_pure and Score.
        endo_critic: Log-space probe detecting alignment between Z_endo_pure and Z_exo_pure.
    """

    endo_latent: int = eqx.field(static=True)
    exo_latent: int = eqx.field(static=True)
    obs_features: int = eqx.field(static=True)
    endo_purifier: GatedProjection
    exo_purifier: GatedProjection
    endo_predictor: LogSpaceProjection
    exo_predictor: LogSpaceProjection
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
        num_groups: int,
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
                encoded_endo_features, endo_latent, num_groups, streams=streams
            ),
            exo_purifier=GatedProjection.create(
                encoded_exo_features, exo_latent, num_groups, streams=streams
            ),
            endo_predictor=LogSpaceProjection.create(
                endo_latent, encoded_obs_features, streams=streams
            ),
            exo_predictor=LogSpaceProjection.create(
                exo_latent, encoded_obs_features, streams=streams
            ),
            exo_critic=GatedProjection.create(
                exo_latent, encoded_obs_features, num_groups, streams=streams
            ),
            endo_critic=GatedProjection.create(
                endo_latent, exo_latent, num_groups, streams=streams
            ),
            _x_flattener=FixedParameter(x_flattener),
            _y_flattener=FixedParameter(y_flattener),
            _frequencies=FixedParameter(freqs),
        )

    def _adversarial_loss(
        self,
        critic: GatedProjection,
        z_pure: JaxArray,
        target: JaxArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxArray:
        """Compute the combined adversarial loss for one critic/purifier pair.

        Returns purifier_loss + critic_loss, where:

        - purifier_loss: gradient flows through z_pure (critic params stopped).
        - critic_loss: gradient flows through critic params (z_pure stopped), negated so
          that minimising the total loss makes the critic *maximise* alignment.

        Args:
            critic: The adversarial critic module.
            z_pure: Purified latent representation, shape (..., latent).
            target: Target phasors to align against (stop_gradient applied internally).
            streams: RNG streams.
            inference: Whether to run in inference mode.

        Returns:
            Scalar adversarial loss contribution.
        """
        sg_target = jax.lax.stop_gradient(target)
        purifier_loss = decorrelation_loss(
            jax.lax.stop_gradient(critic)
            .infer(PhasorMessage(z_pure), streams=streams, inference=inference)
            .data,
            sg_target,
        )
        critic_loss = -decorrelation_loss(
            critic.infer(
                PhasorMessage(jax.lax.stop_gradient(z_pure)),
                streams=streams,
                inference=inference,
            ).data,
            sg_target,
        )
        return jnp.mean(purifier_loss) + jnp.mean(critic_loss)

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
        z_input = PhasorMessage.from_distribution(
            x_dist, self._frequencies.value, raveled=True
        ).data
        z_obs = PhasorMessage.from_distribution(y_dist, self._frequencies.value, raveled=True).data

        # Purify: map inputs to latent representations.
        z_endo_pure = self.endo_purifier.infer(
            PhasorMessage(z_input), streams=streams, inference=inference
        ).data
        z_exo_pure = self.exo_purifier.infer(
            PhasorMessage(z_input), streams=streams, inference=inference
        ).data

        # Predict: reconstruct the observation from both pathways.
        z_endo_hat = self.endo_predictor.infer(z_endo_pure)
        z_exo_hat = self.exo_predictor.infer(z_exo_pure)
        z_hat = z_endo_hat + z_exo_hat

        # Reconstruction loss and score (∂loss/∂ẑ).
        loss_and_score = spectral_reconstruction_loss_and_score(
            PhasorMessage(z_obs), PhasorMessage(z_hat)
        )
        recon_loss = loss_and_score.loss
        s = loss_and_score.score.data

        # Monitoring losses (stop_gradient so they don't affect training).
        exo_loss: JaxArray = jax.lax.stop_gradient(
            decorrelation_loss(
                self.exo_critic.infer(
                    PhasorMessage(z_exo_pure), streams=streams, inference=inference
                ).data,
                jax.lax.stop_gradient(s),
            )
        )
        endo_loss: JaxArray = jax.lax.stop_gradient(
            decorrelation_loss(
                self.endo_critic.infer(
                    PhasorMessage(z_endo_pure), streams=streams, inference=inference
                ).data,
                jax.lax.stop_gradient(z_exo_pure),
            )
        )

        total_loss = (
            jnp.mean(recon_loss)
            + self._adversarial_loss(
                self.exo_critic, z_exo_pure, s, streams=streams, inference=inference
            )
            + self._adversarial_loss(
                self.endo_critic, z_endo_pure, z_exo_pure, streams=streams, inference=inference
            )
        )

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
        num_groups: Number of rivalry groups for the nonlinear layers.
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
    num_groups: int = int_field(default=2, domain=IntDistribution(1, 8), optimize=True)
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
            num_groups=self.num_groups,
            streams=streams,
        )
