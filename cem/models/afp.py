from __future__ import annotations

from collections.abc import Mapping
from typing import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from tjax import JaxArray, RngStream

from cem.loss import decorrelation_loss, reconstruction_loss, score
from cem.structure.graph import Kernel, NodeConfiguration
from cem.transforms import Linear, Nonlinear


class AFPConfiguration(NodeConfiguration):
    """Configuration produced by the AFP forward pass.

    Attributes:
        z_hat: Combined prediction phasors, shape (..., obs_features).
        z_endo_pure: Purified endogenous latents, shape (..., endo_latent).
        z_exo_pure: Purified exogenous latents, shape (..., exo_latent).
        recon_loss: Von Mises cross-entropy, shape (..., obs_features).
        exo_loss: Exogeneity alignment Re(h_exo^H Score), shape (...).
        endo_loss: Endogeneity alignment Re(h_endo^H Z_exo), shape (...).
    """

    z_hat: JaxArray
    z_endo_pure: JaxArray
    z_exo_pure: JaxArray
    recon_loss: JaxArray
    exo_loss: JaxArray
    endo_loss: JaxArray

    @override
    def total_loss(self) -> JaxArray:
        return jnp.mean(self.recon_loss)


class AFP(Kernel[AFPConfiguration]):
    """Adversarial Factor Purification (AFP).

    Separates endogenous (confounded) and exogenous (causally valid) contributions to a
    prediction by enforcing two independence constraints via adversarial critics:

      - Exogeneity:  Z_exo ⊥ Score(Z_obs, Ẑ)  — exo latents uninformative about the residual
      - Endogeneity: Z_endo ⊥ Z_exo            — endo latents uninformative about exo latents

    Critics are linear probes trained to maximise alignment with their respective targets;
    purifiers are trained to minimise it.  Stop-gradients on the critic targets prevent the
    model from satisfying the independence constraints by reshaping predictions rather than
    by purifying the representations.

    Gradient routing for AFPConfiguration losses:
      - recon_loss: minimised by endo_purifier, exo_purifier, endo_predictor, exo_predictor.
      - exo_loss:   maximised by exo_critic; minimised by exo_purifier.
      - endo_loss:  maximised by endo_critic; minimised by endo_purifier.

    Attributes:
        endo_latent: Dimension of the purified endogenous latent space.
        exo_latent: Dimension of the purified exogenous latent space.
        obs_features: Dimension of the observation space.
        endo_purifier: Nonlinear map from endogenous inputs to purified latents.
        exo_purifier: Nonlinear map from exogenous inputs to purified latents.
        endo_predictor: Linear map from purified endogenous latents to observation space.
        exo_predictor: Linear map from purified exogenous latents to observation space.
        exo_critic: Linear probe — detects alignment between Z_exo_pure and Score.
        endo_critic: Linear probe — detects alignment between Z_endo_pure and Z_exo_pure.
    """

    endo_latent: int = eqx.field(static=True)
    exo_latent: int = eqx.field(static=True)
    obs_features: int = eqx.field(static=True)
    endo_purifier: Nonlinear
    exo_purifier: Nonlinear
    endo_predictor: Linear
    exo_predictor: Linear
    exo_critic: Nonlinear
    endo_critic: Nonlinear

    @classmethod
    def create(
        cls,
        *,
        endo_features: int,
        exo_features: int,
        obs_features: int,
        endo_latent: int,
        exo_latent: int,
        num_groups: int,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            endo_latent=endo_latent,
            exo_latent=exo_latent,
            obs_features=obs_features,
            endo_purifier=Nonlinear.create(endo_features, endo_latent, num_groups, streams=streams),
            exo_purifier=Nonlinear.create(exo_features, exo_latent, num_groups, streams=streams),
            endo_predictor=Linear.create(endo_latent, obs_features, streams=streams),
            exo_predictor=Linear.create(exo_latent, obs_features, streams=streams),
            exo_critic=Nonlinear.create(exo_latent, obs_features, num_groups, streams=streams),
            endo_critic=Nonlinear.create(endo_latent, exo_latent, num_groups, streams=streams),
        )

    @override
    def zero_configuration(self) -> AFPConfiguration:
        return AFPConfiguration(
            z_hat=jnp.zeros(self.obs_features, dtype=jnp.complex128),
            z_endo_pure=jnp.zeros(self.endo_latent, dtype=jnp.complex128),
            z_exo_pure=jnp.zeros(self.exo_latent, dtype=jnp.complex128),
            recon_loss=jnp.zeros(self.obs_features),
            exo_loss=jnp.zeros(()),
            endo_loss=jnp.zeros(()),
        )

    @override
    def infer(
        self,
        *,
        z_endo: list[object],
        z_exo: list[object],
        z_obs: list[object],
    ) -> AFPConfiguration:  # ty: ignore
        z_endo_arr: JaxArray = z_endo[0]  # ty: ignore
        z_exo_arr: JaxArray = z_exo[0]  # ty: ignore
        z_obs_arr: JaxArray = z_obs[0]  # ty: ignore

        z_endo_pure = self.endo_purifier.infer(z_endo_arr)
        z_exo_pure = self.exo_purifier.infer(z_exo_arr)

        z_hat = self.endo_predictor.infer(z_endo_pure) + self.exo_predictor.infer(z_exo_pure)

        recon = reconstruction_loss(z_obs_arr, z_hat)

        s = score(z_obs_arr, z_hat)
        exo_loss = decorrelation_loss(
            self.exo_critic.infer(z_exo_pure),
            jax.lax.stop_gradient(s),
        )
        endo_loss = decorrelation_loss(
            self.endo_critic.infer(z_endo_pure),
            jax.lax.stop_gradient(z_exo_pure),
        )

        return AFPConfiguration(
            z_hat=z_hat,
            z_endo_pure=z_endo_pure,
            z_exo_pure=z_exo_pure,
            recon_loss=recon,
            exo_loss=exo_loss,
            endo_loss=endo_loss,
        )

    @override
    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        assert isinstance(node_configuration, AFPConfiguration)
        match field_name:
            case "z_hat":
                return node_configuration.z_hat
            case "z_endo_pure":
                return node_configuration.z_endo_pure
            case "z_exo_pure":
                return node_configuration.z_exo_pure
            case "recon_loss":
                return node_configuration.recon_loss
            case "exo_loss":
                return node_configuration.exo_loss
            case "endo_loss":
                return node_configuration.endo_loss
            case _:
                msg = f"AFP does not expose output field {field_name!r}"
                raise ValueError(msg)
