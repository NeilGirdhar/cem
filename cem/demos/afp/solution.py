"""AFP IV solver: adversarial factor purification model and solver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, override

import equinox as eqx
import jax.numpy as jnp
from efax import Flattener, UnitVarianceNormalNP
from jax.lax import stop_gradient
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from tjax import JaxArray, JaxRealArray, RngStream, frozendict, negate_cotangent

from cem.phasor.frequency import geometric_frequencies
from cem.phasor.gated_projection import GatedProjection
from cem.phasor.loss import decorrelation_loss, spectral_reconstruction_loss_and_score
from cem.structure.graph import FixedParameter, Model, ModelResult
from cem.structure.graph.node import NodeConfiguration
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, float_field, hardware_friendly_ints, int_field
from cem.transforms import AffineWithDropout, encode_phasor

from .problem import IVObservation, IVProblem, NonlinearityKind, build_iv_problem


class AFPConfiguration(NodeConfiguration):
    """Per-example AFP objective terms, stored for telemetry.

    Each field is a summed scalar objective term for one model inference.  Telemetry accumulation
    may add leading episode/example axes to these arrays.

    Attributes:
        recon_loss: Summed reconstruction objective, scalar before telemetry stacking.
        exo_loss: Summed exogeneity objective from
            Re(exo_critic(score)^H · z_exo_pure), scalar before telemetry stacking.
        endo_loss: Summed endogenous-separation objective from
            Re(endo_critic(z_exo_pure)^H · z_endo_pure), scalar before telemetry stacking.
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

    def predict_endo_phasor(
        self,
        observation: IVObservation,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxArray:
        """Run only the endogenous channel and return ``z_endo_hat``.

        Encodes ``observation.x`` as input phasors, passes them through the
        endogenous purifier and predictor.  Decoded value tracks the confounded
        (U-driven) contribution to Y.
        """
        z_input = encode_phasor(observation.x, self._x_flattener.value, self._frequencies.value)
        z_endo_pure = self.endo_purifier.infer(z_input, streams=streams, inference=inference)
        return self.endo_predictor.infer(z_endo_pure, streams=streams, inference=inference)

    def predict_exo_phasor(
        self,
        observation: IVObservation,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxArray:
        """Run only the exogenous channel and return ``z_exo_hat``.

        Encodes ``observation.x`` as input phasors, passes them through the
        exogenous purifier and predictor.  Decoded value tracks the causal-pathway
        contribution ``γα·Z`` — what the structural causal effect contributes to Y.
        """
        z_input = encode_phasor(observation.x, self._x_flattener.value, self._frequencies.value)
        z_exo_pure = self.exo_purifier.infer(z_input, streams=streams, inference=inference)
        return self.exo_predictor.infer(z_exo_pure, streams=streams, inference=inference)

    def predict_phasor(
        self,
        observation: IVObservation,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxArray:
        """Return the full predicted output phasor ``z_endo_hat + z_exo_hat``."""
        return self.predict_endo_phasor(
            observation, streams=streams, inference=inference
        ) + self.predict_exo_phasor(observation, streams=streams, inference=inference)

    def _decode_phasor_to_y(self, z_hat: JaxArray) -> JaxRealArray:
        """OLS recovery of the von Mises mean from a raveled output phasor.

        For each outcome ``k`` and frequency ``ω_j``, a true phasor satisfies
        ``z[k, j] = exp(i·ω_j·y_k − ω_j²/2)``, so ``arg z[k, j] = ω_j·y_k`` (modulo
        ``2π``).  A least-squares fit over frequencies gives
        ``y_k = Σ_j ω_j arg z[k, j] / Σ_j ω_j²``.
        """
        freqs = self._frequencies.value
        z_grid = z_hat.reshape(-1, freqs.shape[0])  # (n_outcomes, n_frequencies)
        return jnp.sum(jnp.angle(z_grid) * freqs[None, :], axis=-1) / jnp.sum(freqs**2)

    def predict_y_exo(
        self,
        observation: IVObservation,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Decoded ŷ from the exogenous channel — the AFP estimate of ``γα·Z``."""
        z_hat = self.predict_exo_phasor(observation, streams=streams, inference=inference)
        return self._decode_phasor_to_y(z_hat)

    def predict_y_endo(
        self,
        observation: IVObservation,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Decoded ŷ from the endogenous channel.

        The AFP estimate of the confounded residual ``Y − γα·Z``.
        """
        z_hat = self.predict_endo_phasor(observation, streams=streams, inference=inference)
        return self._decode_phasor_to_y(z_hat)

    def _adversarial_loss(
        self,
        critic: GatedProjection,
        u: JaxArray,
        z: JaxArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxArray:
        """Compute one summed adversarial objective with reversed critic cotangents.

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
            Scalar per-example adversarial objective contribution.
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
        z_input = encode_phasor(observation.x, self._x_flattener.value, self._frequencies.value)
        z_obs = encode_phasor(observation.y, self._y_flattener.value, self._frequencies.value)

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

        # Objective terms are also stored for telemetry; _adversarial_loss handles gradient routing.
        exo_loss = self._adversarial_loss(
            self.exo_critic, score, z_exo_pure, streams=streams, inference=inference
        )
        endo_loss = self._adversarial_loss(
            self.endo_critic, z_exo_pure, z_endo_pure, streams=streams, inference=inference
        )
        total_loss = recon_loss + exo_loss + endo_loss

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
    """Solver for the AFP IV demo on a parameterized synthetic IV problem.

    Structural fields (``n_instruments``, ``n_confounders``, ``n_candidate_confounders``,
    ``n_outcomes``, ``n_environments``, ``nonlinearity``) configure the DGP.  The
    coefficient matrices are sampled deterministically from ``coefficient_seed``
    with i.i.d. ``N(0, coefficient_scale**2)`` entries.

    Attributes:
        endo_latent: Dimension of the purified endogenous latent space.
        exo_latent: Dimension of the purified exogenous latent space.
        n_frequencies: Number of phasor frequencies in the encoded representation.
    """

    training_examples: int = int_field(default=3000, domain=IntDistribution(1, 1 << 17, log=True))
    inference_examples: int = int_field(default=0, domain=IntDistribution(0, 1 << 12))
    n_instruments: int = int_field(default=2, domain=IntDistribution(1, 8))
    n_confounders: int = int_field(default=1, domain=IntDistribution(1, 4))
    n_candidate_confounders: int = int_field(default=1, domain=IntDistribution(1, 4))
    n_outcomes: int = int_field(default=1, domain=IntDistribution(1, 4))
    n_environments: int = int_field(default=1, domain=IntDistribution(1, 4))
    nonlinearity: NonlinearityKind = eqx.field(static=True, default=NonlinearityKind.none)
    coefficient_seed: int = int_field(default=0, domain=IntDistribution(0, 1 << 16))
    coefficient_scale: float = float_field(default=1.0, domain=FloatDistribution(0.1, 4.0))
    endo_latent: int = int_field(
        default=8,
        domain=CategoricalDistribution(hardware_friendly_ints(1, 16)),
        optimize=True,
    )
    exo_latent: int = int_field(
        default=8,
        domain=CategoricalDistribution(hardware_friendly_ints(1, 16)),
        optimize=True,
    )
    n_frequencies: int = int_field(
        default=10,
        domain=CategoricalDistribution(hardware_friendly_ints(2, 16)),
        optimize=True,
    )

    @override
    def problem(self) -> IVProblem:
        return build_iv_problem(
            n_instruments=self.n_instruments,
            n_confounders=self.n_confounders,
            n_candidate_confounders=self.n_candidate_confounders,
            n_outcomes=self.n_outcomes,
            n_environments=self.n_environments,
            nonlinearity=self.nonlinearity,
            seed=self.coefficient_seed,
            scale=self.coefficient_scale,
        )

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        del data_source
        assert isinstance(problem, IVProblem)
        return AFPModel.create(
            endo_features=problem.obs_x_features,
            exo_features=problem.obs_x_features,
            obs_features=problem.obs_y_features,
            n_frequencies=self.n_frequencies,
            endo_latent=self.endo_latent,
            exo_latent=self.exo_latent,
            streams=streams,
        )
