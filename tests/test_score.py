from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from efax import Flattener, NormalNP

from cem.phasor_calculus.loss import reconstruction_loss
from cem.phasor_calculus.message import PhasorMessage, geometric_frequencies
from cem.phasor_calculus.score import (
    ObservedScoreNode,
    ObservedScoreOutput,
    ScoreOutput,
    latent_score,
)

# Use small frequencies for accurate OLS recovery in ObservedScoreNode tests.
_M = 8
_BASE = 1e-4


@pytest.fixture
def freqs() -> jnp.ndarray:
    return geometric_frequencies(_M, base=_BASE)


@pytest.fixture
def observed_node(freqs: jnp.ndarray) -> ObservedScoreNode:
    flattener, _ = Flattener.flatten(
        NormalNP(jnp.array(0.0), jnp.array(0.0)), mapped_to_plane=False
    )
    return ObservedScoreNode.create(flattener, freqs)


# ── latent_score ──────────────────────────────────────────────────────────────


def test_latent_score_returns_score_output() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0 + 1j]))
    assert isinstance(latent_score(z, z), ScoreOutput)


def test_latent_score_score_is_phasor_message() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0.5 - 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.5 + 0.5j, 1 + 0j]))
    assert isinstance(latent_score(z, z_hat).score, PhasorMessage)


def test_latent_score_score_shape() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0.5 - 0.5j, -1 + 0j]))
    z_hat = PhasorMessage(jnp.array([0.5 + 0.5j, 1 + 0j, 0 + 1j]))
    out = latent_score(z, z_hat)
    assert out.score.data.shape == z_hat.data.shape


def test_latent_score_loss_shape() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0.5 - 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.5 + 0.5j, 1 + 0j]))
    out = latent_score(z, z_hat)
    assert out.loss.shape == z_hat.data.shape


def test_latent_score_loss_matches_reconstruction_loss() -> None:
    observed = PhasorMessage(jnp.array([1 + 0j, 0.5 + 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.8 + 0.2j, 0.3 - 0.3j]))
    out = latent_score(observed, z_hat)
    assert jnp.allclose(out.loss, reconstruction_loss(observed.data, z_hat.data))


def test_latent_score_total_loss_is_scalar() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0 + 1j, 0.5 + 0.5j]))
    assert latent_score(z, z).total_loss().shape == ()


def test_latent_score_total_loss_is_sum_of_loss() -> None:
    observed = PhasorMessage(jnp.array([1 + 0j, 0.5 + 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.8 + 0.2j, 0.3 - 0.3j]))
    out = latent_score(observed, z_hat)
    assert jnp.allclose(out.total_loss(), jnp.sum(out.loss))


def test_latent_score_self_score_is_zero() -> None:
    # At the minimum z_hat = observed, the gradient of the loss is zero.
    z = PhasorMessage(jnp.array([1 + 0j, 0 + 1j, 0.5 - 0.5j]))
    assert jnp.allclose(latent_score(z, z).score.data, 0.0, atol=1e-6)


def test_latent_score_score_equals_gradient() -> None:
    # score.data must equal jax.grad of the summed reconstruction loss w.r.t. z_hat.
    observed = PhasorMessage(jnp.array([1 + 0j, 0.5 + 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.8 + 0.2j, 0.3 - 0.3j]))
    out = latent_score(observed, z_hat)
    expected = jax.grad(lambda z: jnp.sum(reconstruction_loss(observed.data, z.data)))(z_hat)
    assert jnp.allclose(out.score.data, expected.data)


def test_latent_score_batched_shapes() -> None:
    observed = PhasorMessage(jnp.ones((3, 4), dtype=jnp.complex128))
    z_hat = PhasorMessage(jnp.ones((3, 4), dtype=jnp.complex128) * (0.5 + 0.5j))
    out = latent_score(observed, z_hat)
    assert out.score.data.shape == (3, 4)
    assert out.loss.shape == (3, 4)
    assert out.total_loss().shape == ()


# ── ObservedScoreNode ─────────────────────────────────────────────────────────


def test_observed_score_node_returns_observed_score_output(
    observed_node: ObservedScoreNode, freqs: jnp.ndarray
) -> None:
    dist = NormalNP(jnp.array(0.5), jnp.array(-0.5))
    z_hat = PhasorMessage.from_distribution(dist, freqs)
    assert isinstance(observed_node.infer(dist.to_exp(), z_hat), ObservedScoreOutput)


def test_observed_score_node_score_is_phasor_message(
    observed_node: ObservedScoreNode, freqs: jnp.ndarray
) -> None:
    dist = NormalNP(jnp.array(0.5), jnp.array(-0.5))
    z_hat = PhasorMessage.from_distribution(dist, freqs)
    out = observed_node.infer(dist.to_exp(), z_hat)
    assert isinstance(out.score, PhasorMessage)
    assert out.score.data.shape == z_hat.data.shape


def test_observed_score_node_total_loss_is_scalar(
    observed_node: ObservedScoreNode, freqs: jnp.ndarray
) -> None:
    dist = NormalNP(jnp.array(0.5), jnp.array(-0.5))
    z_hat = PhasorMessage.from_distribution(dist, freqs)
    assert observed_node.infer(dist.to_exp(), z_hat).total_loss().shape == ()


def test_observed_score_node_total_loss_is_sum_of_loss(
    observed_node: ObservedScoreNode, freqs: jnp.ndarray
) -> None:
    dist = NormalNP(jnp.array(0.5), jnp.array(-0.5))
    z_hat = PhasorMessage.from_distribution(dist, freqs)
    out = observed_node.infer(dist.to_exp(), z_hat)
    assert jnp.allclose(out.total_loss(), jnp.sum(out.loss))


def test_observed_score_node_self_score_near_zero(
    observed_node: ObservedScoreNode, freqs: jnp.ndarray
) -> None:
    # Encoding the observed distribution and predicting with the same phasors gives score ≈ 0.
    # OLS recovery is approximate, so we allow some tolerance.
    dist = NormalNP(jnp.array(0.5), jnp.array(-0.5))
    z_hat = PhasorMessage.from_distribution(dist, freqs)
    out = observed_node.infer(dist.to_exp(), z_hat)
    assert jnp.allclose(jnp.abs(out.score.data), 0.0, atol=1e-2)


def test_observed_score_node_predicted_exp_recovers_mean(
    observed_node: ObservedScoreNode, freqs: jnp.ndarray
) -> None:
    mu, sigma2 = 0.5, 1.0
    dist = NormalNP(jnp.array(mu / sigma2), jnp.array(-0.5 / sigma2))
    z_hat = PhasorMessage.from_distribution(dist, freqs)
    out = observed_node.infer(dist.to_exp(), z_hat)
    assert jnp.allclose(out.predicted_exp.mean, jnp.array(mu), atol=1e-2)


def test_observed_score_node_score_equals_gradient(
    observed_node: ObservedScoreNode, freqs: jnp.ndarray
) -> None:
    dist = NormalNP(jnp.array(0.5), jnp.array(-0.5))
    observed_exp = dist.to_exp()
    # Perturb z_hat slightly so the score is nonzero.
    z_hat = PhasorMessage(PhasorMessage.from_distribution(dist, freqs).data * 1.1)

    def loss_fn(z: PhasorMessage) -> jnp.ndarray:
        return jnp.sum(observed_exp.cross_entropy(z.to_distribution(observed_node.t).to_nat()))

    out = observed_node.infer(observed_exp, z_hat)
    expected = jax.grad(loss_fn)(z_hat)
    assert jnp.allclose(out.score.data, expected.data, atol=1e-6)
