from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.phasor.elementwise_rotation import FrequencyElementwiseRotation
from cem.phasor.evidence_pooling import (
    FrequencyAdaptedEvidencePooling,
    LowRankFrequencyAdaptedEvidencePooling,
)
from cem.phasor.gate import phasor_gate
from cem.phasor.message import JaxComplexArray
from cem.phasor.mobius_summation import LowRankMobiusSummation
from cem.structure.graph import FixedParameter, LearnableParameter
from cem.transforms.dropout import apply_dropout_if_training


class FrequencyGatedProjectionAblation(eqx.Module):
    """Static controls used to isolate gated-projection mechanisms."""

    learn_input_rotation: bool = eqx.field(static=True, default=True)
    gate_enabled: bool = eqx.field(static=True, default=True)


class FrequencyGatedProjection(eqx.Module):
    """Gated projection that preserves a coherent frequency axis.

    An elementwise rotation first calibrates each input without mixing distinct
    features. A low-rank Möbius bank constructs values independently at each
    frequency. A low-rank complex-linear projection constructs one admission gate
    per feature, shared across frequencies, before output pooling maps values to
    target fields.
    """

    input_rotation: FrequencyElementwiseRotation
    value: LowRankMobiusSummation
    admission: LowRankFrequencyAdaptedEvidencePooling
    gate_bias: LearnableParameter[JaxRealArray]
    output: FrequencyAdaptedEvidencePooling
    dropout_rate: FixedParameter[JaxRealArray]
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    ablation: FrequencyGatedProjectionAblation

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        frequencies: JaxRealArray,
        *,
        hidden_features: int,
        mobius_rank: int = 2,
        dropout_rate: float = 0.0,
        ablation: FrequencyGatedProjectionAblation | None = None,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            input_rotation=FrequencyElementwiseRotation.create(
                in_features, frequencies, streams=streams
            ),
            value=LowRankMobiusSummation.create(
                in_features, hidden_features, mobius_rank, streams=streams
            ),
            admission=LowRankFrequencyAdaptedEvidencePooling.create(
                in_features,
                hidden_features,
                mobius_rank,
                frequencies,
                streams=streams,
            ),
            gate_bias=LearnableParameter(jnp.zeros(hidden_features, dtype=jnp.float64)),
            output=FrequencyAdaptedEvidencePooling.create(
                hidden_features, out_features, frequencies, streams=streams
            ),
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
            in_features=in_features,
            hidden_features=hidden_features,
            ablation=ablation or FrequencyGatedProjectionAblation(),
        )

    def infer(
        self, z: JaxComplexArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxComplexArray:
        """Apply the coherent gated projection with optional dropout."""
        n_frequencies = self.input_rotation.frequencies.value.shape[0]
        rotated_flat = self.input_rotation.rotate(z) if self.ablation.learn_input_rotation else z
        rotated = rotated_flat.reshape(*z.shape[:-1], self.in_features, n_frequencies)
        per_frequency = jnp.swapaxes(rotated, -2, -1)
        value = self.value.sum(per_frequency)
        if self.ablation.gate_enabled:
            gate_evidence = self.admission.project(rotated_flat).reshape(
                *z.shape[:-1], self.hidden_features, n_frequencies
            )
            gate_signal = jnp.mean(gate_evidence, axis=-1)[..., jnp.newaxis, :]
            bias = jnp.reshape(
                self.gate_bias.value,
                (1,) * (gate_signal.ndim - 1) + (self.hidden_features,),
            )
            gated = phasor_gate(gate_signal + bias, value)
        else:
            gated = value
        grouped = jnp.swapaxes(gated, -2, -1).reshape(*z.shape[:-1], -1)
        result = self.output.project(grouped)
        return apply_dropout_if_training(
            result, streams=streams, inference=inference, dropout_rate=self.dropout_rate.value
        )
