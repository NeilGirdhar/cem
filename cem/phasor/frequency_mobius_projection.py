from collections.abc import Mapping
from enum import Enum
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.phasor.elementwise_rotation import FrequencyElementwiseRotation
from cem.phasor.evidence_pooling import FrequencyAdaptedEvidencePooling
from cem.phasor.message import JaxComplexArray
from cem.phasor.mobius_summation import (
    LowRankMobiusSummation,
    MobiusSummation,
)
from cem.phasor.phase_activation import FrequencyPhaseActivation
from cem.structure.graph import FixedParameter
from cem.transforms.dropout import apply_dropout_if_training


class MobiusParameterization(Enum):
    """Parameterization of a frequency Möbius candidate bank."""

    low_rank = "low_rank"
    dense = "dense"
    phase_activated_dense = "phase_activated_dense"


class FrequencyMobiusProjection(eqx.Module):
    """Gate-free feature projection over a coherent frequency bank.

    An elementwise rotation calibrates each input without mixing fields. A
    Möbius bank constructs candidate values independently at each frequency,
    then frequency-adapted output pooling maps those candidates to target fields.
    """

    input_rotation: FrequencyElementwiseRotation
    value: LowRankMobiusSummation | MobiusSummation
    activation: FrequencyPhaseActivation | None
    output: FrequencyAdaptedEvidencePooling
    dropout_rate: FixedParameter[JaxRealArray]
    in_features: int = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        frequencies: JaxRealArray,
        *,
        hidden_features: int,
        mobius_rank: int = 2,
        parameterization: MobiusParameterization = MobiusParameterization.low_rank,
        dropout_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if parameterization in {
            MobiusParameterization.dense,
            MobiusParameterization.phase_activated_dense,
        }:
            value = MobiusSummation.create(
                in_features,
                hidden_features,
                streams=streams,
            )
        else:
            value = LowRankMobiusSummation.create(
                in_features,
                hidden_features,
                mobius_rank,
                streams=streams,
            )
        return cls(
            input_rotation=FrequencyElementwiseRotation.create(
                in_features, frequencies, streams=streams
            ),
            value=value,
            activation=(
                FrequencyPhaseActivation.create(hidden_features, frequencies, streams=streams)
                if parameterization == MobiusParameterization.phase_activated_dense
                else None
            ),
            output=FrequencyAdaptedEvidencePooling.create(
                hidden_features, out_features, frequencies, streams=streams
            ),
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
            in_features=in_features,
        )

    def infer(
        self, z: JaxComplexArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxComplexArray:
        """Construct and pool Möbius candidates with optional dropout."""
        n_frequencies = self.input_rotation.frequencies.value.shape[0]
        rotated = self.input_rotation.rotate(z).reshape(
            *z.shape[:-1], self.in_features, n_frequencies
        )
        per_frequency = jnp.swapaxes(rotated, -2, -1)
        value = self.value.sum(per_frequency)
        grouped = jnp.swapaxes(value, -2, -1).reshape(*z.shape[:-1], -1)
        if self.activation is not None:
            grouped = self.activation.activate(grouped)
        result = self.output.project(grouped)
        return apply_dropout_if_training(
            result, streams=streams, inference=inference, dropout_rate=self.dropout_rate.value
        )
