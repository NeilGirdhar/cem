from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.phasor.elementwise_rotation import ElementwiseRotation
from cem.phasor.evidence_pooling import EvidencePooling
from cem.phasor.message import JaxComplexArray
from cem.phasor.mobius_summation import MobiusSummation
from cem.phasor.phase_activation import PhaseActivation
from cem.structure.graph import FixedParameter
from cem.transforms.dropout import apply_dropout_if_training


class PhaseActivatedProjection(eqx.Module):
    """Möbius projection with signal-derived phase activation.

    An elementwise rotation calibrates each input, Möbius summation constructs
    hidden values, and phase activation attenuates candidates according to their
    phases.  Output pooling then combines the activated candidates.  Unlike
    :class:`GatedProjection`, this projection does not construct a separate
    admission signal.

    Attributes:
        input_rotation: Learned phase rotation for each input feature.
        value: Möbius summation from input to hidden features.
        activation: Learned phase-only activation for each hidden feature.
        output: Evidence pooling from hidden to output features.
        dropout_rate: Fraction of outputs zeroed after pooling.
    """

    input_rotation: ElementwiseRotation
    value: MobiusSummation
    activation: PhaseActivation
    output: EvidencePooling
    dropout_rate: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        mid_features: int | None = None,
        dropout_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if mid_features is None:
            mid_features = out_features
        return cls(
            input_rotation=ElementwiseRotation.create(in_features, streams=streams),
            value=MobiusSummation.create(in_features, mid_features, streams=streams),
            activation=PhaseActivation.create(mid_features, streams=streams),
            output=EvidencePooling.create(mid_features, out_features, streams=streams),
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
        )

    def infer(
        self, z: JaxComplexArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxComplexArray:
        """Construct, activate, and pool Möbius candidates."""
        rotated = self.input_rotation.rotate(z)
        value = self.value.sum(rotated)
        activated = self.activation.activate(value)
        result = self.output.project(activated)
        return apply_dropout_if_training(
            result, streams=streams, inference=inference, dropout_rate=self.dropout_rate.value
        )
