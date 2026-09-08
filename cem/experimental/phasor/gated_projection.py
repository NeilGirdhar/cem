from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.experimental.phasor.elementwise_rotation import ElementwiseRotation
from cem.experimental.phasor.evidence_pooling import EvidencePooling
from cem.experimental.phasor.gate import phasor_gate
from cem.experimental.phasor.message import JaxComplexArray
from cem.experimental.phasor.mobius_summation import (
    MobiusPresenceRule,
    MobiusSummation,
    MobiusSummationDiagnostics,
)
from cem.structure.graph import FixedParameter, LearnableParameter
from cem.transforms.dropout import apply_dropout_if_training


class GatedProjection(eqx.Module):
    """Gated nonlinear map over phasor messages, with optional dropout.

    An elementwise rotation first calibrates each input without mixing distinct features.
    Möbius summation constructs hidden values, while a complex-linear projection constructs
    their admission evidence. The phasor gate suppresses the values, then output pooling
    combines admitted candidates.

    The rotation and linear links have no additive bias. ``gate_bias`` is a real bias
    that sets the admission gate's default logit. Absent input therefore produces
    absent output even though the gate has a bias. Pass ``inference=True`` to
    :meth:`infer` to skip dropout at eval time.

    Attributes:
        input_rotation: Learned phase rotation, one displacement per input feature.
        value: Value Möbius summation, in_features → mid_features.
        admission: Complex-linear admission projection, in_features → mid_features.
        gate_bias: Real bias added to the gate signal before gating, shape (mid_features,).
        output: Output pooling link, mid_features → out_features.
        dropout_rate: Fraction of outputs zeroed after ``output``.  0.0 disables.
    """

    input_rotation: ElementwiseRotation
    value: MobiusSummation
    admission: EvidencePooling
    gate_bias: LearnableParameter[JaxRealArray]
    output: EvidencePooling
    dropout_rate: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        mid_features: int | None = None,
        mobius_presence_rule: MobiusPresenceRule = MobiusPresenceRule.parallel,
        dropout_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if mid_features is None:
            mid_features = out_features
        return cls(
            input_rotation=ElementwiseRotation.create(in_features, streams=streams),
            value=MobiusSummation.create(
                in_features,
                mid_features,
                presence_rule=mobius_presence_rule,
                streams=streams,
            ),
            admission=EvidencePooling.create(in_features, mid_features, streams=streams),
            gate_bias=LearnableParameter(jnp.zeros(mid_features, dtype=jnp.float64)),
            output=EvidencePooling.create(mid_features, out_features, streams=streams),
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
        )

    def infer(
        self, z: JaxComplexArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxComplexArray:
        """Apply the GLU-style nonlinear transform with optional dropout.

        Args:
            z: Input phasors, shape (..., in_features).
            streams: RNG streams; the ``"inference"`` stream is used for dropout.
            inference: When ``True``, dropout is skipped.

        Returns:
            Output phasors, shape (..., out_features).
        """
        result, _ = self.infer_with_diagnostics(z, streams=streams, inference=inference)
        return result

    def infer_with_diagnostics(
        self, z: JaxComplexArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> tuple[JaxComplexArray, MobiusSummationDiagnostics]:
        """Apply the projection and return its candidate-presence diagnostics."""
        rotated = self.input_rotation.rotate(z)
        value, diagnostics = self.value.sum_with_diagnostics(rotated)
        gate_signal = self.admission.project(rotated)
        bias = jnp.reshape(self.gate_bias.value, (1,) * (gate_signal.ndim - 1) + (-1,))
        gated = phasor_gate(gate_signal + bias, value)
        result = self.output.project(gated)
        result = apply_dropout_if_training(
            result,
            streams=streams,
            inference=inference,
            dropout_rate=self.dropout_rate.value,
        )
        return result, diagnostics
