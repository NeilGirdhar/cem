from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.phasor.evidence_pooling import EvidencePooling
from cem.phasor.gate import phasor_gate
from cem.phasor.message import JaxComplexArray
from cem.phasor.mobius_summation import MobiusSummation
from cem.structure.graph import FixedParameter, LearnableParameter
from cem.transforms.dropout import apply_dropout_if_training


class GatedProjection(eqx.Module):
    """Gated nonlinear map over phasor messages, with optional dropout.

    Input pooling first collects equivalent sensors. Separate Möbius summation banks
    construct hidden values and their admission signals from the pooled evidence. The
    phasor gate suppresses the values, then output pooling combines admitted candidates
    into the requested output features.

    Both pooling links are bias-free. ``gate_bias`` is a real bias that sets the
    admission gate's default logit. Absent input therefore produces absent output even
    though the gate has a bias. Pass ``inference=True`` to :meth:`infer` to skip
    dropout at eval time.

    Attributes:
        input: Input evidence pooling, in_features → mid_features.
        value: Value Möbius summation, mid_features → mid_features.
        admission: Admission Möbius summation, mid_features → mid_features.
        gate_bias: Real bias added to the gate signal before gating, shape (mid_features,).
        output: Output pooling link, mid_features → out_features.
        dropout_rate: Fraction of outputs zeroed after ``output``.  0.0 disables.
    """

    input: EvidencePooling
    value: MobiusSummation
    admission: MobiusSummation
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
        dropout_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if mid_features is None:
            mid_features = out_features
        return cls(
            input=EvidencePooling.create(in_features, mid_features, streams=streams),
            value=MobiusSummation.create(mid_features, mid_features, streams=streams),
            admission=MobiusSummation.create(mid_features, mid_features, streams=streams),
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
        pooled = self.input.project(z)
        value = self.value.sum(pooled)
        gate_signal = self.admission.sum(pooled)
        bias = jnp.reshape(self.gate_bias.value, (1,) * (gate_signal.ndim - 1) + (-1,))
        gated = phasor_gate(gate_signal + bias, value)
        result = self.output.project(gated)
        return apply_dropout_if_training(
            result, streams=streams, inference=inference, dropout_rate=self.dropout_rate.value
        )
