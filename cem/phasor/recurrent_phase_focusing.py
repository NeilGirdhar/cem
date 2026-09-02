from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.perceptron.mlp import MLP
from cem.phasor.gated_projection import GatedProjection
from cem.phasor.message import JaxComplexArray, phasor_to_real
from cem.phasor.mobius_summation import phase_warp
from cem.structure.graph import FixedParameter
from cem.transforms.dropout import apply_dropout_if_training


class RecurrentPhaseFocusing(eqx.Module):
    """Refine a prediction through example-dependent input views.

    Each iteration applies the current rotation and phase warp to one phasor
    per input feature. A gated projection predicts an output correction from the
    focused view. A real-valued controller sees that view, the current focus
    parameters, and the current prediction, then updates the rotation and weight
    for the next iteration. Both networks share their weights across iterations.
    """

    predictor: GatedProjection
    controller: MLP
    dropout_rate: FixedParameter[JaxRealArray]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    iterations: int = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int,
        iterations: int = 3,
        dropout_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if iterations < 1:
            msg = f"iterations must be positive, got {iterations}"
            raise ValueError(msg)
        controller_inputs = 4 * in_features + 2 * out_features
        controller_outputs = 2 * in_features
        return cls(
            predictor=GatedProjection.create(
                in_features,
                out_features,
                mid_features=hidden_features,
                streams=streams,
            ),
            controller=MLP.create(
                controller_inputs,
                controller_outputs,
                hidden_features=hidden_features,
                streams=streams,
            ),
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
            in_features=in_features,
            out_features=out_features,
            iterations=iterations,
        )

    def focus(
        self,
        z: JaxComplexArray,
        rotations: JaxRealArray,
        weights: JaxRealArray,
    ) -> JaxComplexArray:
        """Apply one example's rotations and signed phase-warp weights."""
        if z.shape[-1] != self.in_features:
            msg = f"expected final input dimension {self.in_features}, got {z.shape[-1]}"
            raise ValueError(msg)
        presence = jnp.abs(z)
        unit = z / jnp.where(presence > 0, presence, 1)
        centred = unit * jnp.exp(-1j * rotations)
        return presence * phase_warp(centred, weights)

    def infer(
        self,
        z: JaxComplexArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxComplexArray:
        """Run the shared predictor and focus controller for a fixed number of steps."""
        if z.shape[-1] != self.in_features:
            msg = f"expected final input dimension {self.in_features}, got {z.shape[-1]}"
            raise ValueError(msg)
        state_shape = (*z.shape[:-1], self.in_features)
        output_shape = (*z.shape[:-1], self.out_features)
        rotations = jnp.zeros(state_shape, dtype=jnp.real(z).dtype)
        weights = jnp.ones(state_shape, dtype=jnp.real(z).dtype)
        output = jnp.zeros(output_shape, dtype=z.dtype)

        for _ in range(self.iterations):
            focused = self.focus(z, rotations, weights)
            output_correction = self.predictor.infer(
                focused,
                streams=streams,
                inference=inference,
            )
            controller_input = jnp.concatenate(
                (
                    phasor_to_real(focused),
                    rotations,
                    weights,
                    phasor_to_real(output),
                ),
                axis=-1,
            )
            focus_correction = self.controller.infer(
                controller_input,
                streams=streams,
                inference=inference,
            )
            rotation_correction, weight_correction = jnp.split(
                focus_correction,
                2,
                axis=-1,
            )
            output += output_correction
            rotations += rotation_correction
            weights *= 1 + weight_correction

        return apply_dropout_if_training(
            output,
            streams=streams,
            inference=inference,
            dropout_rate=self.dropout_rate.value,
        )
