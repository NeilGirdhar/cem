"""SIP sensation coupling: bottom-up stack of sensation features."""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.sip.explanatory_coupling import ExplanatoryCoupling
from cem.sip.features import (
    SensationFeature,
    SensationInputFeature,
    SIPFeatureOutputs,
    SIPFeatureSizes,
)
from cem.sip.messages import InstrumentedMessage, concat_instrumented_messages
from cem.sip.settings import MismatchSettings


class Sensations(ExplanatoryCoupling):
    """Build increasingly abstract explanations of the observation in a bottom-up stack.

    See `sensation-coupling` in the thesis.
    """

    input_feature: SensationInputFeature
    interior_features: tuple[SensationFeature, ...]
    # Each interior layer's reconstruction loss and confounding objective from the previous step,
    # shape (interior layers, 2).
    previous_dissonance_index: eqx.nn.StateIndex[JaxRealArray]

    @classmethod
    def create(
        cls,
        sizes: Sequence[int],
        *,
        predictor_features: Sequence[int],
        mismatch_settings: MismatchSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build a bottom-up sensation stack from per-layer sizes.

        `sizes[0]` is both the environment observation dimension and the
        layer-0 feature width. `predictor_features[i]` is the external
        predictor dimension layer `i` consumes; its effective predictor
        dimension is `predictor_features[i] + sizes[i + 1]` to make room
        for the one-layer-above previous message.
        """
        if len(sizes) == 0:
            msg = "sizes must be non-empty"
            raise ValueError(msg)
        if len(predictor_features) != len(sizes):
            msg = "predictor_features must have the same length as sizes"
            raise ValueError(msg)
        effective_predictor_features = [
            predictor_features[i] + (sizes[i + 1] if i + 1 < len(sizes) else 0)
            for i in range(len(sizes))
        ]
        input_feature = SensationInputFeature.create(
            sizes[0],
            predictor_features=effective_predictor_features[0],
            mismatch_settings=mismatch_settings,
            streams=streams,
        )
        interior: list[SensationFeature] = []
        for i in range(1, len(sizes)):
            feature_sizes = SIPFeatureSizes(
                observation_content=2 * sizes[i - 1],
                predictor=effective_predictor_features[i],
                gain=0,
            )
            interior.append(
                SensationFeature.create(
                    sizes[i],
                    feature_sizes,
                    controllable=False,
                    mismatch_settings=mismatch_settings,
                    streams=streams,
                )
            )
        return cls(
            input_feature=input_feature,
            interior_features=tuple(interior),
            previous_dissonance_index=eqx.nn.StateIndex(jnp.zeros((len(interior), 2))),
        )

    def infer(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation: NormalNP,
        predictors: Sequence[InstrumentedMessage],
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, tuple[SIPFeatureOutputs, ...]]:
        """Run bottom-up inference across the stack.

        Layer 0 consumes `observation` directly; each higher layer's content is the previous
        layer's `observation_mismatch`. Each layer's predictor is the external `predictors[i]`
        concatenated with the stored `last_message_index` of the layer directly above it (its
        previous-step value, since higher layers have not yet been processed this step).

        An interior layer's goal is its negated dissonance and that of the layer below. The layer
        below scores predictions that used this layer's previous emission, so its dissonance from
        this step credits that emission. The layer's own dissonance scores its emission on the step
        it is emitted, so the stack holds it for one step, and both arrive with the goal that
        credits the same emission. See `sensation-coupling` and `sip-eligibility-traces` in the
        thesis.
        """
        layer_count = 1 + len(self.interior_features)
        if len(predictors) != layer_count:
            msg = "predictors must have one entry per layer"
            raise ValueError(msg)
        state, input_out = self.input_feature.step(
            state,
            observation=observation,
            predictor=self._predictor_for(state, 0, predictors[0]),
            gain=None,
            streams=streams,
            inference=inference,
        )
        outputs: list[SIPFeatureOutputs] = [input_out]
        if not self.interior_features:
            return state, tuple(outputs)
        previous_dissonance = state.get(self.previous_dissonance_index)
        keys = jr.split(key, len(self.interior_features))
        for i, (feature, feature_key, feature_predictor) in enumerate(
            zip(self.interior_features, keys, predictors[1:], strict=True), start=1
        ):
            below = outputs[-1]
            own_reconstruction, own_confounding = previous_dissonance[i - 1]
            state, out = feature.step(
                state,
                feature_key,
                observation_content=below.observation_mismatch,
                predictor=self._predictor_for(state, i, feature_predictor),
                gain=None,
                reconstruction=below.reconstruction_loss + own_reconstruction,
                confounding=below.confounding_objective + own_confounding,
                streams=streams,
                inference=inference,
            )
            outputs.append(out)
        own_dissonance = jnp.stack(
            [jnp.stack([out.reconstruction_loss, out.confounding_objective]) for out in outputs[1:]]
        )
        state = state.set(self.previous_dissonance_index, jax.lax.stop_gradient(own_dissonance))
        return state, tuple(outputs)

    def _predictor_for(
        self,
        state: eqx.nn.State,
        layer: int,
        external: InstrumentedMessage,
    ) -> InstrumentedMessage:
        """Add the upper layer's previous message to the external predictor when available.

        Bottom-up inference reads the upper layer's state before advancing it.
        """
        above_layer = layer + 1
        if above_layer >= 1 + len(self.interior_features):
            return external
        above = state.get(self.interior_features[above_layer - 1].last_message_index)
        return concat_instrumented_messages(external, above)
