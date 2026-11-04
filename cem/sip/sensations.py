"""SIP sensation coupling: bottom-up stack of sensation features.

See `sensation-coupling` in `~/src/typst/thesis.typ`.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import KeyArray, RngStream

from cem.sip.explanatory_coupling import ExplanatoryCoupling
from cem.sip.features import (
    SensationInputFeature,
    SIPFeature,
    SIPFeatureOutputs,
    SIPFeatureSizes,
)
from cem.sip.messages import InstrumentedMessage, concat_instrumented_messages


class Sensations(ExplanatoryCoupling):
    """Bottom-up explanatory coupling.

    Layer 0 is a `SensationInputFeature`: the environment observation flows
    in unchanged (no `observation_map`, no `Credit`). Each higher layer is a
    non-controllable `SIPFeature` whose observation content is the previous
    layer's `observation_mismatch` and whose Credit goal is the previous
    layer's `dissonance`. Each layer's effective predictor is the externally
    supplied predictor concatenated with the previous-step instrumented
    messages of the layers above (fetched through `last_message_index`), so
    higher sensations condition lower ones.
    """

    input_feature: SensationInputFeature
    interior_features: tuple[SIPFeature, ...]

    @classmethod
    def create(
        cls,
        sizes: Sequence[int],
        *,
        predictor_features: Sequence[int],
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build a bottom-up sensation stack from per-layer sizes.

        `sizes[0]` is both the environment observation dimension and the
        layer-0 feature width. `predictor_features[i]` is the external
        predictor dimension layer `i` consumes; its effective predictor
        dimension is `predictor_features[i] + sum(sizes[i + 1:])` to make
        room for the higher layers' previous messages.
        """
        if len(sizes) == 0:
            msg = "sizes must be non-empty"
            raise ValueError(msg)
        if len(predictor_features) != len(sizes):
            msg = "predictor_features must have the same length as sizes"
            raise ValueError(msg)
        effective_predictor_features = [
            predictor_features[i] + sum(sizes[i + 1 :]) for i in range(len(sizes))
        ]
        input_feature = SensationInputFeature.create(
            sizes[0],
            predictor_features=effective_predictor_features[0],
            streams=streams,
        )
        interior: list[SIPFeature] = []
        for i in range(1, len(sizes)):
            feature_sizes = SIPFeatureSizes(
                observation_content=2 * sizes[i - 1],
                predictor=effective_predictor_features[i],
                gain=0,
            )
            interior.append(
                SIPFeature.create(sizes[i], feature_sizes, controllable=False, streams=streams)
            )
        return cls(input_feature=input_feature, interior_features=tuple(interior))

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

        Layer 0 consumes `observation` directly; each higher layer's content
        is the previous layer's `observation_mismatch`, and its Credit goal
        is the previous layer's `dissonance`. Each layer's predictor is the
        external `predictors[i]` concatenated with the stored
        `last_message_index` of every layer above it (previous-step values,
        since higher layers have not yet been processed this step).
        """
        layer_count = 1 + len(self.interior_features)
        if len(predictors) != layer_count:
            msg = "predictors must have one entry per layer"
            raise ValueError(msg)
        state, input_out = self.input_feature.step(
            state,
            observation=observation,
            predictors=self._predictor_for(state, 0, predictors[0]),
            gain=None,
            streams=streams,
            inference=inference,
        )
        outputs: list[SIPFeatureOutputs] = [input_out]
        if not self.interior_features:
            return state, tuple(outputs)
        keys = jr.split(key, len(self.interior_features))
        for i, (feature, feature_key, feature_predictors) in enumerate(
            zip(self.interior_features, keys, predictors[1:], strict=True), start=1
        ):
            prev = outputs[-1]
            state, out = feature.step(
                state,
                feature_key,
                observation_content=prev.observation_mismatch,
                predictors=self._predictor_for(state, i, feature_predictors),
                gain=None,
                goal=jnp.asarray(prev.dissonance),
                streams=streams,
                inference=inference,
            )
            outputs.append(out)
        return state, tuple(outputs)

    def _predictor_for(
        self,
        state: eqx.nn.State,
        layer: int,
        external: InstrumentedMessage,
    ) -> InstrumentedMessage:
        """Concatenate `external` with the stored messages of every layer above `layer`."""
        higher = [
            state.get(self.interior_features[j - 1].last_message_index)
            for j in range(layer + 1, 1 + len(self.interior_features))
        ]
        if not higher:
            return external
        return concat_instrumented_messages(external, *higher)
