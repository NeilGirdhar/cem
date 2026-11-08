"""SIP intention coupling: top-down stack of intention features."""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.sip.explanatory_coupling import ExplanatoryCoupling
from cem.sip.features import IntentionFeature, SIPFeatureOutputs, SIPFeatureSizes
from cem.sip.messages import InstrumentedMessage, concat_messages
from cem.sip.parameters import advantage_parameter
from cem.sip.settings import CreditSettings, MismatchSettings


class Intentions(ExplanatoryCoupling):
    """Turn abstract intentions into concrete actions through a top-down feature stack.

    See `intention-coupling` in the thesis.
    """

    features: tuple[IntentionFeature, ...]

    @classmethod
    def create(
        cls,
        sizes: Sequence[int],
        *,
        readout_features: Sequence[int],
        credit_settings: CreditSettings,
        mismatch_settings: MismatchSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build a top-down intention stack from per-layer sizes.

        List sizes and readout widths from the bottom action layer to the top.
        Each layer's predictor width is the width of the layer directly below it.
        The action layer has zero predictor inputs.
        """
        if len(sizes) == 0:
            msg = "sizes must be non-empty"
            raise ValueError(msg)
        if len(readout_features) != len(sizes):
            msg = "readout_features must have the same length as sizes"
            raise ValueError(msg)
        features: list[IntentionFeature] = []
        top_index = len(sizes) - 1
        for i, n in enumerate(sizes):
            content_features = readout_features[i]
            if i < top_index:
                content_features += 2 * sizes[i + 1]
            feature_sizes = SIPFeatureSizes(
                observation_content=content_features,
                predictor=sizes[i - 1] if i > 0 else 0,
                gain=0,
            )
            features.append(
                IntentionFeature.create(
                    n,
                    feature_sizes,
                    controllable=True,
                    credit_settings=credit_settings,
                    mismatch_settings=mismatch_settings,
                    streams=streams,
                    observation_parameter_factory=advantage_parameter,
                )
            )
        return cls(features=tuple(features))

    def infer(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        readouts: Sequence[NormalNP],
        reward: JaxRealArray,
        curiosity: JaxRealArray,
        tracking: JaxRealArray,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, tuple[SIPFeatureOutputs, ...]]:
        """Run top-down inference across the stack.

        Each layer receives its current readout and any mismatch from the layer above.
        The previous message from the layer below predicts its observation.
        Supply the scalar advantage for each reward source. Every layer receives all three.
        Outputs are ordered bottom first, with the action in the bottom layer's observation.
        """
        n_layers = len(self.features)
        if len(readouts) != n_layers:
            msg = "readouts must have one entry per layer"
            raise ValueError(msg)
        keys = jr.split(key, n_layers)
        outputs: list[SIPFeatureOutputs | None] = [None] * n_layers
        top_index = n_layers - 1
        state, outputs[top_index] = self.features[top_index].step(
            state,
            keys[top_index],
            observation_content=readouts[top_index],
            predictor=self._predictor_for(state, top_index),
            gain=None,
            reward=reward,
            curiosity=curiosity,
            tracking=tracking,
            streams=streams,
            inference=inference,
        )
        for i in range(top_index - 1, -1, -1):
            above = outputs[i + 1]
            assert above is not None
            content = concat_messages(readouts[i], above.observation_mismatch)
            state, outputs[i] = self.features[i].step(
                state,
                keys[i],
                observation_content=content,
                predictor=self._predictor_for(state, i),
                gain=None,
                reward=reward,
                curiosity=curiosity,
                tracking=tracking,
                streams=streams,
                inference=inference,
            )
        return state, tuple(out for out in outputs if out is not None)

    def _predictor_for(
        self,
        state: eqx.nn.State,
        layer: int,
    ) -> InstrumentedMessage:
        """Use the message from the layer directly below as the predictor.

        Top-down inference reads that layer's state before advancing it.
        The bottom layer uses an empty message.
        """
        if layer == 0:
            return InstrumentedMessage.initial(0)
        return state.get(self.features[layer - 1].last_message_index)
