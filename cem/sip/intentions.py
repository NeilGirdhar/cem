"""SIP intention coupling: top-down stack of intention features.

See `intention-coupling` in `~/src/typst/thesis.typ`.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.sip.explanatory_coupling import ExplanatoryCoupling
from cem.sip.features import SIPFeature, SIPFeatureOutputs, SIPFeatureSizes
from cem.sip.messages import InstrumentedMessage, concat_messages


class Intentions(ExplanatoryCoupling):
    """Top-down explanatory coupling.

    Processed top-down: the top layer takes its observation content from its
    readout alone; every lower layer concatenates its readout with the
    `observation_mismatch` demand signal from the layer above. The goal
    signal fed to each feature's Credit is the per-layer advantage. All
    features are controllable `SIPFeature`s that inject local noise to
    explore. Layer 0 is the bottom (processed last); its outgoing
    observation is the action.
    """

    features: tuple[SIPFeature, ...]

    @classmethod
    def create(
        cls,
        sizes: Sequence[int],
        *,
        predictor_features: Sequence[int],
        readout_features: Sequence[int],
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build a top-down intention stack from per-layer sizes.

        `sizes[0]` is the width of the bottom (action) layer and `sizes[-1]`
        is the width of the top. `readout_features[i]` and
        `predictor_features[i]` are the externally-supplied readout and
        predictor dimensions for layer `i`.
        """
        if len(sizes) == 0:
            msg = "sizes must be non-empty"
            raise ValueError(msg)
        if len(predictor_features) != len(sizes):
            msg = "predictor_features must have the same length as sizes"
            raise ValueError(msg)
        if len(readout_features) != len(sizes):
            msg = "readout_features must have the same length as sizes"
            raise ValueError(msg)
        features: list[SIPFeature] = []
        top_index = len(sizes) - 1
        for i, n in enumerate(sizes):
            content_features = readout_features[i]
            if i < top_index:
                content_features += 2 * sizes[i + 1]
            feature_sizes = SIPFeatureSizes(
                observation_content=content_features,
                predictor=predictor_features[i],
                gain=0,
            )
            features.append(SIPFeature.create(n, feature_sizes, controllable=True, streams=streams))
        return cls(features=tuple(features))

    def infer(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        readouts: Sequence[NormalNP],
        predictors: Sequence[InstrumentedMessage],
        advantages: Sequence[JaxRealArray],
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, tuple[SIPFeatureOutputs, ...]]:
        """Run top-down inference across the stack.

        The top layer's observation content is its readout; each lower
        layer's content is its readout concatenated with the
        `observation_mismatch` of the layer above. `advantages[i]` is the
        Credit goal for layer `i`. Returns per-layer outputs indexed with
        the same convention as `features` (`outputs[0]` is the bottom, whose
        `observation` is the action).
        """
        n_layers = len(self.features)
        if len(readouts) != n_layers:
            msg = "readouts must have one entry per layer"
            raise ValueError(msg)
        if len(predictors) != n_layers:
            msg = "predictors must have one entry per layer"
            raise ValueError(msg)
        if len(advantages) != n_layers:
            msg = "advantages must have one entry per layer"
            raise ValueError(msg)
        keys = jr.split(key, n_layers)
        outputs: list[SIPFeatureOutputs | None] = [None] * n_layers
        top_index = n_layers - 1
        state, outputs[top_index] = self.features[top_index].step(
            state,
            keys[top_index],
            observation_content=readouts[top_index],
            predictors=predictors[top_index],
            gain=None,
            goal=advantages[top_index],
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
                predictors=predictors[i],
                gain=None,
                goal=advantages[i],
                streams=streams,
                inference=inference,
            )
        return state, tuple(out for out in outputs if out is not None)
