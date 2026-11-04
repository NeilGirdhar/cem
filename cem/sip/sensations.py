"""SIP sensation coupling: bottom-up stack of sensation features.

See `sensation-coupling` in `~/src/typst/thesis.typ`.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.sip.explanatory_coupling import ExplanatoryCoupling
from cem.sip.features import FeaturePredictors, SIPFeature, SIPFeatureOutputs, SIPFeatureSizes


class Sensations(ExplanatoryCoupling):
    """Bottom-up explanatory coupling.

    Layer 0 receives the environment observation as its `observation_content`.
    Each higher layer's `observation_content` is the previous layer's
    `observation_mismatch`, a `2 * sizes[i - 1]`-dimensional demand signal.
    Features are non-controllable: sensations are observations of the
    environment, so no local noise is injected.
    """

    @classmethod
    def create(
        cls,
        sizes: Sequence[int],
        *,
        observation_features: int,
        predictor_features: Sequence[int],
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build a bottom-up sensation stack from a list of per-layer sizes.

        `observation_features` is the dimension of layer 0's environment
        observation. `predictor_features[i]` is the dimension of the
        externally-supplied predictor signals that layer `i` consumes;
        these come from outside the coupling in the current scope.
        """
        if len(predictor_features) != len(sizes):
            msg = "predictor_features must have the same length as sizes"
            raise ValueError(msg)
        features: list[SIPFeature] = []
        for i, n in enumerate(sizes):
            content_features = observation_features if i == 0 else 2 * sizes[i - 1]
            feature_sizes = SIPFeatureSizes(
                observation_content=content_features,
                predictor=predictor_features[i],
                gain=0,
            )
            features.append(
                SIPFeature.create(n, feature_sizes, controllable=False, streams=streams)
            )
        return cls(features=tuple(features))

    def infer(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation: NormalNP,
        predictors: Sequence[FeaturePredictors],
        goals: Sequence[JaxRealArray],
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, tuple[SIPFeatureOutputs, ...]]:
        """Run bottom-up inference across the stack.

        Layer 0's `observation_content` is `observation`; each higher layer's
        `observation_content` is the previous layer's `observation_mismatch`.
        `predictors[i]` and `goals[i]` are the externally-supplied predictor
        and goal signals for layer `i`. Returns the per-layer feature outputs.
        """
        if len(predictors) != len(self.features):
            msg = "predictors must have the same length as the feature stack"
            raise ValueError(msg)
        if len(goals) != len(self.features):
            msg = "goals must have the same length as the feature stack"
            raise ValueError(msg)
        keys = jr.split(key, len(self.features))
        outputs: list[SIPFeatureOutputs] = []
        content = observation
        for i, feature in enumerate(self.features):
            state, out = feature.step(
                state,
                keys[i],
                observation_content=content,
                predictors=predictors[i],
                gain=None,
                goal=goals[i],
                streams=streams,
                inference=inference,
            )
            outputs.append(out)
            content = out.observation_mismatch
        return state, tuple(outputs)
