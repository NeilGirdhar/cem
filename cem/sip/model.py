"""Small real-valued SIP chains used for integration tests."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
from tjax import JaxRealArray, RngStream

from cem.sip.emitter import EmitterOutput, SIPEmitter
from cem.sip.score import ScoreOutput, SIPScore


class SIPChainOutput(eqx.Module):
    """Outputs from one source-emitter to target-score chain."""

    source: EmitterOutput
    target: ScoreOutput


class SIPChain(eqx.Module):
    """Connect one real-valued emitter to one real-valued score circuit."""

    emitter: SIPEmitter
    score: SIPScore

    @classmethod
    def create(
        cls,
        innovation_features: int,
        goal_features: int,
        parent_instrument_features: int,
        source_features: int,
        target_features: int,
        *,
        hidden_features: int | tuple[int, ...] = (),
        initial_noise: float = 1e-3,
        streams: Mapping[str, RngStream],
    ) -> Self:
        emitter = SIPEmitter.create(
            innovation_features,
            goal_features,
            parent_instrument_features,
            source_features,
            hidden_features=hidden_features,
            initial_noise=initial_noise,
            streams=streams,
        )
        score = SIPScore.create(
            source_features,
            source_features,
            target_features,
            hidden_features=hidden_features,
            streams=streams,
        )
        return cls(emitter=emitter, score=score)

    def infer(
        self,
        innovation: JaxRealArray,
        goal: JaxRealArray,
        source_gain: JaxRealArray,
        parent_instruments: JaxRealArray,
        target_observation: JaxRealArray,
        target_gain: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> SIPChainOutput:
        """Run the source emitter and score its target."""
        source = self.emitter.infer(
            innovation,
            goal,
            source_gain,
            parent_instruments,
            streams=streams,
            inference=inference,
        )
        target = self.score.infer(
            target_observation,
            source.observation,
            source.instrument,
            target_gain,
            streams=streams,
            inference=inference,
        )
        return SIPChainOutput(source=source, target=target)
