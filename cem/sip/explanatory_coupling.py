"""A real-valued explanatory coupling between two SIP features."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
from tjax import JaxRealArray, RngStream

from cem.sip.emitter import EmitterOutput, SIPEmitter
from cem.sip.score import ScoreOutput, SIPScore


class ExplanatoryCouplingOutput(eqx.Module):
    """Outputs from a source emitter and its downstream target score."""

    source: EmitterOutput
    target: ScoreOutput


class ExplanatoryCoupling(eqx.Module):
    """Couple a source emitter to a downstream target score circuit."""

    emitter: SIPEmitter
    score: SIPScore

    @classmethod
    def create(  # ruff: ignore[too-many-arguments]
        cls,
        innovation_features: int,
        goal_features: int,
        parent_instrument_features: int,
        source_features: int,
        target_features: int,
        *,
        hidden_features: int | tuple[int, ...] = (),
        initial_noise: float = 1e-3,
        learn_noise: bool = True,
        streams: Mapping[str, RngStream],
    ) -> Self:
        emitter = SIPEmitter.create(
            innovation_features,
            goal_features,
            parent_instrument_features,
            source_features,
            hidden_features=hidden_features,
            initial_noise=initial_noise,
            learn_noise=learn_noise,
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
    ) -> ExplanatoryCouplingOutput:
        """Emit the source feature and use it to score the target observation."""
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
        return ExplanatoryCouplingOutput(source=source, target=target)
