"""Expose a SIP agent through the shared model and optimizer interfaces."""

from collections.abc import Mapping
from typing import Self, override

import equinox as eqx
from efax import NormalNP
from jax import tree
from tjax import RngStream
from tjax.gradient import Adam, ChainedGradientTransformation, KeepParamsNonnegative

from cem.sip.parameters import WitnessMultiplierParameter
from cem.sip.sip_agent import SIPAgent
from cem.structure.graph import (
    DisGradientTransformation,
    FixedParameter,
    LearnableParameter,
    Model,
    ModelResult,
    NodeConfiguration,
    ParameterType,
)
from cem.structure.problem.data_source import ProblemObservation


class SIPObservation(ProblemObservation):
    """The environment evidence and reward supplied to a SIP agent."""

    observation: NormalNP
    reward: NormalNP


class SIPActionConfiguration(NodeConfiguration):
    """The action evidence produced by the agent."""

    action: NormalNP


class SIPModel(Model):
    """Adapt an agent and its recurrent state to the shared training loop."""

    agent: SIPAgent
    initial_agent_state: FixedParameter[eqx.nn.State]

    @classmethod
    def create(cls, agent: SIPAgent) -> Self:
        agent, state = eqx.nn.make_with_state(lambda: agent)()
        return cls(agent, FixedParameter(state))

    @staticmethod
    def gradient_transformations(learning_rate: float) -> DisGradientTransformation:
        """Apply Adam to every learnable parameter.

        Witness multipliers are projected onto the nonnegative reals after each Adam step.
        """
        return DisGradientTransformation(
            [
                (ParameterType(FixedParameter), None),
                (
                    ParameterType(WitnessMultiplierParameter),
                    ChainedGradientTransformation[Model](
                        [Adam[Model](learning_rate), KeepParamsNonnegative[Model]()]
                    ),
                ),
                (ParameterType(LearnableParameter), Adam[Model](learning_rate)),
            ]
        )

    @override
    def initial_state(self) -> eqx.nn.State:
        return tree.map(lambda value: value, self.initial_agent_state.value)

    @override
    def infer(
        self,
        observation: object,
        state: object,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ModelResult:
        assert isinstance(observation, SIPObservation)
        assert isinstance(state, eqx.nn.State)
        state, outputs = self.agent.step(
            state,
            streams["inference"].key(),
            observation=observation.observation,
            environment_reward_observation=observation.reward,
            streams=streams,
            inference=inference,
        )
        return ModelResult(
            configurations=frozendict({"action": SIPActionConfiguration(outputs.action)}),
            state=state,
            fixed_weight_loss=outputs.loss,
        )
