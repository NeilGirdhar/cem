from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
from tjax import JaxAbstractClass, JaxArray, RngStream, frozendict

from .node import NodeConfiguration


class ModelResult(eqx.Module):
    """Result of one model inference step."""

    loss: JaxArray
    configurations: frozendict[str, NodeConfiguration]
    state: Any


class Model(eqx.Module, JaxAbstractClass):
    """Abstract base class for all CEM models.

    Concrete models implement :meth:`infer` and optionally override :meth:`initial_state`.
    """

    def initial_state(self) -> object:
        """Return the initial recurrent state for a new episode.

        Feedforward models return ``frozendict()``; recurrent models return a pytree
        of JAX arrays carrying whatever state they need across time steps.
        """
        return frozendict()

    def infer(
        self,
        observation: object,
        state: object,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ModelResult:
        """Run one forward pass.

        Args:
            observation: The current environment observation (a dataclass or similar pytree).
            state: Recurrent state from the previous step (or ``initial_state()`` for the first).
            streams: Named RNG streams for stochastic operations.
            inference: If ``True``, use inference-time behaviour (e.g. no dropout).

        Returns:
            A :class:`ModelResult` with scalar ``loss``, per-node ``configurations``, and
            updated ``state`` for the next step.
        """
        raise NotImplementedError
