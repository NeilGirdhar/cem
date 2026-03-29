from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
from tjax import JaxArray, RngStream


class BatchLoss(eqx.Module):
    """These objects are evaluated once for every minibatch.

    They allow loss functions that depend on the entire minibatch.
    """

    def loss(self, streams: Mapping[str, RngStream]) -> JaxArray:
        raise NotImplementedError
