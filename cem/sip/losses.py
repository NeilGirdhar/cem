"""Named SIP losses and their scalar reduction."""

from collections.abc import Mapping

import equinox as eqx
from tjax import JaxRealArray


class SIPLossOutputs(eqx.Module):
    """Expose the parts of a SIP training objective that use different weighting rules.

    `balanced_losses` maps each loss name to a scalar whose gradient can be balanced
    against the other named gradients. When combining outputs, add contributions that
    have the same name.

    `fixed_weight_loss` collects the terms that bypass adaptive balancing.

    Use `loss` to sum both parts into a scalar objective.
    """

    balanced_losses: dict[str, JaxRealArray]
    fixed_weight_loss: JaxRealArray

    @property
    def loss(self) -> JaxRealArray:
        return sum(self.balanced_losses.values(), self.fixed_weight_loss)


def combine_losses(*groups: Mapping[str, JaxRealArray]) -> dict[str, JaxRealArray]:
    """Add loss terms with the same name."""
    combined: dict[str, JaxRealArray] = {}
    for group in groups:
        for name, term in group.items():
            combined[name] = combined.get(name, 0.0) + term
    return combined
