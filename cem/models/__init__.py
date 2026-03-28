"""Causal neural network models.

Currently provides :class:`AFP` (Adversarial Factor Purification), which separates
endogenous (confounded) and exogenous (causally valid) contributions to a prediction via
two adversarial critics.
"""

from .afp import AFP, AFPConfiguration

__all__ = ["AFP", "AFPConfiguration"]
