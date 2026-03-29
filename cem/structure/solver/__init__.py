"""Solver: hyperparameter fields, execution context management, and path utilities."""

from .context_manager import jax_is_initialized, solver_context_manager
from .hp_field import bool_field, chooser_field, float_field, int_field
from .solver import Solver

__all__ = [
    "Solver",
    "bool_field",
    "chooser_field",
    "float_field",
    "int_field",
    "jax_is_initialized",
    "solver_context_manager",
]
