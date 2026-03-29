"""Solver and trainer: hyperparameter fields, execution context management, and path utilities."""

from .context_manager import jax_is_initialized, solver_context_manager
from .solver import Solver, Trainer
from .hp_field import bool_field, chooser_field, float_field, int_field

__all__ = [
    "Solver",
    "Trainer",
    "bool_field",
    "chooser_field",
    "float_field",
    "int_field",
    "jax_is_initialized",
    "solver_context_manager",
]
