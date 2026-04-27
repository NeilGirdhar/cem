from __future__ import annotations

from collections.abc import Callable
from dataclasses import _MISSING_TYPE, MISSING
from typing import TYPE_CHECKING

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from tjax.dataclasses import field

if TYPE_CHECKING:
    from cem.structure.solver.solver import Solver


def bool_field(
    *,
    default: bool,
    static: bool = False,
    optimize: bool = False,
    condition: Callable[[Solver], bool] | None = None,
) -> bool:
    """A Boolean field shown in the UI or optimized by hyperparameter tuning.

    Args:
        default: The default value.
        static: Whether the parameter is static wrt compilation.
        optimize: Whether the parameter should be optimized.
        condition: Optional callable taking the solver instance; if it returns False the
            field is excluded from the hyperparameter search space.
    """
    domain = CategoricalDistribution((False, True))
    return field(
        static=static,
        default=default,
        metadata={"domain": domain, "optimize": optimize, "condition": condition},
    )


def int_field(
    *,
    default: int,
    static: bool = False,
    domain: IntDistribution,
    optimize: bool = False,
    condition: Callable[[Solver], bool] | None = None,
) -> int:
    """An integer field shown in the UI or optimized by hyperparameter tuning.

    Args:
        default: The default value.
        static: Whether the parameter is static wrt compilation.
        domain: The domain of the parameter.
        optimize: Whether the parameter should be optimized.
        condition: Optional callable taking the solver instance; if it returns False the
            field is excluded from the hyperparameter search space.
    """
    return field(
        static=static,
        default=default,
        metadata={"domain": domain, "optimize": optimize, "condition": condition},
    )


def float_field(
    *,
    default: float,
    static: bool = False,
    domain: FloatDistribution,
    optimize: bool = False,
    condition: Callable[[Solver], bool] | None = None,
) -> float:
    """A real-valued field shown in the UI or optimized by hyperparameter tuning.

    Args:
        default: The default value.
        static: Whether the parameter is static wrt compilation.
        domain: The domain of the parameter.
        optimize: Whether the parameter should be optimized.
        condition: Optional callable taking the solver instance; if it returns False the
            field is excluded from the hyperparameter search space.
    """
    return field(
        static=static,
        default=default,
        metadata={"domain": domain, "optimize": optimize, "condition": condition},
    )


def chooser_field[T: int | float | str | None](
    *,
    default: T | _MISSING_TYPE = MISSING,
    static: bool = True,
    options: tuple[T, ...],
    optimize: bool = False,
) -> T:
    """A categorical-valued field shown in the UI or optimized by hyperparameter tuning.

    Args:
        default: The default value.
        static: Whether the parameter is static wrt compilation.
        options: A tuple of option choices (the corresponding option names will be stringified).
        optimize: Whether the parameter should be optimized.
    """
    domain = CategoricalDistribution(options)
    metadata = {"domain": domain, "optimize": optimize}
    if default == MISSING:
        return field(static=True, kw_only=True, metadata=metadata)
    return field(static=True, kw_only=True, default=default, metadata=metadata)
