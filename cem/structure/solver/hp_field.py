from __future__ import annotations

from collections.abc import Callable
from dataclasses import _MISSING_TYPE, MISSING
from typing import TYPE_CHECKING

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from tjax.dataclasses import field

if TYPE_CHECKING:
    from cem.structure.solver.solver import Solver

type IntHyperparameterDistribution = IntDistribution | CategoricalDistribution


def hardware_friendly_ints(
    lower_bound: int,
    upper_bound: int,
    *,
    values_per_octave: int = 3,
) -> tuple[int, ...]:
    """Return a geometric basis of integer shape values.

    The default mantissas produce choices such as ``4, 5, 6, 8, ...``: close enough
    to search meaningful capacity changes without compiling every adjacent shape.
    """
    if lower_bound < 1:
        msg = f"lower_bound must be >= 1, got {lower_bound}"
        raise ValueError(msg)
    if upper_bound < lower_bound:
        msg = f"upper_bound must be >= lower_bound, got {upper_bound} < {lower_bound}"
        raise ValueError(msg)
    if values_per_octave < 1:
        msg = f"values_per_octave must be >= 1, got {values_per_octave}"
        raise ValueError(msg)

    if values_per_octave == 1:
        mantissas = (1.0,)
    else:
        mantissas = tuple(1.0 + 0.5 * i / (values_per_octave - 1) for i in range(values_per_octave))
    values: set[int] = set()
    octave = 1
    while octave <= upper_bound:
        for mantissa in mantissas:
            value = int(mantissa * octave + 0.5)
            if lower_bound <= value <= upper_bound:
                values.add(value)
        octave *= 2
    return tuple(sorted(values))


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
    domain: IntHyperparameterDistribution,
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
