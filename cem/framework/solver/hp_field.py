"""Declare solver fields that can participate in an Optuna search.

Set `optimize` to true to include a field. Where supported, a condition that
returns false excludes the field for that solver configuration.
"""

from collections.abc import Callable
from dataclasses import MISSING

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from tjax.dataclasses import field

lazy from cem.framework.solver.solver import Solver

type IntHyperparameterDistribution = IntDistribution | CategoricalDistribution


def hardware_friendly_ints(
    lower_bound: int,
    upper_bound: int,
    *,
    values_per_octave: int = 3,
) -> tuple[int, ...]:
    """Return geometrically spaced integer sizes within the supplied bounds.

    The default choices include four, five, six, and eight. This gives a search
    several widths per octave without compiling a model for every adjacent size.
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
    """Declare a Boolean solver field with both values as search choices."""
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
    """Declare an integer solver field with a search range or discrete choices."""
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
    """Declare a real-valued solver field with an Optuna search range."""
    return field(
        static=static,
        default=default,
        metadata={"domain": domain, "optimize": optimize, "condition": condition},
    )


def chooser_field[T: int | float | str | None](
    *,
    default: T | MISSING = MISSING,  # ty: ignore[invalid-type-form]
    static: bool = True,
    options: tuple[T, ...],
    optimize: bool = False,
) -> T:
    """Declare a static keyword-only solver field with categorical search choices.

    Omit the default to require an explicit value when constructing the solver.
    """
    domain = CategoricalDistribution(options)
    metadata = {"domain": domain, "optimize": optimize}
    if default is MISSING:
        return field(static=True, kw_only=True, metadata=metadata)
    return field(static=True, kw_only=True, default=default, metadata=metadata)
