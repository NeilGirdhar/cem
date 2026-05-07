from __future__ import annotations

import pytest

from cem.structure.solver import hardware_friendly_ints


def test_hardware_friendly_ints_returns_sorted_unique_values_in_bounds() -> None:
    assert hardware_friendly_ints(4, 128) == (
        4,
        5,
        6,
        8,
        10,
        12,
        16,
        20,
        24,
        32,
        40,
        48,
        64,
        80,
        96,
        128,
    )


def test_hardware_friendly_ints_includes_small_anchors() -> None:
    assert hardware_friendly_ints(1, 16) == (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
    assert hardware_friendly_ints(2, 16) == (2, 3, 4, 5, 6, 8, 10, 12, 16)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "values_per_octave", "match"),
    [
        (0, 16, 3, "lower_bound"),
        (16, 1, 3, "upper_bound"),
        (1, 16, 0, "values_per_octave"),
    ],
)
def test_hardware_friendly_ints_validates_inputs(
    lower_bound: int,
    upper_bound: int,
    values_per_octave: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        hardware_friendly_ints(
            lower_bound,
            upper_bound,
            values_per_octave=values_per_octave,
        )
