import math

import pytest

from cem.sip import MismatchSettings, TDSettings


def test_mismatch_damping_weight_may_be_zero() -> None:
    assert MismatchSettings(damping_weight=0.0).damping_weight == 0.0
    assert MismatchSettings().damping_weight == 1.0


@pytest.mark.parametrize("damping_weight", [-0.5, math.inf, math.nan])
def test_mismatch_damping_weight_rejects_negative_and_nonfinite(damping_weight: float) -> None:
    with pytest.raises(ValueError, match="damping_weight"):
        MismatchSettings(damping_weight=damping_weight)


@pytest.mark.parametrize("damping_weight", [0.0, -0.5, math.inf])
def test_td_damping_weight_must_be_positive(damping_weight: float) -> None:
    with pytest.raises(ValueError, match="damping_weight"):
        TDSettings(damping_weight=damping_weight)
