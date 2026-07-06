from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.intervals import (  # noqa: E402
    asymptotic_variance,
    empirical_coverage,
    interval_width,
    linear_coefficient_ci,
    mean_ci,
    ratio_delta_ci,
    z_value,
)


def test_default_z_value_is_90_percent_two_sided_quantile():
    assert z_value() == pytest.approx(1.64485, abs=1e-5)
    assert z_value(0.95) == pytest.approx(1.95996, abs=1e-5)


def test_mean_ci_uses_influence_variance_over_sample_size():
    influence = np.array([-1.0, 0.0, 1.0])
    point = 2.0

    ci = mean_ci(point, influence)
    radius = z_value() * np.sqrt(asymptotic_variance(influence) / influence.size)

    assert ci == pytest.approx((point - radius, point + radius))


def test_interval_width_uses_endpoint_difference():
    assert interval_width((1.25, 3.75)) == pytest.approx(2.5)


def test_empirical_coverage_includes_interval_boundaries():
    intervals = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )

    assert empirical_coverage(intervals, truth=2.0) == pytest.approx(0.5)


def test_ratio_delta_ci_has_finite_endpoints_and_rejects_zero_denominator():
    ci = ratio_delta_ci(
        numer_point=2.0,
        denom_point=4.0,
        influence_numer=np.array([-0.2, 0.0, 0.1, 0.3]),
        influence_denom=np.array([0.1, -0.1, 0.2, -0.2]),
    )

    assert np.all(np.isfinite(ci))
    assert ci[0] < 0.5 < ci[1]
    with pytest.raises(ValueError, match="zero"):
        ratio_delta_ci(1.0, 1e-14, [0.0, 1.0], [1.0, 0.0])


def test_linear_coefficient_ci_uses_covariance_divided_by_n():
    beta = np.array([1.0, 2.0])
    covariance = np.array([[1.0, 0.0], [0.0, 4.0]])

    ci = linear_coefficient_ci(beta, covariance, n=100, index=1)
    radius = z_value() * np.sqrt(covariance[1, 1] / 100)

    assert ci == pytest.approx((2.0 - radius, 2.0 + radius))
    assert interval_width(ci) == pytest.approx(2 * radius)
