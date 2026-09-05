"""
Unit tests for experiment_time.upscale_simple, which had no direct test
coverage before (only indirectly exercised, via CLI --experiment_time, by a
few regression tests that never enable Poisson sampling).
"""
import numpy as np
import pytest

from mcstas_gisans.experiment_time import upscale_simple


@pytest.fixture
def synthetic_hist():
    rng = np.random.default_rng(1)
    hist = rng.poisson(20, (5, 4)).astype(float)
    hist_error = np.sqrt(hist)
    return hist, hist_error


def test_deterministic_scaling_matches_manual_calculation(synthetic_hist):
    hist, hist_error = synthetic_hist
    experiment_time, background = 100.0, 2.0

    scaled_hist, scaled_error = upscale_simple(hist, hist_error, experiment_time, background, poisson_sampling=False)

    expected_hist = hist * experiment_time + background
    expected_error = np.sqrt((hist_error * experiment_time) ** 2 + expected_hist)
    assert np.allclose(scaled_hist, expected_hist)
    assert np.allclose(scaled_error, expected_error)


def test_does_not_mutate_caller_arrays(synthetic_hist):
    """Regression guard: the pre-refactor implementation mutated hist/hist_error
    in place (hist *= experiment_time), silently corrupting the caller's array."""
    hist, hist_error = synthetic_hist
    hist_before = hist.copy()
    hist_error_before = hist_error.copy()

    upscale_simple(hist, hist_error, 100.0, 2.0, poisson_sampling=False)

    assert np.array_equal(hist, hist_before)
    assert np.array_equal(hist_error, hist_error_before)


def test_poisson_sampling_reproducible_with_seeded_rng(synthetic_hist):
    hist, hist_error = synthetic_hist
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    result1 = upscale_simple(hist, hist_error, 50.0, 1.0, poisson_sampling=True, rng=rng1)
    result2 = upscale_simple(hist, hist_error, 50.0, 1.0, poisson_sampling=True, rng=rng2)

    assert np.array_equal(result1[0], result2[0])
    assert np.array_equal(result1[1], result2[1])


def test_poisson_sampling_error_is_sqrt_of_sampled_counts(synthetic_hist):
    hist, hist_error = synthetic_hist
    sampled_hist, sampled_error = upscale_simple(
        hist, hist_error, 50.0, 1.0, poisson_sampling=True, rng=np.random.default_rng(7)
    )
    assert np.allclose(sampled_error, np.sqrt(sampled_hist))


def test_poisson_sampling_matches_expected_lambda_on_average():
    """Statistical sanity check: over many bins with the same lambda, the
    sampled mean should be close to the deterministic expected value."""
    n = 20000
    hist = np.full(n, 30.0)
    hist_error = np.sqrt(hist)
    experiment_time, background = 10.0, 5.0
    expected_lambda = 30.0 * 10.0 + 5.0  # = 305

    sampled_hist, _ = upscale_simple(
        hist, hist_error, experiment_time, background, poisson_sampling=True, rng=np.random.default_rng(99)
    )
    assert sampled_hist.mean() == pytest.approx(expected_lambda, rel=0.02)
