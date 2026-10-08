"""
Tests for fit.calculate_fitness.

The expected values are derived from statistics, not from the implementation:
Poisson data generated from a known model must give a Poisson deviance of ~1 per pixel,
and fitting an intensity scale with the loss must recover the true scale -- also at low
counts, where a chi^2 with the Poisson variance counted twice (the previous definition)
is biased.
"""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from mcstas_gisans.fit import LOSS_FUNCTIONS, calculate_fitness, poisson_deviance_with_mc

RNG = np.random.default_rng(20260924)


def _poisson_data(expected, size=200_000):
    model = np.full(size, float(expected))
    counts = RNG.poisson(model).astype(float)
    return counts, model


@pytest.mark.parametrize("expected", [5.0, 20.0, 200.0])
def test_perfect_model_gives_unit_deviance_and_chi2(expected):
    counts, model = _poisson_data(expected)
    metrics = calculate_fitness(counts, model, np.zeros_like(model))
    assert metrics['poisson_deviance'] == pytest.approx(1.0, abs=0.05)
    assert metrics['reduced_chi2'] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("expected", [0.5, 2.0, 5.0])
def test_fitted_intensity_scale_is_unbiased_at_low_counts(expected):
    """argmin_f deviance(N, f*m) must be the true scale (1), even for a few counts per pixel."""
    counts, model = _poisson_data(expected)
    result = minimize_scalar(lambda f: calculate_fitness(counts, f * model, np.zeros_like(model))['poisson_deviance'],
                             bounds=(0.5, 1.5), method='bounded')
    assert result.x == pytest.approx(1.0, abs=0.01)


def test_chi2_includes_monte_carlo_variance():
    counts, model = _poisson_data(50.0)
    mc_error = np.sqrt(model)  # MC variance equal to the Poisson variance
    without = calculate_fitness(counts, model, np.zeros_like(model))
    with_mc = calculate_fitness(counts, model, mc_error)
    assert with_mc['reduced_chi2'] == pytest.approx(without['reduced_chi2'] / 2, rel=1e-9)
    assert with_mc['mc_to_poisson_variance'] == pytest.approx(1.0)
    assert without['mc_to_poisson_variance'] == 0.0


def test_deviance_reduces_to_poisson_deviance_without_mc_uncertainty():
    counts = np.array([0.0, 1.0, 3.0, 10.0, 100.0, 1e4, 2e5])
    model = np.array([0.5, 2.0, 3.0, 12.0, 90.0, 1.1e4, 5e3])
    poisson = 2 * (model - counts + np.where(counts > 0, counts * np.log(np.where(counts > 0, counts, 1) / model), 0))
    np.testing.assert_allclose(poisson_deviance_with_mc(counts, model, np.zeros_like(model)), poisson, atol=1e-9)
    np.testing.assert_allclose(poisson_deviance_with_mc(counts, model, np.full_like(model, 1e-12)), poisson, atol=1e-8)


@pytest.mark.parametrize("expected, mc_rel", [(5.0, 0.1), (200.0, 0.1), (200.0, 1.0)])
def test_deviance_for_data_with_monte_carlo_and_counting_noise(expected, mc_rel):
    """
    Data drawn from Poisson(gamma(mean m, var s^2)) (counting noise on an MC-uncertain
    expectation): the deviance per pixel is ~1 + ln(1 + s^2/m) at high counts (the Pearson
    term plus the likelihood normalisation), and the plain Poisson deviance is larger.
    """
    size = 200_000
    s2 = mc_rel * expected
    true = RNG.gamma(expected ** 2 / s2, s2 / expected, size)
    counts = RNG.poisson(true).astype(float)
    model = np.full(size, expected)
    with_mc = poisson_deviance_with_mc(counts, model, np.full(size, s2)).mean()
    plain = poisson_deviance_with_mc(counts, model, np.zeros(size)).mean()
    if expected >= 100:
        assert with_mc == pytest.approx(1 + np.log(1 + mc_rel), abs=0.03)
    assert with_mc < plain


@pytest.mark.parametrize("expected, mc_rel", [(2.0, 0.2), (5.0, 0.5)])
def test_fitted_scale_is_unbiased_with_monte_carlo_uncertainty(expected, mc_rel):
    size = 200_000
    s2 = mc_rel * expected
    true = RNG.gamma(expected ** 2 / s2, s2 / expected, size)
    counts = RNG.poisson(true).astype(float)
    model = np.full(size, expected)
    # scaling the simulation by f scales its MC variance by f^2
    result = minimize_scalar(lambda f: calculate_fitness(counts, f * model, f * np.sqrt(np.full(size, s2)))['poisson_deviance'],
                             bounds=(0.5, 1.5), method='bounded')
    assert result.x == pytest.approx(1.0, abs=0.01)


def test_large_mc_uncertainty_reduces_the_weight_of_a_pixel():
    counts, model = np.array([2e5]), np.array([5e3])
    assert calculate_fitness(counts, model, np.array([3e4]))['poisson_deviance'] < calculate_fitness(counts, model, np.zeros(1))['poisson_deviance'] / 10


def test_masked_and_non_finite_pixels_are_excluded():
    counts = np.array([10.0, np.nan, 7.0])
    model = np.array([10.0, 1e6, np.nan])
    metrics = calculate_fitness(counts, model, np.zeros(3))
    assert metrics['poisson_deviance'] == pytest.approx(0.0)
    assert metrics['reduced_chi2'] == pytest.approx(0.0)


def test_fully_masked_gives_nan():
    metrics = calculate_fitness(np.full(4, np.nan), np.ones(4), np.zeros(4))
    assert all(np.isnan(v) for v in metrics.values())


def test_zero_prediction_for_measured_counts_is_heavily_penalised():
    metrics = calculate_fitness(np.array([5.0]), np.array([0.0]), np.array([0.0]))
    assert np.isfinite(metrics['poisson_deviance']) and metrics['poisson_deviance'] > 100


def _few_mc_dominated_pixels(n_pixels=1000, n_bad=40):
    """
    A model that matches the counts (MC variance 0.1 m) except in 4% of the pixels, where it
    overshoots the counts 2.5x and the MC variance is 10x the counting variance (a steep part of the
    pattern on a coarse outgoing-direction grid). The 95th percentile of sigma^2 / m is only 0.1.
    """
    model = np.full(n_pixels, 50.0)
    counts = model.copy()
    mc_variance = 0.1 * model
    counts[:n_bad] = 20.0
    mc_variance[:n_bad] = 10 * model[:n_bad]
    return counts, model, np.sqrt(mc_variance)


def test_mc_discount_of_few_mc_dominated_pixels():
    counts, model, mc_error = _few_mc_dominated_pixels()
    metrics = calculate_fitness(counts, model, mc_error)
    without = calculate_fitness(counts, model, np.zeros_like(model))
    assert metrics['mc_to_poisson_variance'] == pytest.approx(0.1)
    assert metrics['mc_dominated_fraction'] == pytest.approx(0.04)
    # chi^2: only the 40 bad pixels contribute, (20 - 50)^2 / 50 without and / (50 + 500) with MC
    assert metrics['reduced_chi2_without_mc'] == pytest.approx(0.04 * 900 / 50)
    assert metrics['mc_discount_reduced_chi2'] == pytest.approx(1 - 50 / 550)
    assert metrics['poisson_deviance_without_mc'] == pytest.approx(without['poisson_deviance'])
    assert metrics['mc_discount_poisson_deviance'] == pytest.approx(1 - metrics['poisson_deviance'] / without['poisson_deviance'])
    assert metrics['mc_discount_poisson_deviance'] > 0.5
    # without MC variance there is nothing to discount
    assert without['mc_discount_poisson_deviance'] == pytest.approx(0.0, abs=1e-12)
    assert without['mc_discount_reduced_chi2'] == 0.0


@pytest.mark.parametrize("loss_function", ['poisson_deviance', 'reduced_chi2'])
def test_mc_warning_when_few_pixels_lower_the_loss(monkeypatch, capsys, loss_function):
    from types import SimpleNamespace
    from mcstas_gisans import fit
    monkeypatch.setattr(fit, '_MC_WARNING_ISSUED', [False])
    metrics = calculate_fitness(*_few_mc_dominated_pixels())
    assert metrics['mc_to_poisson_variance'] < fit.MC_TO_POISSON_WARNING_RATIO  # the 5% criterion alone is silent
    args = SimpleNamespace(loss_function=loss_function)
    fit._warn_if_mc_uncertainty_large(metrics, args)
    out = capsys.readouterr().out
    assert "WARNING" in out and "4.0% of the unmasked pixels" in out
    assert f"lowers the {loss_function} from {metrics[f'{loss_function}_without_mc']:.4g}" in out
    fit._warn_if_mc_uncertainty_large(metrics, args)  # once per run
    assert capsys.readouterr().out == ""


def test_no_mc_warning_for_small_mc_variance(monkeypatch, capsys):
    from types import SimpleNamespace
    from mcstas_gisans import fit
    monkeypatch.setattr(fit, '_MC_WARNING_ISSUED', [False])
    counts, model, _ = _few_mc_dominated_pixels()  # the same misfit, but small MC variance everywhere
    metrics = calculate_fitness(counts, model, np.sqrt(0.01 * model))
    assert metrics['mc_discount_poisson_deviance'] < fit.MC_DISCOUNT_WARNING_FRACTION
    for loss_function in LOSS_FUNCTIONS:
        fit._warn_if_mc_uncertainty_large(metrics, SimpleNamespace(loss_function=loss_function))
    assert capsys.readouterr().out == ""


def test_mc_warning_for_many_mc_dominated_pixels_also_with_log_residual(monkeypatch, capsys):
    from types import SimpleNamespace
    from mcstas_gisans import fit
    monkeypatch.setattr(fit, '_MC_WARNING_ISSUED', [False])
    counts, model = _poisson_data(50.0, size=1000)
    fit._warn_if_mc_uncertainty_large(calculate_fitness(counts, model, np.sqrt(2 * model)),
                                      SimpleNamespace(loss_function='log_residual'))
    assert "95th percentile of their ratio: 2.00" in capsys.readouterr().out
