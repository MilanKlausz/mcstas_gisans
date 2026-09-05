"""
Unit tests for fit.calculate_fitness, covering the numpy-vs-scipp-mask
refactor's edge cases (mask floors, zero-intensity pixels, stray non-finite
values outside the geometric mask, fully-masked and unmasked data).
Expected values were computed with the original NaN-indexed NumPy
implementation before the scipp-based rewrite, on this same fixed seed/data,
so this also serves as a regression guard for that refactor.
"""
import numpy as np
import pytest

from mcstas_gisans.fit import calculate_fitness
from mcstas_gisans.masking import apply_mask


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    shape = (10, 8)
    hist_nxs_raw = rng.poisson(50, shape).astype(float)
    hist_sim_raw = hist_nxs_raw + rng.normal(0, 3, shape)
    hist_nxs_error_raw = np.sqrt(hist_nxs_raw)
    hist_sim_error_raw = np.sqrt(np.abs(hist_sim_raw))
    mask = np.ones(shape, dtype=bool)
    mask[3:6, 2:5] = False  # exclude a box, like a specular-peak mask
    return hist_nxs_raw, hist_nxs_error_raw, hist_sim_raw, hist_sim_error_raw, mask


def test_basic_masked_fitness(synthetic_data):
    hist_nxs_raw, hist_nxs_error_raw, hist_sim_raw, hist_sim_error_raw, mask = synthetic_data
    hist_nxs = apply_mask(hist_nxs_raw, mask, np.nan)
    hist_nxs_error = apply_mask(hist_nxs_error_raw, mask, 0.0)

    reduced_chi2, log_residual = calculate_fitness(hist_nxs, hist_nxs_error, hist_sim_raw, hist_sim_error_raw)

    assert reduced_chi2 == pytest.approx(0.08928430702728465)
    assert log_residual == pytest.approx(0.0006869647701394064)


def test_zero_sigma_exp_is_floored(synthetic_data):
    hist_nxs_raw, hist_nxs_error_raw, hist_sim_raw, hist_sim_error_raw, mask = synthetic_data
    hist_nxs = apply_mask(hist_nxs_raw, mask, np.nan)
    hist_nxs_error = apply_mask(hist_nxs_error_raw, mask, 0.0)
    hist_nxs_error[0, 0] = 0.0

    reduced_chi2, log_residual = calculate_fitness(hist_nxs, hist_nxs_error, hist_sim_raw, hist_sim_error_raw)

    assert reduced_chi2 == pytest.approx(0.09136860266354983)
    assert log_residual == pytest.approx(0.0006869647701394064)


def test_zero_intensity_pixel(synthetic_data):
    hist_nxs_raw, hist_nxs_error_raw, hist_sim_raw, hist_sim_error_raw, mask = synthetic_data
    hist_nxs_raw = hist_nxs_raw.copy(); hist_sim_raw = hist_sim_raw.copy()
    hist_nxs_error_raw = hist_nxs_error_raw.copy(); hist_sim_error_raw = hist_sim_error_raw.copy()
    hist_nxs_raw[1, 1] = 0.0; hist_nxs_error_raw[1, 1] = 0.0
    hist_sim_raw[1, 1] = 0.0; hist_sim_error_raw[1, 1] = 0.0

    hist_nxs = apply_mask(hist_nxs_raw, mask, np.nan)
    hist_nxs_error = apply_mask(hist_nxs_error_raw, mask, 0.0)

    reduced_chi2, log_residual = calculate_fitness(hist_nxs, hist_nxs_error, hist_sim_raw, hist_sim_error_raw)

    assert reduced_chi2 == pytest.approx(0.08927504245583563)
    assert log_residual == pytest.approx(0.0006967154066283796)


def test_stray_nan_in_unmasked_sim_data_is_excluded(synthetic_data):
    """hist_sim is passed in *unmasked* -- a stray non-finite value there
    (outside the geometric mask) must still be excluded via the isfinite
    safety net, matching the pre-refactor behavior."""
    hist_nxs_raw, hist_nxs_error_raw, hist_sim_raw, hist_sim_error_raw, mask = synthetic_data
    hist_sim_raw = hist_sim_raw.copy()
    hist_sim_raw[7, 7] = np.nan

    hist_nxs = apply_mask(hist_nxs_raw, mask, np.nan)
    hist_nxs_error = apply_mask(hist_nxs_error_raw, mask, 0.0)

    reduced_chi2, log_residual = calculate_fitness(hist_nxs, hist_nxs_error, hist_sim_raw, hist_sim_error_raw)

    assert reduced_chi2 == pytest.approx(0.09026806224864062)
    assert log_residual == pytest.approx(0.0006941360188201239)


def test_fully_masked_gives_nan(synthetic_data):
    hist_nxs_raw, hist_nxs_error_raw, hist_sim_raw, hist_sim_error_raw, _ = synthetic_data
    mask_all_false = np.zeros(hist_nxs_raw.shape, dtype=bool)
    hist_nxs = apply_mask(hist_nxs_raw, mask_all_false, np.nan)
    hist_nxs_error = apply_mask(hist_nxs_error_raw, mask_all_false, 0.0)

    reduced_chi2, log_residual = calculate_fitness(hist_nxs, hist_nxs_error, hist_sim_raw, hist_sim_error_raw)

    assert np.isnan(reduced_chi2)
    assert np.isnan(log_residual)


def test_no_masking(synthetic_data):
    hist_nxs_raw, hist_nxs_error_raw, hist_sim_raw, hist_sim_error_raw, _ = synthetic_data

    reduced_chi2, log_residual = calculate_fitness(hist_nxs_raw, hist_nxs_error_raw, hist_sim_raw, hist_sim_error_raw)

    assert reduced_chi2 == pytest.approx(0.08946629927134803)
    assert log_residual == pytest.approx(0.0006802167355547115)
