"""
Tests for the 1D losses of mg_fit: the Qy profile of the Qz band [--q_min, --q_max]
(fit.qz_band_profile), its losses (*_1d) and --fit_objective.

The expected values come from the construction of the data: hand-summed small arrays, Poisson
counts drawn from a known signal plus a known flat background, and a fake simulation whose 2D and
1D losses have their minima at different, known parameter values.
"""
import argparse
import csv
import os
import sys

import numpy as np
import pytest

from mcstas_gisans import fit
from mcstas_gisans.fit import LOSS_FUNCTIONS, calculate_fitness

RNG = np.random.default_rng(20261008)
LOSS_KEYS_1D = [f"{key}_1d" for key in LOSS_FUNCTIONS]

# 4 Qy bins x 6 Qz bins; Qz edges 0, 0.01, ..., 0.06
Z_EDGES = np.linspace(0.0, 0.06, 7)
Y_EDGES = np.linspace(-0.02, 0.02, 5)


def _small_case():
    nxs = np.arange(24, dtype=float).reshape(4, 6)
    sim = 0.5 * nxs + 1.0
    mc_error = np.full((4, 6), 2.0)
    nxs[0, 2] = np.nan            # a masked pixel in the band
    nxs[2, 1:4] = np.nan          # a Qy bin without any unmasked pixel in the band
    sim[3, 3] = np.nan            # a pixel without a simulated value
    return nxs, sim, mc_error


def test_profile_sums_the_used_pixels_of_the_band():
    nxs, sim, mc_error = _small_case()
    # [0.015, 0.035]: from the bin containing 0.015 (1) to the bin containing 0.035 (3)
    counts, expected, error, n_pixels = fit.qz_band_profile(nxs, sim, mc_error, Z_EDGES, 0.015, 0.035)
    assert fit.qz_band_indices(0.015, 0.035, Z_EDGES) == (1, 3)
    np.testing.assert_array_equal(n_pixels, [2, 3, 0, 2])
    np.testing.assert_allclose(counts[[0, 1, 3]], [1 + 3, 7 + 8 + 9, 19 + 20])
    assert np.isnan(counts[2])    # excluded: no used pixel
    np.testing.assert_allclose(expected[[0, 1, 3]], [0.5 * 4 + 2, 0.5 * 24 + 3, 0.5 * 39 + 2])
    # Monte Carlo variances add: sqrt(n * 2^2)
    np.testing.assert_allclose(error[[0, 1, 3]], np.sqrt([2 * 4.0, 3 * 4.0, 2 * 4.0]))


def test_band_limits_are_clipped_and_ordered():
    assert fit.qz_band_indices(0.035, 0.015, Z_EDGES) == (1, 3)
    assert fit.qz_band_indices(-1.0, 1.0, Z_EDGES) == (0, 5)
    assert fit.qz_band_indices(0.021, 0.022, Z_EDGES) == (2, 2)


def test_empty_bins_are_left_out_of_the_losses():
    nxs, sim, mc_error = _small_case()
    counts, expected, error, _ = fit.qz_band_profile(nxs, sim, mc_error, Z_EDGES, 0.015, 0.035)
    keep = [0, 1, 3]
    assert calculate_fitness(counts, expected, error) == pytest.approx(
        calculate_fitness(counts[keep], expected[keep], error[keep]), nan_ok=True)


def test_profile_is_the_1d_slice_of_the_comparison_plot(tmp_path, monkeypatch):
    """The 1D losses compare exactly the curves of the 1D panel of save_comparison_plot."""
    from mcstas_gisans import plotting_utils
    plotted = []
    monkeypatch.setattr(plotting_utils, "plot_q_1d", lambda values, *a, **kw: plotted.append(np.array(values)))
    nxs, _, mc_error = _small_case()
    sim = 0.5 * np.nan_to_num(nxs) + 1.0                     # simulated everywhere, as in run_simulation_evaluation
    sim_masked = np.where(np.isfinite(nxs), sim, np.nan)    # its hist_sim_masked, shown in the plot
    for q_min, q_max in [(0.015, 0.035), (0.0, 0.06), (0.045, 0.012)]:
        plotted.clear()
        fit.save_comparison_plot(nxs, np.ones_like(nxs), Y_EDGES, Z_EDGES, sim_masked, np.ones_like(nxs),
                                 Y_EDGES, Z_EDGES, q_min, q_max, [-0.02, 0.02], [0.0, 0.06],
                                 str(tmp_path / "plot.png"), "Sim")
        counts, expected, _, n_pixels = fit.qz_band_profile(nxs, sim_masked, mc_error, Z_EDGES, q_min, q_max)
        used = n_pixels > 0
        np.testing.assert_allclose(plotted[0][used], counts[used])
        np.testing.assert_allclose(plotted[1][used], expected[used])
        assert np.all(plotted[0][~used] == 0)                 # shown as 0, left out of the losses


@pytest.mark.parametrize("true_background", [1.0, 6.1])
def test_background_of_the_profile_scales_with_the_summed_pixels(true_background):
    """Fitted on the profile, b is still the background per pixel: each bin gets n_pixels * b."""
    y, z = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(0, 1, 40), indexing='ij')
    signal = 300 * np.exp(-((np.abs(y) - 0.3) / 0.05) ** 2) * np.exp(-3 * z) + 20 * np.exp(-5 * np.abs(y))
    counts = RNG.poisson(signal + true_background).astype(float)
    counts[90:110, :] = np.nan            # a masked stripe: bins without used pixels
    counts[:, 10:12] = np.nan             # masked pixels within the band: fewer pixels per bin
    z_edges = np.linspace(0, 1, 41)
    counts_1d, signal_1d, error_1d, n_pixels = fit.qz_band_profile(counts, signal, np.zeros_like(signal), z_edges, 0.21, 0.59)
    assert set(np.unique(n_pixels)) == {0, 14}   # Qz bins 8-23 without the masked 10, 11
    background, loss = fit.fit_flat_background(counts_1d, signal_1d, error_1d, 'poisson_deviance', pixels_per_bin=n_pixels)
    # 180 bins x 14 pixels: the precision is ~sqrt(b / 2500) per pixel
    assert background == pytest.approx(true_background, abs=0.25)
    assert loss == pytest.approx(1.0, abs=0.3)
    # the loss is that of the profile with n_pixels * b of background
    assert loss == pytest.approx(calculate_fitness(counts_1d, signal_1d + n_pixels * background, error_1d)['poisson_deviance'])


def test_pixels_per_bin_of_one_is_the_pixel_fit():
    signal = RNG.uniform(1, 50, size=500)
    counts = RNG.poisson(signal + 3.0).astype(float)
    error = 0.1 * signal
    plain = fit.fit_flat_background(counts, signal, error, 'poisson_deviance')
    ones = fit.fit_flat_background(counts, signal, error, 'poisson_deviance', pixels_per_bin=np.ones(500))
    assert ones == pytest.approx(plain, rel=1e-6)


# --- run_simulation_evaluation with a fake simulation: which background and which columns ---

class _FakeCoords:
    @staticmethod
    def rotate_detector_image(hist):
        return hist


class _FakeInstrument:
    wavelength_selected = 6.0

    class detector:
        coords = _FakeCoords()

    def __init__(self, y_edges, z_edges):
        self.edges = (y_edges, z_edges)

    def get_q_pixel_limits(self, wavelength=None):
        return self.edges


def _evaluation_case(monkeypatch):
    n_y, n_z = 60, 30
    y_edges, z_edges = np.linspace(-0.1, 0.1, n_y + 1), np.linspace(0.1, 0.25, n_z + 1)
    y, z = np.meshgrid(0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1]), indexing='ij')
    rate = 0.05 * np.exp(-((np.abs(y) - 0.03) / 0.01) ** 2) + 0.01   # counts per second (time 1000 s)
    weights_squared = (0.02 * rate) ** 2
    monkeypatch.setattr(fit, "pack_parameters", lambda args, particle_type: {'instrument': _FakeInstrument(y_edges, z_edges)})
    monkeypatch.setattr(fit, "process_particles", lambda particles, params: {
        'pixelHist': rate, 'pixelHistWeightsSquared': weights_squared})
    counts = RNG.poisson(1000 * rate * 1.2 + 4.0).astype(float)     # a 20% too weak model, background 4
    counts[:, :3] = np.nan                                            # masked
    counts[25:35, 10:15] = np.nan
    return counts, y_edges, z_edges


def _simulated_profile(monkeypatch, counts, z_edges):
    """The profile of the fake simulation's expected counts (without background) and MC error."""
    result = fit.process_particles(None, None)
    return fit.qz_band_profile(counts, 1000 * result['pixelHist'], 1000 * np.sqrt(result['pixelHistWeightsSquared']),
                               z_edges, 0.15, 0.17)


def _args(tmp_path, **kw):
    args = dict(sample_arguments='', no_parallel=True, background=None, fit_background=False, experiment_time=1000.0,
                loss_function='poisson_deviance', q_min=0.0, q_max=0.0, fit_objective='2d', png=False,
                output_dir=str(tmp_path))
    args.update(kw)
    return argparse.Namespace(**args)


def _evaluate(tmp_path, counts, y_edges, z_edges, **kw):
    mask = np.isfinite(counts)
    return fit.run_simulation_evaluation({'radius': 50.0}, _args(tmp_path, **kw), None, 'neutron', counts,
                                         np.sqrt(np.nan_to_num(counts)), y_edges, z_edges, mask)


@pytest.mark.parametrize("background_option", [{'background': 4.0}, {'fit_background': True}])
def test_2d_results_do_not_depend_on_the_band_with_the_2d_objective(tmp_path, monkeypatch, background_option):
    counts, y_edges, z_edges = _evaluation_case(monkeypatch)
    metrics0, record0, _ = _evaluate(tmp_path, counts, y_edges, z_edges, **background_option)
    metrics, record, _ = _evaluate(tmp_path, counts, y_edges, z_edges, q_min=0.15, q_max=0.17, **background_option)
    assert not any(key in metrics0 for key in LOSS_KEYS_1D)
    for key in list(LOSS_FUNCTIONS) + ['background']:
        assert metrics[key] == metrics0[key]
    # the 2D columns keep their names and order, the 1D columns follow them
    assert list(record) == list(record0) + LOSS_KEYS_1D
    assert all(np.isfinite(record[key]) for key in LOSS_KEYS_1D)


def test_1d_losses_with_a_fixed_background(tmp_path, monkeypatch):
    """--background b: each profile bin includes (number of summed pixels) * b."""
    counts, y_edges, z_edges = _evaluation_case(monkeypatch)
    metrics, _, _ = _evaluate(tmp_path, counts, y_edges, z_edges, background=4.0, q_min=0.15, q_max=0.17)
    counts_1d, signal_1d, error_1d, n_pixels = _simulated_profile(monkeypatch, counts, z_edges)
    reference = calculate_fitness(counts_1d, signal_1d + 4.0 * n_pixels, error_1d)
    for key in LOSS_FUNCTIONS:
        assert metrics[f"{key}_1d"] == pytest.approx(reference[key], rel=1e-9)


def test_1d_objective_fits_the_background_on_the_profile(tmp_path, monkeypatch):
    counts, y_edges, z_edges = _evaluation_case(monkeypatch)
    metrics_2d, _, _ = _evaluate(tmp_path, counts, y_edges, z_edges, fit_background=True, q_min=0.15, q_max=0.17)
    metrics_1d, _, _ = _evaluate(tmp_path, counts, y_edges, z_edges, fit_background=True, q_min=0.15, q_max=0.17,
                                 fit_objective='1d')
    counts_1d, signal_1d, error_1d, n_pixels = _simulated_profile(monkeypatch, counts, z_edges)
    background_1d, loss_1d = fit.fit_flat_background(counts_1d, signal_1d, error_1d, 'poisson_deviance', pixels_per_bin=n_pixels)
    assert metrics_1d['background'] == pytest.approx(background_1d)
    assert metrics_1d['poisson_deviance_1d'] == pytest.approx(loss_1d)
    assert metrics_1d['background'] != pytest.approx(metrics_2d['background'], rel=1e-3)
    # each background minimises its own loss
    assert metrics_1d['poisson_deviance_1d'] < metrics_2d['poisson_deviance_1d']
    assert metrics_2d['poisson_deviance'] < metrics_1d['poisson_deviance']


# --- the minimised loss (--fit_objective) in fits and scans, with a fake evaluation ---

def _fake_evaluation(grid_point, args, *a, **kw):
    """2D losses with their minimum at radius 45, 1D losses at 55."""
    r = grid_point['radius']
    metrics = {key: 1.0 + (r - 45.0) ** 2 / 10 for key in LOSS_FUNCTIONS}
    if args.q_max > args.q_min:
        metrics.update({f"{key}_1d": 2.0 + (r - 55.0) ** 2 / 10 for key in LOSS_FUNCTIONS})
    metrics['background'] = 0.0
    record = dict(grid_point)
    record.update({key: value for key, value in metrics.items() if key != 'background'})
    return metrics, record, {}


def _read_csv(path):
    """The records of a summary CSV (the printed summary is appended after an empty line)."""
    with open(path) as f:
        return list(csv.DictReader(f.read().split('\n\n')[0].splitlines()))


@pytest.mark.parametrize("objective, best", [('2d', 45.0), ('1d', 55.0)])
def test_fit_minimises_the_selected_objective(tmp_path, monkeypatch, objective, best):
    monkeypatch.setattr(fit, "run_simulation_evaluation", _fake_evaluation)
    args = _args(tmp_path, q_min=0.15, q_max=0.17, fit_objective=objective, gif=False, nxs2=None, fit2=None,
                 fit_common=None, fit=[['radius', '50', '40', '60']], fit_integer=None, optimizer='nelder-mead',
                 max_evals=60, xatol=1e-4, fatol=1e-6)
    fit.run_automated_fit(args, None, 'neutron', None, None, None, None, None)
    rows = _read_csv(os.path.join(str(tmp_path), "fit_summary.csv"))
    key = 'poisson_deviance' + ('_1d' if objective == '1d' else '')
    assert float(rows[0]['radius']) == pytest.approx(best, abs=0.1)   # sorted by the selected loss
    assert [float(r[key]) for r in rows] == sorted(float(r[key]) for r in rows)
    assert set(LOSS_FUNCTIONS) | set(LOSS_KEYS_1D) <= set(rows[0])


@pytest.mark.parametrize("objective, best", [('2d', '45.0'), ('1d', '55.0')])
def test_scan_is_sorted_by_the_selected_objective(tmp_path, monkeypatch, capsys, objective, best):
    monkeypatch.setattr(fit, "run_simulation_evaluation", _fake_evaluation)
    args = _args(tmp_path, q_min=0.15, q_max=0.17, fit_objective=objective, scan=[['radius', '45.0', '50.0', '55.0']])
    fit.run_parameter_scan(args, None, 'neutron', None, None, None, None, None)
    rows = _read_csv(os.path.join(str(tmp_path), "scan_summary.csv"))
    assert rows[0]['radius'] == best
    out = capsys.readouterr().out
    assert "poisson_deviance_1d=" in out and "reduced_chi2_1d=" in out


# --- validation ---

BASE_ARGV = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0",
             "--nxs", "data/paper/d22_measurement/073174.nxs", "--experiment_time", "60",
             "--fit", "radius", "50", "40", "60"]


def _validated(extra, monkeypatch):
    from mcstas_gisans.run_cli import parse_args
    parser = fit.create_fit_parser()
    monkeypatch.setattr(sys, "argv", ["mg_fit"] + BASE_ARGV + extra)
    args = parse_args(parser)
    fit.validate_fit_args(args, parser)
    return args


@pytest.mark.parametrize("band", [[], ['--q_min', '0.17', '--q_max', '0.15'], ['--q_min', '0.15', '--q_max', '0.15']])
def test_1d_objective_needs_a_qz_band(band, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _validated(['--fit_objective', '1d'] + band, monkeypatch)
    assert '--fit_objective 1d compares the Qy profiles of the Qz band' in capsys.readouterr().err


def test_objective_options_are_accepted(monkeypatch):
    assert _validated([], monkeypatch).fit_objective == '2d'
    args = _validated(['--fit_objective', '1d', '--q_min', '0.15', '--q_max', '0.17'], monkeypatch)
    assert fit.objective_loss_key(args) == 'poisson_deviance_1d'


def test_1d_objective_needs_unmasked_pixels_in_the_band(monkeypatch):
    from mcstas_gisans.run_cli import parse_args
    argv = ["mg_fit", "--nxs", "data/paper/d22_measurement/073174.nxs", "-i", "d22", "--wavelength_selected", "6.0",
            "--alpha", "0.24", "--sample_orientation", "2", "--instrument_detector_centre_offset", "0.290838", "-0.016061",
            "--mask_qz_min_cut", "0.2", "--q_min", "0.15", "--q_max", "0.17", "--fit_objective", "1d"]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args(fit.create_fit_parser())
    with pytest.raises(ValueError, match="no unmasked pixel"):
        fit.prepare_experimental_data(args)


def test_mc_warning_refers_to_the_minimised_loss(monkeypatch, capsys):
    """With --fit_objective 1d, the Monte Carlo discount of the 1D loss decides (and is reported)."""
    monkeypatch.setattr(fit, '_MC_WARNING_ISSUED', [False])
    model = np.full(200, 100.0)
    counts = RNG.poisson(model).astype(float)
    metrics = calculate_fitness(counts, model, np.sqrt(0.01 * model))           # pixels: small MC variance
    counts_1d = np.array([100.0, 160.0, 40.0])
    profile = calculate_fitness(counts_1d, np.full(3, 100.0), np.full(3, 60.0))  # profile: MC dominated
    metrics.update({f"{key}_1d": value for key, value in profile.items()})
    fit._warn_if_mc_uncertainty_large(metrics, argparse.Namespace(loss_function='poisson_deviance', fit_objective='2d'))
    assert capsys.readouterr().out == ""
    fit._warn_if_mc_uncertainty_large(metrics, argparse.Namespace(loss_function='poisson_deviance', fit_objective='1d'))
    out = capsys.readouterr().out
    assert f"lowers the poisson_deviance_1d from {profile['poisson_deviance_without_mc']:.4g}" in out
