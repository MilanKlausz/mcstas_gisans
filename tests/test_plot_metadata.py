"""
mg_plot must interpret every simulation file with the instrument configuration stored in
that file (explicit command line values take precedence), and must never modify the
module-level instrument defaults.
"""
import copy
import json

import numpy as np
import pytest
import scipp as sc

from mcstas_gisans import plot
from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults, resolve_instrument_parameters
from mcstas_gisans.plot_cli import create_argparser


def _write_sim_file(path, offset=(0.0, 0.0), alpha=0.24, wavelength=6.0, orientation=1, beam_angle=0.0):
    params = resolve_instrument_parameters('d22')
    params['detector']['direct_beam_centre_offset'] = list(offset)
    params['beam_angle'] = beam_angle
    nx, ny = params['detector']['pixels']
    counts = np.random.default_rng(0).random(nx * ny)
    data = sc.DataArray(sc.array(dims=['detector_id'], values=counts, variances=counts, unit='counts'))
    instrument = sc.DataGroup({
        'name': sc.scalar('d22'),
        'is_tof_instrument': sc.scalar(False),
        'detector_centre_offset_x': sc.scalar(offset[0], unit='m'),
        'detector_centre_offset_y': sc.scalar(offset[1], unit='m'),
        'alpha_inc_deg': sc.scalar(alpha, unit='deg'),
        'beam_angle': sc.scalar(beam_angle, unit='deg'),
        'sample_orientation': sc.scalar(orientation),
        'wavelength_selected': sc.scalar(wavelength, unit='angstrom'),
        'no_gravity': sc.scalar(False),
        'wfm': sc.scalar(False),
        'parameters_json': sc.scalar(json.dumps(params)),
    })
    sc.DataGroup({'data': data, 'instrument': instrument}).save_hdf5(str(path))
    return params


def _expected_edges(params, alpha, wavelength, orientation):
    return Instrument(params, alpha, wavelength, orientation).get_q_pixel_limits(wavelength)


def _args(argv):
    return create_argparser().parse_args(argv)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_each_file_uses_its_own_stored_configuration(tmp_path, order):
    specs = [dict(offset=(0.29, -0.016), alpha=0.24, wavelength=6.0), dict(offset=(0.0, 0.05), alpha=0.5, wavelength=5.0)]
    files, params = [], []
    for i, spec in enumerate(specs):
        path = tmp_path / f"sim{i}.h5"
        params.append(_write_sim_file(path, **spec))
        files.append(str(path))
    defaults_before = copy.deepcopy(instrument_defaults)

    ordered = [files[i] for i in order]
    datasets = plot.get_datasets(_args(['-f', *ordered]))

    for (hist, err, y_edges, z_edges, label), i in zip(datasets, order):
        spec = specs[i]
        y_exp, z_exp = _expected_edges(params[i], spec['alpha'], spec['wavelength'], 1)
        np.testing.assert_allclose(y_edges, y_exp)
        np.testing.assert_allclose(z_edges, z_exp)
    assert instrument_defaults == defaults_before


def test_explicit_command_line_values_override_stored_ones(tmp_path, capsys):
    path = tmp_path / "sim.h5"
    params = _write_sim_file(path, offset=(0.29, -0.016), alpha=0.24, wavelength=5.0)
    # '--alpha=0.5' form (and abbreviations) must be honoured: no sys.argv sniffing
    (_, _, y_edges, z_edges, _), = plot.get_datasets(_args(['-f', str(path), '--alpha=0.5', '--wavelength', '6']))
    y_exp, z_exp = _expected_edges(params, 0.5, 6.0, 1)
    np.testing.assert_allclose(y_edges, y_exp)
    np.testing.assert_allclose(z_edges, z_exp)
    out = capsys.readouterr().out
    assert "--alpha 0.5 overrides" in out and "--wavelength 6.0 overrides" in out


def test_stored_wavelength_is_used_by_default(tmp_path):
    path = tmp_path / "sim.h5"
    params = _write_sim_file(path, wavelength=5.0)
    (_, _, y_edges, z_edges, _), = plot.get_datasets(_args(['-f', str(path)]))
    y_exp, z_exp = _expected_edges(params, 0.24, 5.0, 1)
    np.testing.assert_allclose(z_edges, z_exp)
    y_6, z_6 = _expected_edges(params, 0.24, 6.0, 1)
    assert not np.allclose(z_edges, z_6), "the default 6 Å must not be used when the file stores 5 Å"


def test_nexus_data_use_the_simulation_configuration(tmp_path):
    path = tmp_path / "sim.h5"
    params = _write_sim_file(path, offset=(0.290852, -0.016066), alpha=0.24, wavelength=6.0, orientation=2)
    datasets = plot.get_datasets(_args(['-f', str(path), '--nxs', 'data/paper/d22_measurement/073162.nxs']))
    (_, _, y_nxs, z_nxs, _), (_, _, y_sim, z_sim, _) = datasets
    np.testing.assert_allclose(y_nxs, y_sim)
    np.testing.assert_allclose(z_nxs, z_sim)


def test_legacy_npz_is_summed_over_the_last_axis(tmp_path):
    path = tmp_path / "legacy.npz"
    hist = np.arange(24, dtype=float).reshape(4, 3, 2)
    np.savez(path, hist=hist, error=np.sqrt(hist), yEdges=np.linspace(-1, 1, 5), zEdges=np.linspace(0, 1, 4))
    (h, e, y_edges, z_edges, _), = plot.get_datasets(_args(['-f', str(path)]))
    np.testing.assert_allclose(h, hist.sum(axis=2))
    assert h.shape == (len(y_edges) - 1, len(z_edges) - 1)


def test_normalise_to_nxs_with_poisson_upscaling_does_not_crash(tmp_path):
    path = tmp_path / "sim.h5"
    _write_sim_file(path, offset=(0.290852, -0.016066), orientation=2)
    datasets = plot.get_datasets(_args(['-f', str(path), '--nxs', 'data/paper/d22_measurement/073162.nxs',
                                        '--normalise_to_nxs', '-t', '60']))
    nxs_sum = datasets[0][0].sum()
    assert datasets[1][0].sum() == pytest.approx(nxs_sum)
