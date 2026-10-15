import h5py
import numpy as np
import pytest

from mcstas_gisans.nexus_reader import read_sans2d_events, sans2d_rear_pixel_row


def _write_raw_with_events(path, event_spectra, event_tof_us):
    # spectra 9, 10, 11: rear (pixel 5, row 0), rear (pixel 7, row 25), front detector
    detector_ids = np.array([1100005, 1201007, 2100000])
    with h5py.File(path, 'w') as f:
        raw = f.create_group('raw_data_1')
        raw['beamline'] = np.array([b'SANS2D'])
        raw['title'] = np.array([b'test run'])
        raw['duration'] = np.array([10.0], dtype=np.float32)
        raw['detector_1/spectrum_index'] = np.array([9, 10, 11], dtype=np.int32)
        raw['isis_vms_compat/UDET'] = np.concatenate([np.arange(1, 9), detector_ids]).astype(np.int32)
        raw['detector_1_events/event_id'] = np.asarray(event_spectra, dtype=np.uint32)
        raw['detector_1_events/event_time_offset'] = np.asarray(event_tof_us, dtype=np.float32)


def test_pixel_row_of_the_detector_ids():
    pixel, row = sans2d_rear_pixel_row([1100000, 1101511, 1102000, 1523511])
    assert list(pixel) == [0, 511, 0, 511]
    assert list(row) == [0, 1, 2, 119]  # left tube 0, right tube 0, left tube 1, last right tube


def test_events_are_binned_per_rear_pixel_in_seconds(tmp_path):
    pytest.importorskip('scipp')
    path = tmp_path / 'raw.nxs'
    _write_raw_with_events(path, [10, 9, 11, 10], [1000.0, 2000.0, 3000.0, 4000.0])
    da = read_sans2d_events(path)
    per_pixel = da.bins.size().values
    assert per_pixel.sum() == 3  # the front-detector event is dropped
    assert per_pixel[7 * 120 + 25] == 2 and per_pixel[5 * 120 + 0] == 1
    tof = np.sort(da.bins.constituents['data'].coords['tof'].values)
    assert tof == pytest.approx([1e-3, 2e-3, 4e-3])


def test_mg_plot_reduces_measured_events_like_simulated_ones(tmp_path):
    pytest.importorskip('scipp')
    pytest.importorskip('scippneutron')
    from mcstas_gisans.plot_cli import create_argparser
    from mcstas_gisans.plot import get_datasets
    path = tmp_path / 'raw.nxs'
    # two rear events at 50 ms (~6.3 AA over ~31.3 m), one at 70 ms (~8.9 AA), one front-detector event
    _write_raw_with_events(path, [9, 10, 10, 11], [50000.0, 50000.0, 70000.0, 50000.0])
    args = create_argparser().parse_args(['--nxs', str(path), '--instrument', 'sans2d', '--alpha', '0.3',
                                          '--sample_orientation', '1', '--wavelength_slice', '6.0', '7.0'])
    hist, _, _, _, _ = get_datasets(args)[0]
    assert hist.sum() == pytest.approx(2.0)
    args = create_argparser().parse_args(['--nxs', str(path), '--instrument', 'sans2d', '--alpha', '0.3',
                                          '--sample_orientation', '1'])
    assert get_datasets(args)[0][0].sum() == pytest.approx(3.0)


def _tof_dataset(path, extra):
    from mcstas_gisans.plot_cli import create_argparser
    from mcstas_gisans.plot import get_datasets
    args = create_argparser().parse_args(['--nxs', str(path), '--instrument', 'sans2d', '--alpha', '0.3',
                                          '--sample_orientation', '1'] + extra)
    return get_datasets(args)[0]


def test_tof_q_bins_do_not_depend_on_the_plot_range(tmp_path):
    # the plot range only sets the axis limits (as in Mantid / esssans, the Q bins are a separate choice): zooming
    # must not make the bins narrower than a pixel (combs of empty bins)
    pytest.importorskip('scipp')
    pytest.importorskip('scippneutron')
    path = tmp_path / 'raw.nxs'
    _write_raw_with_events(path, [9, 10, 10, 11], [50000.0, 50000.0, 70000.0, 50000.0])
    hist, _, y_edges, z_edges, _ = _tof_dataset(path, [])
    zoomed, _, y_zoomed, z_zoomed, _ = _tof_dataset(path, ['--y_plot_range', '-0.05', '0.05', '--z_plot_range', '0', '0.1'])
    np.testing.assert_array_equal(y_zoomed, y_edges)
    np.testing.assert_array_equal(z_zoomed, z_edges)
    np.testing.assert_array_equal(zoomed, hist)


def test_tof_q_bins_option_sets_the_bin_edges(tmp_path):
    pytest.importorskip('scipp')
    pytest.importorskip('scippneutron')
    path = tmp_path / 'raw.nxs'
    _write_raw_with_events(path, [9, 10, 10, 11], [50000.0, 50000.0, 70000.0, 50000.0])
    hist, _, y_edges, z_edges, _ = _tof_dataset(path, ['--qy_bins', '-5', '5', '40', '--qz_bins', '-5', '5', '25'])
    np.testing.assert_allclose(y_edges, np.linspace(-5, 5, 41))
    np.testing.assert_allclose(z_edges, np.linspace(-5, 5, 26))
    assert hist.shape == (40, 25)
    assert hist.sum() == pytest.approx(3.0)  # all rear events inside the bins


@pytest.mark.parametrize('bins', [['0.2', '-0.2', '10'], ['-0.2', '0.2', '0'], ['-0.2', '0.2', '2.5']])
def test_invalid_q_bins_are_rejected(monkeypatch, bins):
    from mcstas_gisans.plot_cli import create_argparser, parse_args
    monkeypatch.setattr('sys.argv', ['mg_plot', '--nxs', 'x.nxs', '--qy_bins'] + bins)
    with pytest.raises(SystemExit):
        parse_args(create_argparser())
