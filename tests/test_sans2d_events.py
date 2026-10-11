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
