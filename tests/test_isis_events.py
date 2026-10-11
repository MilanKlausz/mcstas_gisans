import h5py
import numpy as np
import pytest

from mcstas_gisans.isis_events import events_to_datagroup, read_sans2d_events
from mcstas_gisans.nexus_reader import sans2d_rear_pixel_row


def _write_raw_with_events(path):
    # spectra 9, 10, 11: two rear-detector pixels and one front-detector pixel
    detector_ids = np.array([1100005, 1201007, 2100000])  # (pixel 5, row 0), (pixel 7, row 25), front
    with h5py.File(path, 'w') as f:
        raw = f.create_group('raw_data_1')
        raw['title'] = np.array([b'test run'])
        raw['duration'] = np.array([10.0], dtype=np.float32)
        raw['detector_1/spectrum_index'] = np.array([9, 10, 11], dtype=np.int32)
        raw['isis_vms_compat/UDET'] = np.concatenate([np.arange(1, 9), detector_ids]).astype(np.int32)
        raw['detector_1_events/event_id'] = np.array([10, 9, 11, 10], dtype=np.uint32)
        raw['detector_1_events/event_time_offset'] = np.array([1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float32)


def test_pixel_row_of_the_detector_ids():
    pixel, row = sans2d_rear_pixel_row([1100000, 1101511, 1102000, 1523511])
    assert list(pixel) == [0, 511, 0, 511]
    assert list(row) == [0, 1, 2, 119]  # left tube 0, right tube 0, left tube 1, last right tube


def test_events_are_mapped_to_rear_pixels_and_seconds(tmp_path):
    path = tmp_path / 'raw.nxs'
    _write_raw_with_events(path)
    ev = read_sans2d_events(path)
    assert ev['n_events'] == 4
    assert list(ev['pixel']) == [7, 5, 7]          # the front-detector event is dropped
    assert list(ev['row']) == [25, 0, 25]
    assert ev['tof'] == pytest.approx([1e-3, 2e-3, 4e-3])
    assert ev['title'] == 'test run'


def test_events_to_datagroup_bins_by_detector_id(tmp_path):
    sc = pytest.importorskip('scipp')
    path = tmp_path / 'raw.nxs'
    _write_raw_with_events(path)
    positions = sc.vectors(dims=['detector_id'], values=np.zeros((512 * 120, 3)), unit='m')
    template = sc.DataGroup({'data': sc.DataArray(sc.zeros(dims=['detector_id'], shape=[512 * 120]),
                                                  coords={'position': positions}),
                             'instrument': sc.DataGroup({'name': sc.scalar('sans2d')})})
    dg = events_to_datagroup(read_sans2d_events(path), template)
    per_pixel = dg['data'].bins.size().values
    assert per_pixel.sum() == 3
    assert per_pixel[7 * 120 + 25] == 2 and per_pixel[5 * 120 + 0] == 1
    assert dg['instrument']['name'].value == 'sans2d'
