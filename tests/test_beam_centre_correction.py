"""
Tests for the beam_centre_correction module.

Expected values are derived independently from the raw NeXus image (intensity centroid)
and the analytic gravity drop / beam-angle geometry, not captured from the implementation.
"""
import os

import h5py
import numpy as np
import pytest

from mcstas_gisans.beam_centre_correction import find_required_centre_offset
from mcstas_gisans.instrument_defaults import instrument_defaults

DIRECT_BEAM_FILE = os.path.join("data", "paper", "d22_measurement", "073162.nxs")
H, M_N, G = 6.62607015e-34, 1.67492749804e-27, 9.80665


def _expected_offset(wavelength):
    """Detector centre relative to the undeflected beam axis: landing point (0, -drop) minus centroid."""
    with h5py.File(DIRECT_BEAM_FILE, 'r') as f:
        image = f['entry0/D22/Detector 1/data1'][:, :, 0].astype(float)
    nx, ny = image.shape
    size_x, size_y = instrument_defaults['d22']['detector']['size']
    x_rel = (np.arange(nx) + 0.5) * size_x / nx - size_x / 2
    y_rel = (np.arange(ny) + 0.5) * size_y / ny - size_y / 2
    centroid = np.array([(image.sum(axis=1) * x_rel).sum(), (image.sum(axis=0) * y_rel).sum()]) / image.sum()
    L = instrument_defaults['d22']['sample_detector_distance']
    drop = 0.5 * G * (L * M_N * wavelength * 1e-10 / H) ** 2
    return np.array([0.0, -drop]) - centroid


@pytest.mark.parametrize("orientation", [0, 1, 2])
def test_direct_beam_offset_matches_independent_calculation(orientation):
    offset = find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=orientation)
    assert offset.shape == (2,)
    np.testing.assert_allclose(offset, _expected_offset(6.0), atol=1e-7)
    # D22 file 073162: detector translated sideways by ~300 mm (entry0/D22/Detector 1/dtr1_actual = 300.06 mm)
    assert offset[0] == pytest.approx(0.290, abs=0.005)


@pytest.mark.parametrize("orientation, axis, sign", [(0, 0, -1), (1, 1, +1), (2, 0, +1)])
def test_beam_angle_shifts_offset_geometrically(orientation, axis, sign):
    """A beam tilted by b towards the sample normal lands L*tan(b) further along the normal direction."""
    b = 0.44
    L = instrument_defaults['d22']['sample_detector_distance']
    base = find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=orientation)
    tilted = find_required_centre_offset(DIRECT_BEAM_FILE, beam_angle=b, sample_orientation=orientation)
    shift = tilted - base
    assert shift[axis] == pytest.approx(sign * L * np.tan(np.deg2rad(b)), rel=1e-4)
    # the other axis only changes through the (second-order) change of the flight time
    assert abs(shift[1 - axis]) < 1e-5


def test_beam_angle_override_does_not_leak_into_instrument_defaults():
    before = dict(instrument_defaults['d22'])
    find_required_centre_offset(DIRECT_BEAM_FILE, beam_angle=0.44)
    assert instrument_defaults['d22'] == before
    np.testing.assert_allclose(find_required_centre_offset(DIRECT_BEAM_FILE), _expected_offset(6.0), atol=1e-7)


def test_find_required_centre_offset_file_not_found():
    with pytest.raises(FileNotFoundError):
        find_required_centre_offset("non_existent_file.nxs")
