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
    # windowed centroid: pixels within 50 mm of the centroid, iterated from the brightest pixel
    X, Y = np.meshgrid(x_rel, y_rel, indexing='ij')
    i, j = np.unravel_index(np.argmax(image), image.shape)
    centroid = np.array([x_rel[i], y_rel[j]])
    for _ in range(50):
        w = image * (((X - centroid[0]) ** 2 + (Y - centroid[1]) ** 2) <= 0.05 ** 2)
        centroid = np.array([(w * X).sum(), (w * Y).sum()]) / w.sum()
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


def test_background_far_from_the_beam_does_not_bias_the_centroid(tmp_path):
    """Uniform background on a detector that is off-centre w.r.t. the beam must not pull the centroid."""
    with h5py.File(DIRECT_BEAM_FILE, 'r') as f:
        image = f['entry0/D22/Detector 1/data1'][:, :, 0].astype(float)
    noisy = image + np.random.default_rng(0).poisson(0.5, image.shape)  # ~16k background counts
    path = tmp_path / "noisy.nxs"
    with h5py.File(path, 'w') as f:
        f.create_dataset('entry0/D22/Detector 1/data1', data=noisy[:, :, None])
    clean = find_required_centre_offset(DIRECT_BEAM_FILE)
    np.testing.assert_allclose(find_required_centre_offset(str(path)), clean, atol=2e-4)
    assert abs(find_required_centre_offset(str(path), beam_radius=0)[0] - clean[0]) > 5e-3  # whole detector: biased


def test_find_required_centre_offset_file_not_found():
    with pytest.raises(FileNotFoundError):
        find_required_centre_offset("non_existent_file.nxs")


def test_simulated_direct_beam_matches_the_measurement(tmp_path):
    """The paper MCPL beam ray-traced with the found offset lands on the measured direct beam, and the
    intensity factor equals the hand calculation of examples/paper/README.md (120538 / 60 / 9639.83)."""
    from mcstas_gisans.beam_centre_correction import compare_with_simulated_direct_beam
    offset = find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=2)
    results = compare_with_simulated_direct_beam(
        DIRECT_BEAM_FILE, "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz", offset, sample_orientation=2,
        experiment_time=60, figure='png', savename=str(tmp_path / "check"))
    assert np.all(np.abs(results['residual_pixels']) < 0.2)
    assert results['intensity_factor'] == pytest.approx(120538 / 60 / 9639.83, rel=1e-3)
    assert results['mcpl_mean_wavelength'] == pytest.approx(6.0, rel=0.02)
    assert (tmp_path / "check.png").exists()


def test_incident_angle_is_measured_from_the_specular_spot():
    """073174 (orientation 2, nominal 0.24 deg): 2*alpha between the specular spot and the direct beam."""
    from mcstas_gisans.beam_centre_correction import measure_incident_angle
    alpha = measure_incident_angle("data/paper/d22_measurement/073174.nxs", DIRECT_BEAM_FILE, sample_orientation=2)
    assert alpha == pytest.approx(0.24, rel=0.05)
