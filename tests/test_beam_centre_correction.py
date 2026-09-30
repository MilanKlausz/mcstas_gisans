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
WINDOW = (0.03, 0.06)  # explicit direct-beam window half-sizes [m] (x, y) for the independent calculation
H, M_N, G = 6.62607015e-34, 1.67492749804e-27, 9.80665


def _expected_offset(wavelength, window=WINDOW):
    """Detector centre relative to the undeflected beam axis: landing point (0, -drop) minus centroid."""
    with h5py.File(DIRECT_BEAM_FILE, 'r') as f:
        image = f['entry0/D22/Detector 1/data1'][:, :, 0].astype(float)
    nx, ny = image.shape
    size_x, size_y = instrument_defaults['d22']['detector']['size']
    x_rel = (np.arange(nx) + 0.5) * size_x / nx - size_x / 2
    y_rel = (np.arange(ny) + 0.5) * size_y / ny - size_y / 2
    # windowed centroid: pixels in the rectangle |x - cx| <= window[0], |y - cy| <= window[1], iterated from the brightest pixel
    X, Y = np.meshgrid(x_rel, y_rel, indexing='ij')
    i, j = np.unravel_index(np.argmax(image), image.shape)
    centroid = np.array([x_rel[i], y_rel[j]])
    for _ in range(50):
        w = image * ((np.abs(X - centroid[0]) <= window[0]) & (np.abs(Y - centroid[1]) <= window[1]))
        centroid = np.array([(w * X).sum(), (w * Y).sum()]) / w.sum()
    L = instrument_defaults['d22']['sample_detector_distance']
    drop = 0.5 * G * (L * M_N * wavelength * 1e-10 / H) ** 2
    return np.array([0.0, -drop]) - centroid


@pytest.mark.parametrize("orientation", [0, 1, 2])
def test_direct_beam_offset_matches_independent_calculation(orientation):
    offset = find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=orientation, beam_window=WINDOW)
    assert offset.shape == (2,)
    np.testing.assert_allclose(offset, _expected_offset(6.0), atol=1e-7)
    # the automatic window (3 RMS widths) holds the whole spot: the same centre within 0.02 pixel
    np.testing.assert_allclose(find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=orientation), offset, atol=1e-4)
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
    np.testing.assert_allclose(find_required_centre_offset(DIRECT_BEAM_FILE, beam_window=WINDOW), _expected_offset(6.0), atol=1e-7)


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
    assert abs(find_required_centre_offset(str(path), beam_window=(1.0, 1.0))[0] - clean[0]) > 5e-3  # whole detector: biased


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


def test_incident_angle_uses_the_whole_flat_topped_specular(tmp_path):
    """A wide beam gives a flat-topped specular spot, here 5 rows tall and 20 tubes wide with its brightest
    row at the lower edge: its position is the centroid of the whole spot, not of the brightest pixels."""
    from mcstas_gisans.beam_centre_correction import measure_incident_angle
    nx, ny = 128, 256
    size_x, size_y = instrument_defaults['d22']['detector']['size']
    y_rel = (np.arange(ny) + 0.5) * size_y / ny - size_y / 2
    direct, sample = np.zeros((nx, ny)), np.zeros((nx, ny))
    direct[55:75, 100:105] = 1000.0
    sample[55:75, 100:105] = 50.0                   # attenuated direct beam below the sample horizon
    rows = np.arange(160, 165)
    heights = np.array([1.05, 1.0, 1.0, 1.0, 1.0])  # the brightest row is the lowest one
    sample[55:75, rows] = 1000.0 * heights
    paths = []
    for name, image in (("direct.nxs", direct), ("sample.nxs", sample)):
        paths.append(tmp_path / name)
        with h5py.File(paths[-1], 'w') as f:
            f.create_dataset('entry0/D22/Detector 1/data1', data=image[:, :, None])
    s = (heights * y_rel[rows]).sum() / heights.sum() - y_rel[100:105].mean()
    expected = np.rad2deg(0.5 * np.arctan(s / instrument_defaults['d22']['sample_detector_distance']))
    alpha = measure_incident_angle(str(paths[1]), str(paths[0]), sample_orientation=1)
    assert alpha == pytest.approx(expected, abs=1e-5)
    narrow = np.rad2deg(0.5 * np.arctan((y_rel[160:163].mean() - y_rel[100:105].mean()) / instrument_defaults['d22']['sample_detector_distance']))
    assert abs(narrow - expected) > 1e-3  # a window around the brightest row would be biased low


def test_incident_angle_figure_shows_the_found_positions(tmp_path):
    from mcstas_gisans.beam_centre_correction import measure_incident_angle
    measure_incident_angle("data/paper/d22_measurement/073174.nxs", DIRECT_BEAM_FILE, sample_orientation=2, figure='png',
                           savename=str(tmp_path / "angle"))
    assert (tmp_path / "angle.png").exists()


def _synthetic_detector():
    from mcstas_gisans.beam_centre_correction import _instrument
    return _instrument('d22', [0.0, 0.0], None, 6.0, 1).detector


def test_automatic_window_holds_a_rectangular_beam_and_ignores_a_hot_pixel():
    """A flat rectangular beam (20 x 10 pixels) on background, with a single hot pixel far away that is brighter than
    the whole beam: the search starts at the maximum of the 3x3-median-filtered image, and the window of 3 RMS widths holds the whole beam."""
    from mcstas_gisans.beam_centre_correction import beam_spot, _pixel_coordinates
    det = _synthetic_detector()
    image = np.random.default_rng(0).poisson(0.2, (det.pixels_x_nexus, det.pixels_y_nexus)).astype(float)
    image[40:60, 100:110] += 100.0
    image[5, 20] = 5000.0
    spot = beam_spot(image, det)
    x_rel, y_rel = _pixel_coordinates(det)
    np.testing.assert_allclose(spot['centre'], [x_rel[40:60].mean(), y_rel[100:110].mean()], atol=0.2 * det.pixel_size_y_nexus)
    beam_half = np.array([10 * det.pixel_size_x_nexus, 5 * det.pixel_size_y_nexus])
    assert np.all(spot['half'] > beam_half) and np.all(spot['half'] < 2 * beam_half)  # 3/sqrt(12) = 0.87 of the full width


def test_a_beam_cut_by_the_detector_edge_is_reported(capsys):
    from mcstas_gisans.beam_centre_correction import beam_spot
    det = _synthetic_detector()
    image = np.zeros((det.pixels_x_nexus, det.pixels_y_nexus))
    image[0:6, 100:110] = 100.0  # the beam runs off the detector at ix = 0
    beam_spot(image, det)
    assert "reaches the edge of the detector" in capsys.readouterr().out
    image = np.zeros_like(image)
    image[40:60, 100:110] = 100.0
    beam_spot(image, det)
    assert "reaches the edge" not in capsys.readouterr().out


def test_incident_angle_search_uses_the_expected_position(tmp_path):
    """With alpha given, a brighter feature far from the expected specular position is not taken for the specular."""
    from mcstas_gisans.beam_centre_correction import measure_incident_angle, _pixel_coordinates
    det = _synthetic_detector()
    nx, ny = det.pixels_x_nexus, det.pixels_y_nexus
    _, y_rel = _pixel_coordinates(det)
    direct, sample = np.zeros((nx, ny)), np.zeros((nx, ny))
    direct[55:75, 100:105] = 1000.0
    sample[55:75, 160:165] = 1000.0            # specular
    sample[55:75, 230:235] = 5000.0            # brighter feature much higher up
    paths = []
    for name, image in (("direct.nxs", direct), ("sample.nxs", sample)):
        paths.append(tmp_path / name)
        with h5py.File(paths[-1], 'w') as f:
            f.create_dataset('entry0/D22/Detector 1/data1', data=image[:, :, None])
    L = instrument_defaults['d22']['sample_detector_distance']
    s = y_rel[160:165].mean() - y_rel[100:105].mean()
    alpha = np.rad2deg(0.5 * np.arctan(s / L))
    assert measure_incident_angle(str(paths[1]), str(paths[0]), sample_orientation=1, alpha=round(alpha, 2)) == pytest.approx(alpha, abs=1e-5)
    assert measure_incident_angle(str(paths[1]), str(paths[0]), sample_orientation=1) > alpha + 0.1  # without alpha: the brighter one
