"""
Physics-invariant tests for the detector geometry, gravity handling and Q calculation.

Unlike golden-value tests, these check properties that must hold for any correct
implementation, derived independently of the code under test:

* Gravity always pulls neutrons DOWN in the laboratory (NeXus) frame, whatever the
  sample orientation.
* A ray's raw detector pixel must be the pixel at its laboratory-frame intersection
  point (NeXus frame: x horizontal, y up, z along the beam; raw pixel index ix grows
  with +x, iy grows with +y).
* The detector offset is a property of the detector position only: the offset found
  from the same direct-beam measurement must not depend on the sample orientation,
  and changing the assumed wavelength must shift it by exactly the change of the
  gravity drop.
* The unscattered (direct) beam maps to Q = 0, and the specular reflection maps to
  Qz = 2 k sin(alpha), Qy = 0, for every orientation and wavelength (TOF and non-TOF).
"""
import copy

import h5py
import numpy as np
import pytest
import scipp as sc

from mcstas_gisans.beam_centre_correction import find_required_centre_offset
from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults

H = 6.62607015e-34
M_N = 1.67492749804e-27
G = 9.80665
ORIENTATIONS = [0, 1, 2]
DIRECT_BEAM_FILE = "data/paper/d22_measurement/073162.nxs"


# Known bugs at the time these tests were written (dev-scipp 4b46a00), see
# review_dev_scipp_findings_local.md: A1 = gravity counted twice in calculate_q_limits,
# A2 = orientation 0/2 pixel mapping rotated 180 degrees relative to CoordinateTransform.
# Strict xfail: once a bug is fixed the test XPASSes and fails the suite, forcing this
# list to be emptied together with the fix.
KNOWN_FAILURES = {
    'test_beam_centre_closed_loop[0-12.0]',
    'test_beam_centre_closed_loop[0-6.0]',
    'test_beam_centre_closed_loop[1-12.0]',
    'test_beam_centre_closed_loop[1-6.0]',
    'test_beam_centre_closed_loop[2-12.0]',
    'test_beam_centre_closed_loop[2-6.0]',
    'test_direct_beam_maps_to_q_zero_non_tof[0-12.0]',
    'test_direct_beam_maps_to_q_zero_non_tof[0-6.0]',
    'test_direct_beam_maps_to_q_zero_non_tof[1-12.0]',
    'test_direct_beam_maps_to_q_zero_non_tof[1-6.0]',
    'test_direct_beam_maps_to_q_zero_non_tof[2-12.0]',
    'test_direct_beam_maps_to_q_zero_non_tof[2-6.0]',
    'test_direct_beam_maps_to_q_zero_tof[0-12.0]',
    'test_direct_beam_maps_to_q_zero_tof[0-3.0]',
    'test_direct_beam_maps_to_q_zero_tof[0-6.0]',
    'test_direct_beam_maps_to_q_zero_tof[2-12.0]',
    'test_direct_beam_maps_to_q_zero_tof[2-3.0]',
    'test_direct_beam_maps_to_q_zero_tof[2-6.0]',
    'test_gravity_pulls_neutrons_down_in_lab_frame[0]',
    'test_gravity_pulls_neutrons_down_in_lab_frame[2]',
    'test_offset_is_independent_of_sample_orientation',
    'test_offset_wavelength_dependence_equals_gravity_drop[0]',
    'test_offset_wavelength_dependence_equals_gravity_drop[1]',
    'test_offset_wavelength_dependence_equals_gravity_drop[2]',
    'test_raw_pixel_matches_lab_frame_intersection[0-0.0]',
    'test_raw_pixel_matches_lab_frame_intersection[0-0.4]',
    'test_raw_pixel_matches_lab_frame_intersection[2-0.0]',
    'test_raw_pixel_matches_lab_frame_intersection[2-0.4]',
    'test_rotate_detector_image_matches_coordinate_transform[0]',
    'test_rotate_detector_image_matches_coordinate_transform[2]',
    'test_specular_maps_to_expected_qz_non_tof[0-12.0]',
    'test_specular_maps_to_expected_qz_non_tof[0-6.0]',
    'test_specular_maps_to_expected_qz_non_tof[1-12.0]',
    'test_specular_maps_to_expected_qz_non_tof[1-6.0]',
    'test_specular_maps_to_expected_qz_non_tof[2-12.0]',
    'test_specular_maps_to_expected_qz_non_tof[2-6.0]',
    'test_specular_maps_to_expected_qz_tof[0-12.0]',
    'test_specular_maps_to_expected_qz_tof[0-3.0]',
    'test_specular_maps_to_expected_qz_tof[0-6.0]',
    'test_specular_maps_to_expected_qz_tof[2-12.0]',
    'test_specular_maps_to_expected_qz_tof[2-3.0]',
    'test_specular_maps_to_expected_qz_tof[2-6.0]',
}


@pytest.fixture(autouse=True)
def _mark_known_failures(request):
    if request.node.name in KNOWN_FAILURES:
        request.applymarker(pytest.mark.xfail(strict=True, reason="known bug A1/A2 (see module comment)"))


def velocity(wavelength_aa):
    return H / (M_N * wavelength_aa * 1e-10)


def gravity_drop(wavelength_aa, distance):
    return 0.5 * G * (distance / velocity(wavelength_aa)) ** 2


def make_instrument(orientation, alpha=0.0, wavelength=6.0, offset=(0.0, 0.0), no_gravity=False, instrument='d22'):
    params = copy.deepcopy(instrument_defaults[instrument])
    params['detector']['resolution'] = [0.0, 0.0]
    params['detector']['direct_beam_centre_offset'] = list(offset)
    wavelength_selected = None if params['tof_instrument'] else wavelength
    return Instrument(params, alpha, wavelength_selected, orientation, no_gravity=no_gravity)


def lab_rays(wavelength, n=20000, divergence=1e-3, seed=0):
    """Unscattered beam along +z_nexus with a small Gaussian divergence (lab frame velocities)."""
    rng = np.random.default_rng(seed)
    v = velocity(wavelength)
    vx = rng.normal(0.0, divergence, n) * v
    vy = rng.normal(0.0, divergence, n) * v
    vz = np.full(n, v)
    return vx, vy, vz


def hit_raw_pixels(instrument, vx_lab, vy_lab, vz_lab):
    """Ray-trace lab-frame velocities (starting at the sample) to raw NeXus pixel indices."""
    vx, vy, vz = instrument.detector.coords.nexus_to_bornagain(vx_lab, vy_lab, vz_lab)
    zero = np.zeros_like(vx)
    ix, iy, valid, tof = instrument.calculate_pixel_hit(zero, zero, zero, zero, vx, vy, vz)
    return ix[valid], iy[valid], tof[valid]


def raw_image(instrument, ix, iy):
    det = instrument.detector
    img = np.zeros((det.pixels_x_nexus, det.pixels_y_nexus))
    np.add.at(img, (ix, iy), 1.0)
    return img


def mean_q_of_image(instrument, raw, wavelength):
    """Intensity-weighted mean (Qy, Qz) [1/nm] of a raw detector image, using the reduction's Q axes."""
    image = instrument.detector.coords.rotate_detector_image(raw)
    q_y, q_z = instrument.get_q_pixel_limits(wavelength)
    qy_c = 0.5 * (q_y[:-1] + q_y[1:])
    qz_c = 0.5 * (q_z[:-1] + q_z[1:])
    w = image / image.sum()
    return float((w.sum(axis=1) * qy_c).sum()), float((w.sum(axis=0) * qz_c).sum()), np.diff(q_y).mean(), np.diff(q_z).mean()


# ---------------------------------------------------------------------------
# Raw-frame geometry and gravity direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_gravity_pulls_neutrons_down_in_lab_frame(orientation):
    """A horizontal 12 Å neutron must land drop(λ) BELOW its no-gravity landing point in the raw image."""
    wavelength = 12.0
    offset = (0.0023, 0.0011)  # keep the beam away from pixel boundaries
    vx, vy, vz = (np.array([0.0]), np.array([0.0]), np.array([velocity(wavelength)]))
    ix_g, iy_g, _ = hit_raw_pixels(make_instrument(orientation, wavelength=wavelength, offset=offset), vx, vy, vz)
    ix_0, iy_0, _ = hit_raw_pixels(make_instrument(orientation, wavelength=wavelength, offset=offset, no_gravity=True), vx, vy, vz)

    instrument = make_instrument(orientation, wavelength=wavelength, offset=offset)
    drop = gravity_drop(wavelength, instrument.sample_detector_distance)
    ix_expected, iy_expected, _ = instrument.detector.get_pixel_indices_from_position(0.0, -drop)

    assert ix_g[0] == ix_0[0], "gravity must not move the beam horizontally"
    assert iy_g[0] < iy_0[0], "gravity must move the beam DOWN (towards lower raw iy)"
    assert (ix_g[0], iy_g[0]) == (ix_expected, iy_expected)


@pytest.mark.parametrize("alpha", [0.0, 0.4])
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_raw_pixel_matches_lab_frame_intersection(orientation, alpha):
    """Without gravity, the raw pixel of a ray is the pixel at its straight-line lab-frame intersection."""
    instrument = make_instrument(orientation, alpha=alpha, offset=(0.03, -0.02), no_gravity=True)
    rng = np.random.default_rng(1)
    n, v = 2000, velocity(6.0)
    vx = rng.uniform(-0.02, 0.02, n) * v
    vy = rng.uniform(-0.02, 0.02, n) * v
    vz = np.full(n, v)
    ix, iy, _ = hit_raw_pixels(instrument, vx, vy, vz)

    L = instrument.sample_detector_distance
    ix_exp, iy_exp, valid = instrument.detector.get_pixel_indices_from_position(L * vx / vz, L * vy / vz)
    assert valid.all()
    np.testing.assert_array_equal(ix, ix_exp)
    np.testing.assert_array_equal(iy, iy_exp)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_rotate_detector_image_matches_coordinate_transform(orientation):
    """rotate_detector_image must put a raw pixel at the BornAgain-frame index given by CoordinateTransform."""
    instrument = make_instrument(orientation, offset=(0.03, -0.02))
    det = instrument.detector
    rng = np.random.default_rng(2)
    for _ in range(50):
        ix = rng.integers(det.pixels_x_nexus)
        iy = rng.integers(det.pixels_y_nexus)
        x_lab = det.min_edge_x_nexus + (ix + 0.5) * det.pixel_size_x_nexus
        y_lab = det.min_edge_y_nexus + (iy + 0.5) * det.pixel_size_y_nexus
        y_ba, z_ba = det.coords.apply_sample_orientation_transform(x_lab, y_lab)
        iy_ba = int(np.floor((y_ba - det.min_edge_y_bornagain) / det.pixel_size_y_bornagain))
        iz_ba = int(np.floor((z_ba - det.min_edge_z_bornagain) / det.pixel_size_z_bornagain))
        raw = np.zeros((det.pixels_x_nexus, det.pixels_y_nexus))
        raw[ix, iy] = 1.0
        rotated = det.coords.rotate_detector_image(raw)
        assert np.unravel_index(np.argmax(rotated), rotated.shape) == (iy_ba, iz_ba)


# ---------------------------------------------------------------------------
# Unscattered beam -> Q = 0 ; specular -> Qz = 2 k sin(alpha)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wavelength", [6.0, 12.0])
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_direct_beam_maps_to_q_zero_non_tof(orientation, wavelength):
    """The simulated direct beam (gravity on) must be centred at Q = 0 on the reduction's Q axes."""
    instrument = make_instrument(orientation, wavelength=wavelength, offset=(0.05, -0.02))
    ix, iy, _ = hit_raw_pixels(instrument, *lab_rays(wavelength))
    qy, qz, dqy, dqz = mean_q_of_image(instrument, raw_image(instrument, ix, iy), wavelength)
    assert abs(qy) < 0.1 * dqy, f"direct beam at Qy={qy:.2e} (pixel {dqy:.2e})"
    assert abs(qz) < 0.1 * dqz, f"direct beam at Qz={qz:.2e} (pixel {dqz:.2e})"


@pytest.mark.parametrize("wavelength", [6.0, 12.0])
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_specular_maps_to_expected_qz_non_tof(orientation, wavelength):
    """A specularly reflected beam must appear at Qz = 2 k sin(alpha), Qy = 0 (k = 2π/λ)."""
    alpha = 0.5
    instrument = make_instrument(orientation, alpha=alpha, wavelength=wavelength, offset=(0.05, -0.02))
    a = np.deg2rad(alpha)
    rng = np.random.default_rng(3)
    n, v = 20000, velocity(wavelength)
    # launch directions in the BornAgain (sample) frame: specular = (cos a, 0, +sin a) with small spread
    # spread over several pixels so that pixel-centre quantisation does not bias the centroid
    dy = rng.normal(0, 1e-3, n)
    dz = rng.normal(0, 1e-3, n)
    vx_ba = np.full(n, np.cos(a)) * v
    vy_ba = dy * v
    vz_ba = (np.sin(a) + dz) * v
    zero = np.zeros(n)
    ix, iy, valid, _ = instrument.calculate_pixel_hit(zero, zero, zero, zero, vx_ba, vy_ba, vz_ba)
    qy, qz, dqy, dqz = mean_q_of_image(instrument, raw_image(instrument, ix[valid], iy[valid]), wavelength)
    qz_expected = 2 * (2 * np.pi / (wavelength * 0.1)) * np.sin(a)
    assert abs(qy) < 0.1 * dqy
    assert abs(qz - qz_expected) < 0.1 * dqz, f"specular at Qz={qz:.5f}, expected {qz_expected:.5f} (pixel {dqz:.2e})"


def _tof_event_q(orientation, wavelength, alpha=0.0, launch_ba=None):
    """Simulate TOF events for one wavelength on a TOF instrument and return mean (Qy, Qz) [1/Å] from compute_q_scipp."""
    instrument = make_instrument(orientation, alpha=alpha, offset=(0.05, -0.02), instrument='skadi')
    if launch_ba is None:
        vx, vy, vz = lab_rays(wavelength, n=5000)
        vx, vy, vz = instrument.detector.coords.nexus_to_bornagain(vx, vy, vz)
    else:
        vx, vy, vz = launch_ba
    zero = np.zeros_like(vx)
    ix, iy, valid, sd_tof = instrument.calculate_pixel_hit(zero, zero, zero, zero, vx, vy, vz)
    ix, iy, sd_tof = ix[valid], iy[valid], sd_tof[valid]
    det = instrument.detector
    positions = det.get_pixel_positions(instrument.sample_detector_distance)[ix * det.pixels_y_nexus + iy]
    tof = instrument.nominal_source_sample_distance / velocity(wavelength) + sd_tof
    da = sc.DataArray(
        data=sc.ones(dims=['event'], shape=[len(tof)], unit='counts'),
        coords={'position': sc.vectors(dims=['event'], values=positions, unit='m'),
                'tof': sc.array(dims=['event'], values=tof, unit='s')})
    da = instrument.compute_q_scipp(da)
    k = 2 * np.pi / wavelength
    # TOF positions are pixel centres: use the larger pixel dimension as the Q resolution scale
    pixel_q = k * max(det.pixel_size_x_nexus, det.pixel_size_y_nexus) / instrument.sample_detector_distance
    return (float(sc.mean(da.coords['Qy']).value), float(sc.mean(da.coords['Qz']).value), pixel_q,
            float(sc.mean(da.coords['wavelength']).value))


@pytest.mark.parametrize("wavelength", [3.0, 6.0, 12.0])
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_direct_beam_maps_to_q_zero_tof(orientation, wavelength):
    """TOF: the unscattered beam must map to Q = 0 at every wavelength, and λ must be recovered from the TOF."""
    qy, qz, pixel_q, lam = _tof_event_q(orientation, wavelength)
    assert lam == pytest.approx(wavelength, rel=2e-3)
    assert abs(qy) < 0.1 * pixel_q
    assert abs(qz) < 0.1 * pixel_q


@pytest.mark.parametrize("wavelength", [3.0, 6.0, 12.0])
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_specular_maps_to_expected_qz_tof(orientation, wavelength):
    """TOF: specular reflection at Qz = 2 k sin(alpha), Qy = 0 for every wavelength and orientation."""
    alpha = 0.5
    a = np.deg2rad(alpha)
    rng = np.random.default_rng(4)
    n, v = 5000, velocity(wavelength)
    launch = (np.full(n, np.cos(a)) * v, rng.normal(0, 1e-3, n) * v, (np.sin(a) + rng.normal(0, 1e-3, n)) * v)
    qy, qz, pixel_q, _ = _tof_event_q(orientation, wavelength, alpha=alpha, launch_ba=launch)
    qz_expected = 2 * (2 * np.pi / wavelength) * np.sin(a)
    assert abs(qy) < 0.1 * pixel_q
    assert abs(qz - qz_expected) < 0.1 * pixel_q, f"specular at Qz={qz:.5f} 1/Å, expected {qz_expected:.5f}"


# ---------------------------------------------------------------------------
# Detector offset (beam-centre correction)
# ---------------------------------------------------------------------------

def _write_fake_d22_nexus(path, image):
    with h5py.File(path, 'w') as f:
        f.create_dataset('entry0/D22/Detector 1/data1', data=image[:, :, None].astype(np.int32))


@pytest.mark.parametrize("wavelength", [6.0, 12.0])
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_beam_centre_closed_loop(tmp_path, orientation, wavelength):
    """A simulated direct beam with a known detector offset must give back that offset."""
    true_offset = (0.12, -0.035)
    instrument = make_instrument(orientation, wavelength=wavelength, offset=true_offset)
    ix, iy, _ = hit_raw_pixels(instrument, *lab_rays(wavelength))
    path = tmp_path / "direct_beam.nxs"
    _write_fake_d22_nexus(path, raw_image(instrument, ix, iy))
    found = find_required_centre_offset(str(path), wavelength=wavelength, sample_orientation=orientation)
    np.testing.assert_allclose(found, true_offset, atol=3e-4)


def test_offset_is_independent_of_sample_orientation():
    """The detector offset from one direct-beam measurement must not depend on the sample orientation."""
    offsets = [find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=o) for o in ORIENTATIONS]
    for o, off in zip(ORIENTATIONS, offsets):
        np.testing.assert_allclose(off, offsets[1], atol=5e-5, err_msg=f"orientation {o}")


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_offset_wavelength_dependence_equals_gravity_drop(orientation):
    """Assuming a longer wavelength must move the vertical offset down by exactly the extra gravity drop."""
    L = instrument_defaults['d22']['sample_detector_distance']
    off6 = find_required_centre_offset(DIRECT_BEAM_FILE, wavelength=6.0, sample_orientation=orientation)
    off12 = find_required_centre_offset(DIRECT_BEAM_FILE, wavelength=12.0, sample_orientation=orientation)
    expected = -(gravity_drop(12.0, L) - gravity_drop(6.0, L))
    assert off12[0] == pytest.approx(off6[0], abs=5e-5)
    assert off12[1] - off6[1] == pytest.approx(expected, abs=5e-5)
