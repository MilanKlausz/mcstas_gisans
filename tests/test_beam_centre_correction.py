"""
Tests for beam_centre_correction module
"""
import copy
import os
import h5py
import numpy as np
import pytest
from mcstas_gisans.beam_centre_correction import find_required_centre_offset
from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults
from mcstas_gisans.read_d22 import read_nexus_data

DIRECT_BEAM_FILE = os.path.join("data", "paper", "d22_measurement", "073162.nxs")

def _windowed_q_centre(filepath, offset, sample_orientation, window_pixels):
  """Q centroid of the pixels in window_pixels (the independent check of the offset condition)."""
  params = copy.deepcopy(instrument_defaults['d22'])
  params['detector']['direct_beam_centre_offset'] = list(offset)
  hist, _, _, _ = read_nexus_data(filepath, 0.0, 6.0, sample_orientation=sample_orientation)
  q_y, q_z = Instrument(params, 0.0, 6.0, sample_orientation=sample_orientation).get_q_pixel_limits()
  w = np.where(window_pixels, hist, 0.0)
  return np.array([(w.sum(1) * 0.5 * (q_y[1:] + q_y[:-1])).sum(), (w.sum(0) * 0.5 * (q_z[1:] + q_z[:-1])).sum()]) / w.sum()

@pytest.mark.parametrize("sample_orientation", [1, 2])
def test_offset_puts_the_windowed_direct_beam_at_q_zero(sample_orientation):
  from mcstas_gisans.beam_centre_correction import beam_spot, _instrument
  offset = find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=sample_orientation)
  assert isinstance(offset, np.ndarray) and offset.shape == (2,)
  hist, _, _, _ = read_nexus_data(DIRECT_BEAM_FILE, 0.0, 6.0, sample_orientation=sample_orientation)
  spot = beam_spot(hist, _instrument('d22', offset, None, 6.0, sample_orientation).detector)
  np.testing.assert_allclose(_windowed_q_centre(DIRECT_BEAM_FILE, offset, sample_orientation, spot['inside']), [0.0, 0.0], atol=1e-8)
  # D22 file 073162: detector translated sideways by ~300 mm (entry0/D22/Detector 1/dtr1_actual = 300.06 mm)
  assert abs(offset[0]) == pytest.approx(0.290, abs=0.005)

def test_automatic_and_explicit_windows_agree():
  """An explicit window a bit larger than the automatic one gives the same offset (the whole spot is inside both)."""
  auto = find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=2)
  explicit = find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=2, beam_window=(0.012, 0.035))
  np.testing.assert_allclose(auto, explicit, atol=1e-4)

def test_background_far_from_the_beam_does_not_bias_the_centroid(tmp_path):
  with h5py.File(DIRECT_BEAM_FILE, 'r') as f:
    image = f['entry0/D22/Detector 1/data1'][:, :, 0].astype(float)
  noisy = image + np.random.default_rng(0).poisson(0.5, image.shape)  # ~16k background counts
  path = tmp_path / "noisy.nxs"
  with h5py.File(path, 'w') as f:
    f.create_dataset('entry0/D22/Detector 1/data1', data=noisy[:, :, None])
  clean = find_required_centre_offset(DIRECT_BEAM_FILE)
  np.testing.assert_allclose(find_required_centre_offset(str(path)), clean, atol=2e-4)
  assert np.max(np.abs(find_required_centre_offset(str(path), beam_window=(1.0, 1.0)) - clean)) > 5e-3  # whole detector: biased

def test_find_required_centre_offset_declination_override():
  """A beam declination moves the expected landing point, and so the offset, by about L*tan(declination) (vertically)."""
  base = find_required_centre_offset(DIRECT_BEAM_FILE)
  tilted = find_required_centre_offset(DIRECT_BEAM_FILE, beam_declination_angle=0.44)
  L = instrument_defaults['d22']['sample_detector_distance']
  assert abs(tilted[1] - base[1]) == pytest.approx(L * np.tan(np.radians(0.44)), rel=0.02)
  assert tilted[0] == pytest.approx(base[0], abs=1e-3)

def _detector():
  from mcstas_gisans.beam_centre_correction import _instrument
  return _instrument('d22', None, None, 6.0, 1).detector

def test_automatic_window_holds_a_rectangular_beam_and_ignores_a_hot_pixel():
  """A flat rectangular beam (20 x 10 pixels) on background, with a single hot pixel far away that has more counts
  than the whole beam: the search starts at the maximum of the 3x3-median-filtered image, and the window of 3 RMS
  widths holds the whole beam."""
  from mcstas_gisans.beam_centre_correction import beam_spot, _pixel_centres
  det = _detector()
  image = np.random.default_rng(0).poisson(0.2, (det.pixels_y_bornagain, det.pixels_z_bornagain)).astype(float)
  image[40:60, 100:110] += 100.0
  image[5, 20] = 50000.0
  spot = beam_spot(image, det)
  y_c, z_c = _pixel_centres(det)
  np.testing.assert_allclose(spot['centre'], [y_c[40:60].mean(), z_c[100:110].mean()], atol=0.2 * det.pixel_size_z_bornagain)
  beam_half = np.array([10 * det.pixel_size_y_bornagain, 5 * det.pixel_size_z_bornagain])
  assert np.all(spot['half'] > beam_half) and np.all(spot['half'] < 2 * beam_half)  # 3/sqrt(12) = 0.87 of the full width

def test_a_beam_cut_by_the_detector_edge_is_reported(capsys):
  from mcstas_gisans.beam_centre_correction import beam_spot
  det = _detector()
  image = np.zeros((det.pixels_y_bornagain, det.pixels_z_bornagain))
  image[0:6, 100:110] = 100.0  # the beam runs off the detector
  beam_spot(image, det)
  assert "reaches the edge of the detector" in capsys.readouterr().out
  image = np.zeros_like(image)
  image[40:60, 100:110] = 100.0
  beam_spot(image, det)
  assert "reaches the edge" not in capsys.readouterr().out

def _write(path, image):
  with h5py.File(path, 'w') as f:
    f.create_dataset('entry0/D22/Detector 1/data1', data=image[:, :, None])

def test_incident_angle_uses_the_whole_flat_topped_specular(tmp_path):
  """A flat-topped specular, 5 rows tall and 20 tubes wide, with its brightest row at the lower edge: its position is
  the centroid of the whole spot, not of the brightest pixels (horizontal sample: the image is the raw one)."""
  from mcstas_gisans.beam_centre_correction import measure_incident_angle, _pixel_centres
  det = _detector()
  _, z_c = _pixel_centres(det)
  direct, sample = np.zeros((det.pixels_y_bornagain, det.pixels_z_bornagain)), np.zeros((det.pixels_y_bornagain, det.pixels_z_bornagain))
  direct[55:75, 100:105] = 1000.0
  sample[55:75, 100:105] = 50.0                   # attenuated direct beam below the sample horizon
  heights = np.array([1.05, 1.0, 1.0, 1.0, 1.0])  # the brightest row is the lowest one
  sample[55:75, 160:165] = 1000.0 * heights
  _write(tmp_path / "direct.nxs", direct); _write(tmp_path / "sample.nxs", sample)
  L = instrument_defaults['d22']['sample_detector_distance']
  s = (heights * z_c[160:165]).sum() / heights.sum() - z_c[100:105].mean()
  expected = np.rad2deg(0.5 * np.arctan(s / L))
  alpha = measure_incident_angle(str(tmp_path / "sample.nxs"), str(tmp_path / "direct.nxs"), sample_orientation=1)
  assert alpha == pytest.approx(expected, abs=1e-5)
  narrow = np.rad2deg(0.5 * np.arctan((z_c[160:163].mean() - z_c[100:105].mean()) / L))
  assert abs(narrow - expected) > 1e-3  # a window around the brightest row would be biased low

def test_incident_angle_search_uses_the_expected_position(tmp_path):
  """With alpha given, a brighter feature far from the expected specular position is not taken for the specular."""
  from mcstas_gisans.beam_centre_correction import measure_incident_angle, _pixel_centres
  det = _detector()
  _, z_c = _pixel_centres(det)
  direct, sample = np.zeros((det.pixels_y_bornagain, det.pixels_z_bornagain)), np.zeros((det.pixels_y_bornagain, det.pixels_z_bornagain))
  direct[55:75, 100:105] = 1000.0
  sample[55:75, 160:165] = 1000.0            # specular
  sample[55:75, 230:235] = 5000.0            # brighter feature much higher up
  _write(tmp_path / "direct.nxs", direct); _write(tmp_path / "sample.nxs", sample)
  L = instrument_defaults['d22']['sample_detector_distance']
  alpha = np.rad2deg(0.5 * np.arctan((z_c[160:165].mean() - z_c[100:105].mean()) / L))
  args = (str(tmp_path / "sample.nxs"), str(tmp_path / "direct.nxs"))
  assert measure_incident_angle(*args, sample_orientation=1, alpha=round(alpha, 2)) == pytest.approx(alpha, abs=1e-5)
  assert measure_incident_angle(*args, sample_orientation=1) > alpha + 0.1  # without alpha: the brighter one

def test_incident_angle_of_the_paper_data_and_figure(tmp_path):
  """073174 (vertical sample, nominal 0.24 deg): the specular lies 144.5 mm from the direct beam, 0.2353 deg."""
  from mcstas_gisans.beam_centre_correction import measure_incident_angle
  alpha = measure_incident_angle("data/paper/d22_measurement/073174.nxs", DIRECT_BEAM_FILE, sample_orientation=2, alpha=0.24,
                                 figure='png', savename=str(tmp_path / "angle"))
  assert alpha == pytest.approx(0.2353, abs=2e-4)
  assert (tmp_path / "angle.png").exists()

def test_simulated_direct_beam_matches_the_measurement(tmp_path):
  """The paper MCPL beam sent to the detector with the found offset lands on the measured direct beam, and the
  intensity factor equals the hand calculation of the paper example (120538 / 60 / 9639.83)."""
  from mcstas_gisans.beam_centre_correction import compare_with_simulated_direct_beam
  offset = find_required_centre_offset(DIRECT_BEAM_FILE, sample_orientation=2)
  results = compare_with_simulated_direct_beam(DIRECT_BEAM_FILE, "tests/data/d22_1e8/test_events.mcpl.gz", offset,
                                               sample_orientation=2, experiment_time=60, figure='png', savename=str(tmp_path / "check"))
  assert np.all(np.abs(results['residual_pixels']) < 0.2)
  assert results['intensity_factor'] == pytest.approx(120538 / 60 / 9639.83, rel=1e-3)
  assert (tmp_path / "check.png").exists()
