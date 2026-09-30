"""
Module to calculate the required detector centre_offset for NeXus measurements
so that the beam centre aligns with (qy, qz) = (0, 0) in Q-space, to cross-check it
with a simulated direct beam, and to measure the real incident angle from the
specular spot of a sample measurement.

The detector images are used in the sample frame (as read by read_nexus_data): the
horizontal axis is BornAgain y (along the sample surface), the vertical axis is
BornAgain z (the sample normal), for every sample orientation.
"""

import sys
import copy
import numpy as np
from scipy.optimize import root

from .read_d22 import read_nexus_data
from .instrument import Instrument
from .instrument_defaults import instrument_defaults

BEAM_WINDOW_WIDTHS = 3.0  # automatic direct-beam window: half-sizes of this many RMS widths of the beam, per axis
SPECULAR_ALONG_FACTOR = 1.5  # automatic specular window: the direct-beam half-size across the normal, this times it along
INITIAL_WINDOW_PIXELS = 5  # half-size [pixels] of the first window around the starting point
EDGE_WARNING_FRACTION = 0.01  # warn if this fraction of a window's counts lies in detector pixels at the window's cut edge

def _instrument(instrument_name, offset=None, beam_declination_angle=None, wavelength=6.0, sample_orientation=1, alpha=0.0):
  params = copy.deepcopy(instrument_defaults[instrument_name])
  if offset is not None:
    params['detector']['direct_beam_centre_offset'] = list(offset)
  if beam_declination_angle is not None:
    params['beam_declination_angle'] = beam_declination_angle
  return Instrument(params, alpha, wavelength, sample_orientation=sample_orientation)

def _pixel_centres(detector):
  """Pixel centres [m] of the sample-frame image along BornAgain y (horizontal) and z (vertical, the sample
  normal), relative to the detector centre."""
  y = (np.arange(detector.pixels_y_bornagain) + 0.5) * detector.pixel_size_y_bornagain - 0.5 * detector.size_y_bornagain
  z = (np.arange(detector.pixels_z_bornagain) + 0.5) * detector.pixel_size_z_bornagain - 0.5 * detector.size_z_bornagain
  return y, z

def _image_axes_window(window_nexus, sample_orientation):
  """--beam_window is given along the NeXus x and y axes of the detector; the sample-frame image of a vertical
  sample has these axes swapped."""
  if window_nexus is None:
    return None
  half = np.asarray(window_nexus, dtype=float)
  return half[::-1].copy() if sample_orientation in (0, 2) else half

def _nexus_axes_window(half, sample_orientation):
  return half[::-1] if sample_orientation in (0, 2) else half

def _smoothed_maximum(image, allowed=None):
  """Pixel of the maximum of the 3x3-median-filtered image, optionally among allowed pixels: an isolated hot
  pixel is removed by the median, however many counts it has, while an extended spot keeps its level."""
  padded = np.pad(image, 1)
  neighbours = np.stack([padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dy:padded.shape[1] - 1 + dy]
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
  filtered = np.median(neighbours, axis=0)
  if allowed is not None:
    filtered = np.where(allowed, filtered, -np.inf)
  return np.unravel_index(np.argmax(filtered), filtered.shape)

def _window_centroid(image, U, V, centre, half, max_iterations=100):
  """Intensity centroid and RMS width in the rectangle |U - cu| <= half[0], |V - cv| <= half[1], re-centred until it settles."""
  centre = np.asarray(centre, dtype=float)
  for _ in range(max_iterations):
    inside = (np.abs(U - centre[0]) <= half[0]) & (np.abs(V - centre[1]) <= half[1])
    w = np.where(inside, image, 0.0)
    total = w.sum()
    if total <= 0:
      raise ValueError("No counts in the centroid window.")
    new_centre = np.array([(w * U).sum(), (w * V).sum()]) / total
    converged = np.allclose(new_centre, centre, rtol=0, atol=1e-9)
    centre = new_centre
    if converged:
      break
  width = np.sqrt(np.array([(w * (U - centre[0]) ** 2).sum(), (w * (V - centre[1]) ** 2).sum()]) / total)
  return centre, width, inside

def _warn_if_cut(image, inside, label):
  """Warn if the window reaches the detector edge and a noticeable part of its counts lies in the edge pixels."""
  edge = np.zeros(image.shape, dtype=bool)
  edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
  total = image[inside].sum()
  if total > 0 and image[inside & edge].sum() > EDGE_WARNING_FRACTION * total:
    print(f"WARNING: the {label} reaches the edge of the detector ({100 * image[inside & edge].sum() / total:.1f}% of its "
          f"counts are in the outermost pixels): the spot may be cut off and its centre biased.")

def beam_spot(image, detector, window=None, label="direct beam"):
  """
  Centre, RMS width and window of the beam spot in a sample-frame detector image (m, relative to the
  detector centre, along BornAgain y and z). The search starts at the maximum of the 3x3-median-filtered
  image. The centroid is taken in a rectangle aligned with the detector axes and re-centred until it
  settles; by default (window=None) the half-sizes of the rectangle are BEAM_WINDOW_WIDTHS RMS widths of
  the spot along each axis, measured and updated together with the centre, starting from
  INITIAL_WINDOW_PIXELS pixels. This works for any beam and pixel size: a rectangular beam of full width W
  has an RMS width of W/sqrt(12), so 3 RMS widths give a half-size of 0.87 W. An explicit window
  [half_y, half_z] (m, image axes) is used as given. Background elsewhere on the detector does not bias
  the result. Returns a dict with 'centre', 'width', 'half' (window half-sizes) and 'inside' (mask).
  """
  y_c, z_c = _pixel_centres(detector)
  Y, Z = np.meshgrid(y_c, z_c, indexing='ij')
  i, j = _smoothed_maximum(image)
  centre = np.array([y_c[i], z_c[j]])
  pixel = np.array([detector.pixel_size_y_bornagain, detector.pixel_size_z_bornagain])
  if window is not None:
    half = np.asarray(window, dtype=float)
    centre, width, inside = _window_centroid(image, Y, Z, centre, half)
  else:
    half = INITIAL_WINDOW_PIXELS * pixel
    for _ in range(50):
      centre, width, inside = _window_centroid(image, Y, Z, centre, half)
      new_half = np.maximum(BEAM_WINDOW_WIDTHS * width, pixel)
      converged = np.allclose(new_half, half, rtol=1e-4, atol=0)
      half = new_half
      if converged:
        break
    centre, width, inside = _window_centroid(image, Y, Z, centre, half)
  _warn_if_cut(image, inside, f"window of the {label}")
  return {'centre': centre, 'width': width, 'half': half, 'inside': inside}

def _print_beam_window(spot, sample_orientation, given):
  half = _nexus_axes_window(spot['half'], sample_orientation)
  how = "as given" if given else f"{BEAM_WINDOW_WIDTHS:g} RMS widths of the beam"
  print(f"Direct-beam window: +-{half[0] * 1e3:.1f} mm (detector x) x +-{half[1] * 1e3:.1f} mm (detector y), {how}. "
        f"To set it: --beam_window {half[0]:.4f} {half[1]:.4f}")

def find_required_centre_offset(filepath, initial_guess=None, beam_declination_angle=None, wavelength=6.0, sample_orientation=1, instrument_name='d22', verbose=False, beam_window=None):
  """
  Find the required centre_offset values for the detector so that the beam centre
  measured in a NeXus file is positioned at (qy, qz) = (0, 0) in Q-space.

  Parameters
  ----------
  filepath : str
      Path to the NeXus file.
  initial_guess : list or np.ndarray, optional
      Initial guess for the centre_offset [x, y] in meters.
      If None, the default centre_offset for the selected instrument is used.
  beam_declination_angle : float, optional
      Override beam declination angle in degrees.
  wavelength : float, optional
      Wavelength in Angstroms (default: 6.0).
  sample_orientation : int, optional
      Sample orientation: 0 (vertical, beam from left), 1 (horizontal), 2 (vertical, beam from right). (default: 1).
  instrument_name : str, optional
      The name of the instrument key in instrument_defaults (default: 'd22').
  verbose : bool, optional
      If True, print detailed optimization progress.
  beam_window : sequence of 2 floats, optional
      Half-sizes [m] of the rectangular centroid window along the detector x and y (NeXus) axes
      (default: BEAM_WINDOW_WIDTHS RMS widths of the beam, see beam_spot). Only the pixels in this
      window enter the centroid, so background elsewhere on the detector does not bias it.

  Returns
  -------
  centre_offset : np.ndarray
      The calculated centre_offset [x, y] in meters.
  """
  original_declination = None
  if beam_declination_angle is not None and instrument_name in instrument_defaults:
    original_declination = instrument_defaults[instrument_name].get('beam_declination_angle')
    instrument_defaults[instrument_name]['beam_declination_angle'] = beam_declination_angle

  try:
    alpha_inc_deg = 0.0 #for direct beam measurements, the incident angle is 0 degrees
    hist, _, _, _ = read_nexus_data(filepath, alpha_inc_deg, wavelength, sample_orientation=sample_orientation)

    if initial_guess is None:
      initial_guess = instrument_defaults.get(instrument_name, {}).get('detector', {}).get('direct_beam_centre_offset', [0.0, 0.0])

    # the pixels of the beam spot (the window does not depend on the offset)
    geometry = _instrument(instrument_name, initial_guess, None, wavelength, sample_orientation)
    spot = beam_spot(hist, geometry.detector, _image_axes_window(beam_window, sample_orientation), label=f"direct beam of {filepath}")
    _print_beam_window(spot, sample_orientation, beam_window is not None)
    hist_spot = np.where(spot['inside'], hist, 0.0)

    if verbose:
      print(f"\n--- Starting Beam Centre Minimisation ---")
      print(f"Filepath: {filepath}")
      print(f"Initial Guess: {initial_guess}")

    def residual(direct_beam_centre_offset):
      # Copy defaults to avoid modifying global settings in place
      params = copy.deepcopy(instrument_defaults[instrument_name])
      params['detector']['direct_beam_centre_offset'] = list(direct_beam_centre_offset)

      instrument = Instrument(params, alpha_inc_deg, wavelength, sample_orientation=sample_orientation)
      q_y, q_z = instrument.get_q_pixel_limits()

      # Calculate bin centres
      y_centres = (q_y[:-1] + q_y[1:]) / 2.0
      z_centres = (q_z[:-1] + q_z[1:]) / 2.0

      # Calculate weight distributions of the beam spot
      y_intensity = np.sum(hist_spot, axis=1)
      z_intensity = np.sum(hist_spot, axis=0)
      total_intensity = np.sum(hist_spot)

      if total_intensity <= 0:
        raise ValueError("Total intensity of the NeXus dataset is zero or negative.")

      y_centre = np.sum(y_centres * y_intensity) / total_intensity
      z_centre = np.sum(z_centres * z_intensity) / total_intensity

      if verbose:
        print(f"  Eval offset: [{direct_beam_centre_offset[0]:.6f}, {direct_beam_centre_offset[1]:.6f}] -> Q-centre: ({y_centre:.6f}, {z_centre:.6f})")
      return np.array([y_centre, z_centre])

    res = root(residual, initial_guess)
    print(f"Optimization Success: {res.success}")
    print(f"Optimization Message: {res.message}")

    if not res.success:
      raise RuntimeError(f"Optimization failed to find required direct_beam_centre_offset: {res.message}")

    return res.x
  finally:
    if original_declination is not None:
      instrument_defaults[instrument_name]['beam_declination_angle'] = original_declination

def simulate_direct_beam(mcpl_path, offset, beam_declination_angle=None, wavelength=6.0, sample_orientation=1, instrument_name='d22', seed=0):
  """
  Send the neutrons of a McStas MCPL file (at the sample position) straight to the detector, as mg_run
  does for a direct-beam simulation (no sample, gravity and detector resolution included), and return
  the sample-frame detector image (sum of weights, i.e. rate per second) on the same pixel grid as the
  measured image, and some beam properties.
  """
  from .input_output import get_particles
  from .preconditioning import transform_to_sample_system, propagate_to_sample_surface
  instrument = _instrument(instrument_name, offset, beam_declination_angle, wavelength, sample_orientation)
  particles = get_particles(mcpl_path, 1.0, [float('-inf'), float('inf')], 0.0, False)[0]
  p, lam = particles[:, 0], particles[:, 7]
  info = {'mcpl_mean_wavelength': float(np.average(lam, weights=p)), 'mcpl_total_rate': float(p.sum())}
  declination = beam_declination_angle if beam_declination_angle is not None else instrument_defaults[instrument_name].get('beam_declination_angle', 0.0)
  particles = transform_to_sample_system(particles, 0.0, sample_orientation, declination)
  particles = propagate_to_sample_surface(particles, 0.0, 0.0, allow_sample_miss=True)  # no sample: straight to the detector
  weight, x, y, z, vx, vy, vz, _, t = particles.T[:9]
  np.random.seed(seed)  # detector resolution smearing
  q = instrument.calculate_q(x, y, z, t, vx, vy, vz)
  q_y, q_z = instrument.get_q_pixel_limits()
  image, _, _ = np.histogram2d(q[:, 0], q[:, 1], bins=[q_y, q_z], weights=weight)
  return image, info, instrument

def compare_with_simulated_direct_beam(filepath, mcpl_path, offset, beam_declination_angle=None, wavelength=6.0, sample_orientation=1,
                                       instrument_name='d22', experiment_time=None, figure=None, savename='beam_centre_check',
                                       residual_warning_pixels=0.5, beam_window=None):
  """
  Cross-check the detector offset with a direct-beam simulation of the McStas beam: report the centroid
  residual (simulated - measured), the spot widths, the MCPL mean wavelength, and the intensity factor (if
  experiment_time is given). Both centroids use the window size of the measured beam. The offset itself
  always comes from the measurement; a residual points at the McStas model (beam direction/position,
  slits), the beam declination, or the wavelength. Returns a dict of the results.
  """
  measured, _, _, _ = read_nexus_data(filepath, 0.0, wavelength, sample_orientation=sample_orientation)
  simulated, info, instrument = simulate_direct_beam(mcpl_path, offset, beam_declination_angle, wavelength, sample_orientation, instrument_name)
  det = instrument.detector
  spot_meas = beam_spot(measured, det, _image_axes_window(beam_window, sample_orientation), label="measured direct beam")
  spot_sim = beam_spot(simulated, det, spot_meas['half'], label="simulated direct beam")  # the same window size
  residual = spot_sim['centre'] - spot_meas['centre']
  pixel = np.array([det.pixel_size_y_bornagain, det.pixel_size_z_bornagain])
  results = dict(info, residual=residual, residual_pixels=residual / pixel, width_measured=spot_meas['width'],
                 width_simulated=spot_sim['width'])
  print("\n--- Direct-beam simulation cross-check (sample frame: y along the sample surface, z along its normal) ---")
  print(f"Centroid residual simulated - measured [mm]: y={residual[0]*1e3:+.3f}, z={residual[1]*1e3:+.3f} "
        f"({residual[0]/pixel[0]:+.2f} px, {residual[1]/pixel[1]:+.2f} px)")
  print(f"RMS spot width measured / simulated [mm]: y={spot_meas['width'][0]*1e3:.2f} / {spot_sim['width'][0]*1e3:.2f}, "
        f"z={spot_meas['width'][1]*1e3:.2f} / {spot_sim['width'][1]*1e3:.2f}")
  print(f"Wavelength: given {wavelength:.4f} Å, MCPL mean {info['mcpl_mean_wavelength']:.4f} Å")
  if np.any(np.abs(residual / pixel) > residual_warning_pixels):
    print(f"WARNING: the simulated direct beam is more than {residual_warning_pixels} pixel off the measured one: the difference "
          f"points at the McStas beam (direction, position, slits), the beam declination or the wavelength.")
  if abs(info['mcpl_mean_wavelength'] - wavelength) > 0.02 * wavelength:
    print("WARNING: the mean wavelength of the MCPL file differs from --wavelength by more than 2%: wrong MCPL file or wavelength?")
  if experiment_time:
    results['intensity_factor'] = measured.sum() / experiment_time / simulated.sum()
    print(f"Intensity factor = measured counts / experiment_time / simulated rate on the detector = "
          f"{measured.sum():.0f} / {experiment_time} / {simulated.sum():.2f} = {results['intensity_factor']:.4f}")
  if figure:
    _direct_beam_figure(measured, simulated, det, spot_meas, spot_sim, figure, savename)
  return results

def _axis_labels(ax):
  ax.set_xlabel('along the sample surface (BornAgain y) [mm]')
  ax.set_ylabel('along the sample normal (BornAgain z) [mm]')

def _show_image(ax, image, det, centre, half_view, scale=None):
  from matplotlib.colors import LogNorm
  y_c, z_c = _pixel_centres(det)
  extent = np.array([y_c[0] - 0.5 * det.pixel_size_y_bornagain, y_c[-1] + 0.5 * det.pixel_size_y_bornagain,
                     z_c[0] - 0.5 * det.pixel_size_z_bornagain, z_c[-1] + 0.5 * det.pixel_size_z_bornagain]) * 1e3
  norm = None if scale == 'linear' else LogNorm(vmin=max(image.max() * 1e-5, 0.5), vmax=image.max())
  ax.imshow(np.maximum(image.T, 0.5), origin='lower', extent=extent, norm=norm, cmap='viridis', interpolation='nearest', aspect='equal')
  ax.set_xlim((centre[0] - half_view[0]) * 1e3, (centre[0] + half_view[0]) * 1e3)
  ax.set_ylim((centre[1] - half_view[1]) * 1e3, (centre[1] + half_view[1]) * 1e3)
  _axis_labels(ax)

def _window_patch(centre, half, colour, label):
  from matplotlib.patches import Rectangle
  return Rectangle(((centre[0] - half[0]) * 1e3, (centre[1] - half[1]) * 1e3), 2 * half[0] * 1e3, 2 * half[1] * 1e3,
                   fill=False, ec=colour, ls='--', lw=1.5, label=label)

def _save(plt, figure, savename):
  plt.tight_layout()
  if figure == 'show':
    plt.show()
  else:
    path = f"{savename}.{figure}"
    plt.savefig(path)
    print(f"Created {path}")

def _direct_beam_figure(measured, simulated, det, spot_meas, spot_sim, figure, savename):
  import matplotlib
  if figure != 'show':
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  scaled = simulated * measured.sum() / simulated.sum()
  half = spot_meas['half']
  fig, axes = plt.subplots(2, 2, figsize=(13, 10))
  for ax, image, title, spot in ((axes[0, 0], measured, 'Measured direct beam', spot_meas),
                                 (axes[0, 1], scaled, 'Simulated direct beam (scaled to measured total)', spot_sim)):
    _show_image(ax, image, det, spot_meas['centre'], 1.4 * half + 2 * np.array([det.pixel_size_y_bornagain, det.pixel_size_z_bornagain]))
    ax.plot(spot['centre'][0] * 1e3, spot['centre'][1] * 1e3, 'x', color='red', ms=12, mew=2.5, label='centroid')
    ax.add_patch(_window_patch(spot['centre'], half, 'red', f'centroid window (+-{half[0] * 1e3:.0f} mm along the surface, +-{half[1] * 1e3:.0f} mm along the normal)'))
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
  y_c, z_c = _pixel_centres(det)
  for ax, axis, coords, index, label in ((axes[1, 0], 1, y_c, 0, 'along the sample surface (BornAgain y)'),
                                         (axes[1, 1], 0, z_c, 1, 'along the sample normal (BornAgain z)')):
    ax.plot(coords * 1e3, measured.sum(axis=axis), 'b.-', label='measured')
    ax.plot(coords * 1e3, scaled.sum(axis=axis), 'g.-', label='simulated (scaled)')
    ax.axvline(spot_meas['centre'][index] * 1e3, color='b', ls='--')
    ax.axvline(spot_sim['centre'][index] * 1e3, color='g', ls='--')
    ax.set_xlim((spot_meas['centre'][index] - 1.6 * half[index]) * 1e3, (spot_meas['centre'][index] + 1.6 * half[index]) * 1e3)
    ax.set_yscale('log')
    ax.set_xlabel(f'{label} [mm]')
    ax.set_ylabel('counts (summed)')
    ax.legend()
    ax.grid(True)
  _save(plt, figure, savename)

def measure_incident_angle(sample_filepath, direct_beam_filepath, instrument_name='d22', sample_orientation=1, wavelength=6.0,
                           min_separation_pixels=3, beam_window=None, specular_window=None, alpha=None, figure=None,
                           savename='incident_angle_check', return_details=False):
  """
  Measure the real incident angle from a sample measurement: the specular spot lies 2*alpha from the direct
  beam along the sample normal (both at the same wavelength, so the gravity drop cancels):
  alpha = 0.5 * atan(s / L). Returns alpha [deg] (with return_details, a dict with the positions and windows).
  The direct beam is found with beam_spot. The specular window is a rectangle along the sample normal; by
  default it has the direct-beam window size across the normal and SPECULAR_ALONG_FACTOR times it along the
  normal (a flat sample images the direct beam, slightly broadened), or specular_window =
  [half_along, half_across] (m). The search starts at the maximum of the 3x3-median-filtered sample image near
  the expected position L*tan(2*alpha) if alpha is given, else anywhere at least min_separation_pixels beyond
  the direct beam along the normal; the window is then re-centred on its centroid until it settles. The window
  has to hold the whole spot: the specular of a wide beam is flat-topped, and a window around its brightest
  pixels alone is biased. The measurement fixes L*tan(2*alpha) with the nominal distance L: a distance error
  cannot be told apart from an angle error.
  """
  geometry = _instrument(instrument_name, None, None, wavelength, sample_orientation)
  det = geometry.detector
  L = geometry.sample_detector_distance
  direct, _, _, _ = read_nexus_data(direct_beam_filepath, 0.0, wavelength, sample_orientation=sample_orientation)
  sample, _, _, _ = read_nexus_data(sample_filepath, 0.0, wavelength, sample_orientation=sample_orientation)
  beam = beam_spot(direct, det, _image_axes_window(beam_window, sample_orientation), label=f"direct beam of {direct_beam_filepath}")
  _print_beam_window(beam, sample_orientation, beam_window is not None)
  c_direct = beam['centre']
  y_c, z_c = _pixel_centres(det)
  Y, Z = np.meshgrid(y_c, z_c, indexing='ij')
  along, across = Z - c_direct[1], Y - c_direct[0]  # in the sample frame the normal is BornAgain z
  if specular_window is None:
    half = np.array([SPECULAR_ALONG_FACTOR * beam['half'][1], beam['half'][0]])
    how = f"from the direct-beam window (x{SPECULAR_ALONG_FACTOR:g} along the normal)"
  else:
    half = np.asarray(specular_window, dtype=float)
    how = "as given"
  if alpha is not None:
    s_expected = L * np.tan(2 * np.radians(alpha))
    allowed = (np.abs(along - s_expected) <= max(3 * half[0], 5 * det.pixel_size_z_bornagain)) & (np.abs(across) <= half[1])
  else:
    allowed = along > min_separation_pixels * max(det.pixel_size_y_bornagain, det.pixel_size_z_bornagain)
  if not np.any(allowed & (sample > 0)):
    raise ValueError("No specular reflection found in the sample measurement (check --alpha and --sample_orientation).")
  i, j = _smoothed_maximum(sample, allowed)
  (s, c), _, inside = _window_centroid(sample, along, across, (along[i, j], across[i, j]), half)
  _warn_if_cut(sample, inside, f"specular window of {sample_filepath}")
  print(f"Specular window: +-{half[0] * 1e3:.1f} mm along the sample normal x +-{half[1] * 1e3:.1f} mm across it, {how}. "
        f"To set it: --specular_window {half[0]:.4f} {half[1]:.4f}")
  measured_alpha = float(np.rad2deg(0.5 * np.arctan(s / L)))
  spot = np.array([c_direct[0] + c, c_direct[1] + s])
  if figure:
    _incident_angle_figure(direct, sample, det, c_direct, spot, beam['half'], half, measured_alpha, figure, savename)
  if return_details:
    return {'alpha': measured_alpha, 'separation': float(s), 'direct_beam': c_direct, 'specular': spot,
            'beam_half': beam['half'], 'specular_half': half, 'sample_detector_distance': L}
  return measured_alpha

def _incident_angle_figure(direct, sample, det, c_direct, spot, beam_half, specular_half, alpha, figure, savename):
  """Direct beam and sample images with the found centres (x) and the windows used to find them."""
  import matplotlib
  if figure != 'show':
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  specular_half_image = np.array([specular_half[1], specular_half[0]])  # (along the surface, along the normal)
  separation = spot[1] - c_direct[1]
  fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
  reach = 0.5 * separation + 1.3 * max(specular_half)
  panels = ((axes[0], direct, 'Direct beam', c_direct, 1.25 * np.full(2, max(beam_half)), None),
            (axes[1], sample, f'Sample: specular {separation * 1e3:.1f} mm from the direct beam, alpha = {alpha:.4f} deg',
             0.5 * (c_direct + spot), np.full(2, reach), None),
            (axes[2], sample, 'Specular close-up (linear scale)', spot, 1.25 * np.full(2, max(specular_half)), 'linear'))
  for ax, image, title, centre, half_view, scale in panels:
    _show_image(ax, image, det, centre, half_view, scale)
    if ax is not axes[2]:
      ax.plot(c_direct[0] * 1e3, c_direct[1] * 1e3, 'x', color='red', ms=12, mew=2.5, label='direct-beam centre')
    if ax is axes[0]:
      ax.add_patch(_window_patch(c_direct, beam_half, 'red', f'direct-beam window (+-{beam_half[0] * 1e3:.0f} mm along the surface, +-{beam_half[1] * 1e3:.0f} mm along the normal)'))
    if ax is not axes[0]:
      ax.plot(spot[0] * 1e3, spot[1] * 1e3, 'x', color='magenta', ms=12, mew=2.5, label='specular centre')
      ax.add_patch(_window_patch(spot, specular_half_image, 'magenta',
                                 f'specular window (+-{specular_half[0] * 1e3:.0f} mm along the normal, +-{specular_half[1] * 1e3:.0f} mm across)'))
    ax.set_title(title, fontsize=10)
    ax.legend(loc='best', fontsize=7)
  _save(plt, figure, savename)

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Find required detector centre_offset for a given NeXus data file.")
  parser.add_argument('filepath', type=str, help="Path to the NeXus data file (direct-beam measurement).")
  parser.add_argument('--wavelength', type=float, default=6.0, help="Wavelength in Angstroms (default: 6.0).")
  parser.add_argument('--sample_orientation', type=int, default=1, help="Sample orientation (default: 1).")
  parser.add_argument('--instrument', type=str, default='d22', help="Instrument name in instrument_defaults (default: 'd22').")
  parser.add_argument('--beam_declination', type=float, default=None, help="Override beam declination angle in degrees (default: loaded from instrument defaults).")
  parser.add_argument('--beam_window', type=float, nargs=2, metavar=('HALF_X', 'HALF_Y'), default=None, help=f"Half-sizes [m] of the rectangular window (along the detector x and y axes) in which the direct-beam centroid is computed. Default: {BEAM_WINDOW_WIDTHS:g} RMS widths of the beam on each axis, determined from the image (printed with the option that reproduces it).")
  parser.add_argument('--verbose', action='store_true', help="Print detailed optimization progress.")
  check = parser.add_argument_group('Cross-checks (optional)')
  check.add_argument('--mcpl', type=str, default=None, help="McStas MCPL file of the direct-beam configuration: send its neutrons to the detector with the found offset and compare the simulated with the measured direct beam (centroid residual, widths, wavelength, intensity factor).")
  check.add_argument('--experiment_time', type=float, default=None, help="Duration of the direct-beam measurement [s], for the intensity factor (with --mcpl).")
  check.add_argument('--figure', choices=['png', 'pdf', 'show'], default=None, help="Figures with the found centres and the windows used to find them: the measured and simulated direct beam (with --mcpl) and the incident-angle measurement (with --sample_nxs). Save as png/pdf, or show them.")
  check.add_argument('--savename', type=str, default='beam_centre_check', help="Output file name (without extension) for --figure png/pdf (default: beam_centre_check; the incident-angle figure gets the suffix _incident_angle).")
  check.add_argument('--sample_nxs', type=str, default=None, help="Sample measurement (same wavelength and detector position): measure the real incident angle from the distance of the specular spot to the direct beam, and compare it with --alpha.")
  check.add_argument('--alpha', type=float, default=None, help="Intended incident angle [deg]: the specular is searched near its expected position, and the measured angle is compared with it.")
  check.add_argument('--specular_window', type=float, nargs=2, metavar=('HALF_ALONG', 'HALF_ACROSS'), default=None, help=f"Half-sizes [m] of the rectangular window (along and across the sample normal) in which the specular centroid is computed (with --sample_nxs). Default: the direct-beam window size across the normal and {SPECULAR_ALONG_FACTOR:g} times it along the normal (printed with the option that reproduces it).")

  args = parser.parse_args()
  if args.experiment_time and not args.mcpl:
    parser.error("--experiment_time requires --mcpl.")
  if args.figure and not (args.mcpl or args.sample_nxs):
    parser.error("--figure requires --mcpl or --sample_nxs.")

  try:
    offset = find_required_centre_offset(
        args.filepath,
        beam_declination_angle=args.beam_declination,
        wavelength=args.wavelength,
        sample_orientation=args.sample_orientation,
        instrument_name=args.instrument,
        verbose=args.verbose,
        beam_window=args.beam_window
    )
    print(f"Calculated centre_offset [m]:  [{offset[0]:.6f}, {offset[1]:.6f}]")
    if args.mcpl:
      compare_with_simulated_direct_beam(args.filepath, args.mcpl, offset, beam_declination_angle=args.beam_declination,
                                         wavelength=args.wavelength, sample_orientation=args.sample_orientation,
                                         instrument_name=args.instrument, experiment_time=args.experiment_time,
                                         figure=args.figure, savename=args.savename, beam_window=args.beam_window)
    if args.sample_nxs:
      alpha_measured = measure_incident_angle(args.sample_nxs, args.filepath, args.instrument, args.sample_orientation, args.wavelength,
                                              beam_window=args.beam_window, specular_window=args.specular_window,
                                              alpha=args.alpha, figure=args.figure, savename=f"{args.savename}_incident_angle")
      print(f"\nIncident angle measured from the specular spot of {args.sample_nxs}: {alpha_measured:.4f} deg")
      if args.alpha is not None:
        print(f"Given --alpha: {args.alpha:.4f} deg (difference {alpha_measured - args.alpha:+.4f} deg)")
        if abs(alpha_measured - args.alpha) > max(0.05 * abs(args.alpha), 0.01):
          print("WARNING: the measured incident angle differs from --alpha by more than 5% (or 0.01 deg): check the "
                "incident angle, the sample orientation, and the sample alignment.")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)


if __name__ == '__main__':
  main()
