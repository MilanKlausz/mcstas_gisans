import numpy as np

from .instrument_defaults import set_instrument_parameters
from .instrument import Instrument
from .sample import Sample

# Noise of the direction sampling per pixel: about SAMPLING_NOISE_COEFFICIENT / sqrt(rays per pixel), with the rays
# counted with the effective number of neutrons (measured for the D22 paper data: 1.5/sqrt(N rho) with N_eff = 0.75 N)
SAMPLING_NOISE_COEFFICIENT = 1.3

def outgoing_directions_for_sampling(angle_range, detector, sample_detector_distance, n_effective, rays_per_pixel):
  """
  Number of outgoing directions (horizontal, vertical) with which every detector pixel collects about
  rays_per_pixel rays over the run, in the angle range [horiz_min, horiz_max, vert_min, vert_max] (deg).
  The grid of every neutron is shifted randomly by up to half a bin, so any grid is unbiased; the grid
  only sets the noise of the direction sampling, about SAMPLING_NOISE_COEFFICIENT/sqrt(rays per pixel)
  per pixel. A pixel collects n_effective * rho rays, where n_effective = (sum w)^2 / sum w^2 is the
  effective number of the incident neutrons hitting the sample (their number, for equal weights w) and
  rho the number of directions per pixel of angular area. The directions are split with the same bin
  width (in pixels) along both axes. The pixel sizes are those along the BornAgain horizontal and
  vertical axes, i.e. swapped for vertical samples.
  """
  if n_effective < 1:
    print("WARNING: No incident neutron hits the sample, the outgoing directions are chosen as for a single neutron.")
    n_effective = 1
  pixel_deg_h = np.degrees(detector.pixel_size_y_bornagain / sample_detector_distance)
  pixel_deg_v = np.degrees(detector.pixel_size_z_bornagain / sample_detector_distance)
  pixels_h = (angle_range[1] - angle_range[0]) / pixel_deg_h
  pixels_v = (angle_range[3] - angle_range[2]) / pixel_deg_v
  directions_per_pixel_1d = np.sqrt(rays_per_pixel / n_effective)
  n_h = int(np.ceil(directions_per_pixel_1d * pixels_h - 1e-9))
  n_v = int(np.ceil(directions_per_pixel_1d * pixels_v - 1e-9))
  return max(n_h, 1), max(n_v, 1)

def get_simulated_angle_range(args, instrument):
  """The simulated angle range: --angle_range (or the fit's mask range set into it), else the full detector."""
  return list(args.angle_range) if args.angle_range else list(instrument.get_detector_angle_maximum())

def set_outgoing_directions_from_sampling(args, particles, particle_type):
  """
  With a sampling target (--sampling/--rays_per_pixel), choose the outgoing-direction grid from the
  (preconditioned) particles and the final simulated angle range, and store it in
  args.outgoing_directions_horizontal/_vertical, which pack_parameters then uses. Does nothing otherwise.
  """
  rays_per_pixel = getattr(args, 'rays_per_pixel', None)
  if not rays_per_pixel:
    return
  instr_params = set_instrument_parameters(args)
  no_gravity = args.no_gravity if particle_type != 'photon' else True
  instrument = Instrument(instr_params, args.alpha, args.wavelength_selected, args.sample_orientation, args.wfm, no_gravity)
  angle_range = get_simulated_angle_range(args, instrument)

  sample = Sample(args.sample_size_y, args.sample_size_x, args.model, None)
  _, x, y, z, _, vy, *_ = np.asarray(particles).T if len(particles) else [np.zeros(0)] * 7
  hit = ~np.asarray(sample.sample_missed(x, y, z, vy), dtype=bool)
  weights = (np.asarray(particles).T[0] if len(particles) else np.zeros(0))[hit]
  n_hit = int(np.count_nonzero(hit))
  n_effective = float(weights.sum() ** 2 / (weights ** 2).sum()) if n_hit else 0.0  # effective number for unequal weights

  n_h, n_v = outgoing_directions_for_sampling(angle_range, instrument.detector, instrument.sample_detector_distance, n_effective, rays_per_pixel)
  args.outgoing_directions = None
  args.outgoing_directions_horizontal, args.outgoing_directions_vertical = n_h, n_v

  # rays per pixel actually reached (above the target, since the direction numbers are rounded up)
  pixel_area_deg2 = np.degrees(instrument.detector.pixel_size_y_bornagain / instrument.sample_detector_distance) \
                    * np.degrees(instrument.detector.pixel_size_z_bornagain / instrument.sample_detector_distance)
  angle_area_deg2 = (angle_range[1] - angle_range[0]) * (angle_range[3] - angle_range[2])
  rays = max(n_effective, 1) * n_h * n_v * pixel_area_deg2 / angle_area_deg2
  sampling = getattr(args, 'sampling', None)
  target = f"sampling '{sampling}'" if sampling else f"rays_per_pixel {rays_per_pixel:g}"
  print(f"Outgoing directions: {n_h} x {n_v} ({target}: about {rays:.0f} rays per detector pixel (target {rays_per_pixel:g}) "
        f"from {n_hit} neutrons hitting the sample, {n_effective:.0f} effective for their weights). Noise of the direction "
        f"sampling about {100 * SAMPLING_NOISE_COEFFICIENT / np.sqrt(rays):.2g}% per pixel (the statistical noise of the MCPL file adds to it). "
        f"To reproduce: --outgoing_directions_horizontal {n_h} --outgoing_directions_vertical {n_v}")

def pack_parameters(args, particle_type):
  """Pack parameters necessary for processing in a single dictionary"""
  instr_params = set_instrument_parameters(args)
  no_gravity = args.no_gravity if particle_type != 'photon' else True
  instrument = Instrument(instr_params, args.alpha, args.wavelength_selected, args.sample_orientation, args.wfm, no_gravity)

  wavelength = args.wavelength_selected if args.wavelength_selected else args.wavelength
  q_min, q_max = instrument.calculate_q_limits(wavelength)
  #reorder x,y,z because user input is in BornAgain geometry, but for now the
  #script uses the McStas axis labeling. FIXME
  hist_ranges = [
    args.y_range if args.y_range else [q_min[0], q_max[0]],
    args.z_range if args.z_range else [q_min[1], q_max[1]],
    args.x_range if args.x_range else [-1000, 1000],
  ]
  #reorder x,y,z because user input is in BornAgain geometry, but for now the
  #script uses the McStas axis labeling. FIXME
  hist_bins = [args.bins[1], args.bins[2], args.bins[0]] if args.bins else [instrument.detector.pixels_y_bornagain, instrument.detector.pixels_z_bornagain, 1]

  angle_range = get_simulated_angle_range(args, instrument)
  if getattr(args, 'verbose', False):
    print(f"Simulated angle range [deg]: horiz=[{angle_range[0]:.4f}, {angle_range[1]:.4f}], vert=[{angle_range[2]:.4f}, {angle_range[3]:.4f}]")

  sample = Sample(args.sample_size_y, args.sample_size_x, args.model, args.sample_arguments)

  if getattr(args, 'outgoing_directions_horizontal', None) is not None:
    outgoing_directions_horizontal = args.outgoing_directions_horizontal
    outgoing_directions_vertical = args.outgoing_directions_vertical
  elif getattr(args, 'rays_per_pixel', None):
    raise ValueError("The outgoing directions of the sampling target are not set yet, call set_outgoing_directions_from_sampling first.")
  else:
    outgoing_directions = getattr(args, 'outgoing_directions', 20)
    outgoing_directions_horizontal = outgoing_directions
    outgoing_directions_vertical = outgoing_directions

  return {
    'outgoing_directions_horizontal': outgoing_directions_horizontal,
    'outgoing_directions_vertical': outgoing_directions_vertical,
    'angle_range': angle_range,
    'raw_output': args.raw_output,
    'bins': hist_bins,
    'hist_ranges': hist_ranges,
    'sample': sample,
    'instrument': instrument,
    'use_avg_materials': args.use_avg_materials,
    'specular': args.specular,
    'analyzer_direction': args.analyzer_direction if any(args.analyzer_direction) else None,
    'analyzer_efficiency': args.analyzer_efficiency,
    'analyzer_transmission': args.analyzer_transmission,
    'bornagain_number_of_threads': args.bornagain_number_of_threads,
    'random_seed': getattr(args, 'seed', None),
  }