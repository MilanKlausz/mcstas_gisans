import numpy as np

from .instrument_defaults import set_instrument_parameters
from .instrument import Instrument
from .sample import Sample

def outgoing_directions_for_pixels(angle_range, detector, sample_detector_distance, directions_per_pixel=1.0):
  """
  Number of outgoing directions (horizontal, vertical) that gives directions_per_pixel directions per
  detector pixel in the angle range [horiz_min, horiz_max, vert_min, vert_max] (deg). The grid of every
  neutron is shifted randomly by up to half a bin, so a coarser grid does not blur the pattern but
  samples it with fewer directions per pixel (more statistical noise); about one direction per pixel
  in each direction spends the directions evenly. The pixel sizes are those along the BornAgain
  horizontal and vertical axes, i.e. swapped for vertical samples.
  """
  pixel_deg_h = np.degrees(detector.pixel_size_y_bornagain / sample_detector_distance)
  pixel_deg_v = np.degrees(detector.pixel_size_z_bornagain / sample_detector_distance)
  n_h = int(np.ceil(directions_per_pixel * (angle_range[1] - angle_range[0]) / pixel_deg_h - 1e-9))
  n_v = int(np.ceil(directions_per_pixel * (angle_range[3] - angle_range[2]) / pixel_deg_v - 1e-9))
  return max(n_h, 1), max(n_v, 1)

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

  default_angle_range = list(instrument.get_detector_angle_maximum())
  angle_range = list(args.angle_range) if args.angle_range else default_angle_range
  if getattr(args, 'verbose', False):
    print(f"Simulated angle range [deg]: horiz=[{angle_range[0]:.4f}, {angle_range[1]:.4f}], vert=[{angle_range[2]:.4f}, {angle_range[3]:.4f}]")

  sample = Sample(args.sample_size_y, args.sample_size_x, args.model, args.sample_arguments)

  directions_per_pixel = getattr(args, 'outgoing_directions_per_pixel', None)
  if directions_per_pixel:
    outgoing_directions_horizontal, outgoing_directions_vertical = outgoing_directions_for_pixels(
        angle_range, instrument.detector, instrument.sample_detector_distance, directions_per_pixel)
    print(f"Outgoing directions: {outgoing_directions_horizontal} x {outgoing_directions_vertical} "
          f"({directions_per_pixel:g} per detector pixel in the simulated angle range)")
  elif getattr(args, 'outgoing_directions_horizontal', None) is not None:
    outgoing_directions_horizontal = args.outgoing_directions_horizontal
    outgoing_directions_vertical = args.outgoing_directions_vertical
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