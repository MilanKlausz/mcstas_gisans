
import numpy as np

from .instruments import instrumentParameters
from .detector import Detector

def parse_sample_arguments(args):
  """Parse the --sample_arguments string into keyword arguments."""
  if not args.sample_arguments:
    return {}

  def convert_numbers(value):
    """Attempt to convert strings to integers or floats"""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

  kwargs = {}
  pairs = args.sample_arguments.split(';')
  for pair in pairs:
    key, value = pair.split('=')
    kwargs[key.strip()] = convert_numbers(value.strip())
  return kwargs

def pack_parameters(args):
  """Pack parameters necessary for processing in a single dictionary"""
  inst_params = instrumentParameters[args.instrument]
  beam_declination = 0 if not 'beam_declination_angle' in inst_params else inst_params['beam_declination_angle']
  sample_inclination = float(np.deg2rad(args.alpha - beam_declination))
  nominal_source_sample_distance = inst_params['nominal_source_sample_distance'] - (0 if not args.wfm else inst_params['wfm_virtual_source_distance'])
  
  alpha_inc = float(np.deg2rad(args.alpha))
  detector = Detector(inst_params['detector'], sample_inclination, alpha_inc,  nominal_source_sample_distance, args.wavelength_selected)

  wavelength = args.wavelength_selected if args.wavelength_selected else args.wavelength
  q_min, q_max = detector.calculate_q_limits(wavelength)
  hist_ranges = [
    args.x_range if args.x_range else [q_min[0], q_max[0]],
    args.y_range if args.y_range else [q_min[1], q_max[1]],
    args.z_range if args.z_range else [-1000, 1000]
  ]
  hist_bins = args.bins if args.bins else [detector.pixels_x, detector.pixels_y, 1]
  
  return {
    'sim_module_name': args.model,
    'pixelNr': args.pixel_number,
    'alpha_inc': alpha_inc,
    'angle_range': args.angle_range,
    'raw_output': args.raw_output,
    'bins': hist_bins,
    'hist_ranges': hist_ranges,
    'sample_xwidth': args.sample_xwidth,
    'sample_zheight': args.sample_zheight,
    'sample_kwargs': parse_sample_arguments(args),
    'detector': detector
  }