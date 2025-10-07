
import numpy as np

from .instrument_defaults import instrument_defaults, default_detector
from .instrument import Instrument

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
  instr_params = instrument_defaults[args.instrument]
  detector_params = instr_params.get('detector', default_detector) #add default detector if needed
  #TODO input arguments should be in place to override the detector and instrument parameters
  instr_params['detector'] = detector_params
  instrument = Instrument(instr_params, args.alpha, args.wavelength_selected, args.wfm)

  wavelength = args.wavelength_selected if args.wavelength_selected else args.wavelength
  q_min, q_max = instrument.calculate_q_limits(wavelength)
  hist_ranges = [
    args.x_range if args.x_range else [q_min[0], q_max[0]],
    args.y_range if args.y_range else [q_min[1], q_max[1]],
    args.z_range if args.z_range else [-1000, 1000]
  ]
  hist_bins = args.bins if args.bins else [instrument.detector.pixels_x, instrument.detector.pixels_y, 1]

  angle_x_maximum, angle_y_maximum = instrument.get_detector_angle_maximum()
  angle_range = args.angle_range if args.angle_range else [angle_x_maximum, angle_y_maximum]

  return {
    'sim_module_name': args.model,
    'outgoing_direction_number': args.outgoing_direction_number,
    'alpha_inc': float(np.deg2rad(args.alpha)),
    'angle_range': angle_range,
    'raw_output': args.raw_output,
    'bins': hist_bins,
    'hist_ranges': hist_ranges,
    'sample_xwidth': args.sample_xwidth,
    'sample_zheight': args.sample_zheight,
    'sample_kwargs': parse_sample_arguments(args),
    'instrument': instrument
  }