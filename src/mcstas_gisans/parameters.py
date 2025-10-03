
import numpy as np

from .instruments import instrumentParameters

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
  return {
    'nominal_source_sample_distance': inst_params['nominal_source_sample_distance'] - (0 if not args.wfm else inst_params['wfm_virtual_source_distance']),
    'sample_detector_distance': inst_params['sample_detector_distance'],
    'sampleToRealCoordRotMatrix' : np.array([[np.cos(sample_inclination), -np.sin(sample_inclination)],
                                             [np.sin(sample_inclination), np.cos(sample_inclination)]]),
    'realToSampleCoordRotMatrix' : np.array([[np.cos(-sample_inclination), -np.sin(-sample_inclination)],
                                             [np.sin(-sample_inclination), np.cos(-sample_inclination)]]),
    'sim_module_name': args.model,
    'pixelNr': args.pixel_number,
    'wavelengthSelected':  None if inst_params['tof_instrument'] else args.wavelengthSelected,
    'alpha_inc': float(np.deg2rad(args.alpha)),
    'angle_range': args.angle_range,
    'raw_output': args.raw_output,
    'bins': args.bins,
    'histRanges': [args.x_range, args.y_range, args.z_range],
    'sample_xwidth': args.sample_xwidth,
    'sample_zheight': args.sample_zheight,
    'sample_kwargs': parse_sample_arguments(args),
  }