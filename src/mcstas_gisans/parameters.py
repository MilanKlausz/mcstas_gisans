
from .instrument_defaults import instrument_defaults, default_detector
from .instrument import Instrument
from .sample import Sample

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

  sample = Sample(args.sample_xwidth, args.sample_zheight, args.model, args.sample_arguments)

  return {
    'outgoing_direction_number': args.outgoing_direction_number,
    'angle_range': angle_range,
    'raw_output': args.raw_output,
    'bins': hist_bins,
    'hist_ranges': hist_ranges,
    'sample': sample,
    'instrument': instrument
  }