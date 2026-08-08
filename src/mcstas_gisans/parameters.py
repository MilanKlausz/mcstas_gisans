
from .instrument_defaults import set_instrument_parameters
from .instrument import Instrument
from .sample import Sample

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

  if getattr(args, 'outgoing_directions_horizontal', None) is not None:
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
  }