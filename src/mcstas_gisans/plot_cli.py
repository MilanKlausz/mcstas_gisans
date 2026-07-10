
import argparse
from .instrument_defaults import instrument_defaults

def zeroToOne(x):
  """Argparser type check function for float number in range [0.0, 1.0]"""
  try:
      x = float(x)
  except ValueError:
      raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
  if x < 0.0 or x > 1.0:
      raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
  return x

def create_argparser():
  parser = argparse.ArgumentParser(description = 'Create Q plots from an .npz file containing the derived Q values for each outgoing neutron from the BornAgain simulation.')
  parser.add_argument('-f', '--filename', nargs = '*', help = 'Input filename[s].')
  parser.add_argument('-l', '--label', nargs = '*', help = 'Label for input[s].')
  parser.add_argument('-s', '--savename', default='qPlot', required=False, help = 'Output image filename.')
  parser.add_argument('--pdf', action='store_true', help = 'Export figure as pdf.')
  parser.add_argument('--png', action='store_true', help = 'Export figure as png.')
  parser.add_argument('-t', '--experiment_time', default=None, type=int, help = 'Experiment time in seconds to scale the results up to. (e.g. 10800). Must be a positive integer.')
  parser.add_argument('--background', default=0, type=float, help = 'Add Poisson background to each bin.')
  parser.add_argument('-v', '--verbose', action='store_true', help = 'Verbose output.')
  parser.add_argument('--csv', action='store_true', help = 'Output the resulting histograms in csv format.')
  parser.add_argument('-i', '--instrument', default='d22', type=str.lower, choices=list(instrument_defaults.keys()), help = 'Instrument (from instruments.py).')

  plotParamGroup = parser.add_argument_group('Control plotting', 'Parameters and options for plotting.')
  plotParamGroup.add_argument('--font_size', type=int, default=14, help = 'Global font size for plot elements.')
  plotParamGroup.add_argument('-d', '--dual_plot', default=False, action='store_true', help = 'Create a dual plot in a single figure.')
  plotParamGroup.add_argument('--multi2d', default=False, action='store_true', help = 'Create a figue with multiple subplots for 2D Q plots.')
  plotParamGroup.add_argument('-m', '--intensity_min', default=None, help = 'Intensity minimum for the 2D q plot colorbar.')
  plotParamGroup.add_argument('--individual_colorbars', default=False, action='store_true', help = 'Allow different individual colorbars for multiple 2D q plots.')
  plotParamGroup.add_argument('-q', '--q_min', default=0.09, type=float, help = 'Vertical component of the Q values of interest. Used as the minimum of the range if q_max is provided as well.')
  plotParamGroup.add_argument('--q_max', default=0.10, type=float, help = 'Maximum of the vertical component of the Q range of interest.')
  plotParamGroup.add_argument('--y_plot_range', nargs=2, type=float, help = 'Plot y range.')
  plotParamGroup.add_argument('--z_plot_range', nargs=2, type=float, help = 'Plot z range.')
  plotParamGroup.add_argument('--plot_differences', default=0, type=int, help = 'Plot some measure of difference: 0 - none, 1 - relative absolute difference, 2 - relative difference, 3 - normalised residuals')

  # findTimeParamGroup = parser.add_argument_group('Find experiment time', 'Parameters and options for finding the experiment time to scale up to.')
  # findTimeParamGroup.add_argument('--find_experiment_time', action='store_true', help = 'Find the minimum experiment time the results need to be upscaled to in order to get a certain minimum number of counts in the bins.')
  # findTimeParamGroup.add_argument('-i', '--iterate', action='store_true', help = 'Iteratively find the experiment time for which the bin count criterion is fulfilled after adding Gaussian noise.')
  # findTimeParamGroup.add_argument('--maximum_iteration_number', type=int, default=50, help = 'Maximum number of iterations.')
  # findTimeParamGroup.add_argument('--minimum_count_number', default=36, type=int, help = 'Minimum number of counts expected in the bins.')
  # findTimeParamGroup.add_argument('--minimum_count_fraction', type=zeroToOne, default=0.8, help = 'The fraction of bins that are required to fulfill the minimum count number criterion. [0,1]')

  rawFormat = parser.add_argument_group('Raw Q events data', 'Use (old) raw data format with Q event list in the file instead of an already histogrammed data.')
  rawFormat.add_argument('--bins', nargs=2, type=int, default=[256, 128], help='Number of histogram bins in y,z directions.')
  rawFormat.add_argument('--y_range', nargs=2, type=float, default=[-0.55, 0.55], help='Qy range of the histogram.')
  rawFormat.add_argument('--z_range', nargs=2, type=float, default=[-0.5, 0.6], help='Qx range of the histogram.')

  storedDataParamGroup = parser.add_argument_group('Stored data', 'Use stored data files for plotting or comparison.')
  storedDataParamGroup.add_argument('--nxs', nargs = '*', help = 'Full path to the D22 Nexus file.')
  storedDataParamGroup.add_argument('--nxs_label', nargs = '*', help = 'Label for Nexus input[s]. Must be used together with --nxs if a label is desired. If not provided, the label will be generated from the Nexus file name.')
  storedDataParamGroup.add_argument('--overlay', action='store_true', help = 'Overlay stored data with simulated data.') #TODO isn't it more general than that?
  storedDataParamGroup.add_argument('--normalise_to_nxs', action='store_true', help = 'Normalise simulated data to the total intensity in the Nexus file.')
  storedDataParamGroup.add_argument('--sample_orientation', default=1, choices=[0,1,2], type=float, help = 'Orientation of the sample. 1 - horizontal sample, 0/2 - vertical sample with the beam hitting it from left/right.')

  instrumentGroup = parser.add_argument_group('Instrument overrides', 'Override default parameters for the selected instrument.')
  instrumentGroup.add_argument('--instrument_nominal_source_sample_distance', type=float, help='Override nominal source to sample distance. [m]')
  instrumentGroup.add_argument('--instrument_sample_detector_distance', type=float, help='Override sample to detector distance. [m]')
  instrumentGroup.add_argument('--instrument_detector_size', nargs=2, type=float, help='Override detector dimensions [size_x, size_y] in meters.')
  instrumentGroup.add_argument('--instrument_detector_centre_offset', nargs=2, type=float, help='Override detector centre offset [offset_x, offset_y] in meters.')
  instrumentGroup.add_argument('--instrument_detector_pixels', nargs=2, type=int, help='Override detector pixel counts [pixels_x, pixels_y].')
  instrumentGroup.add_argument('--instrument_detector_resolution', nargs=2, type=float, help='Override detector resolution FWHM [res_x, res_y] in meters.')
  instrumentGroup.add_argument('--instrument_tof_instrument', type=str.lower, choices=['true', 'false'], help='Override whether the instrument is a Time-of-Flight (TOF) instrument.')
  instrumentGroup.add_argument('--instrument_t0_monitor_name', type=str, help='Override t0 monitor name.')
  instrumentGroup.add_argument('--instrument_wfm_t0_monitor_name', type=str, help='Override WFM t0 monitor name.')
  instrumentGroup.add_argument('--instrument_wfm_virtual_source_distance', type=float, help='Override WFM virtual source distance. [m]')

  return parser

def parse_args(parser):
  args = parser.parse_args()

  # Apply instrument parameter overrides in instrument_defaults
  import copy
  instr_name = args.instrument
  if instr_name in instrument_defaults:
    # Deep copy the default dictionary so we don't permanently alter the module defaults for other scripts
    instr_params = copy.deepcopy(instrument_defaults[instr_name])

    if args.instrument_nominal_source_sample_distance is not None:
      instr_params['nominal_source_sample_distance'] = args.instrument_nominal_source_sample_distance

    if args.instrument_sample_detector_distance is not None:
      instr_params['sample_detector_distance'] = args.instrument_sample_detector_distance

    if args.instrument_tof_instrument is not None:
      instr_params['tof_instrument'] = (args.instrument_tof_instrument == 'true')

    if args.instrument_t0_monitor_name is not None:
      instr_params['t0_monitor_name'] = args.instrument_t0_monitor_name

    if args.instrument_wfm_t0_monitor_name is not None:
      instr_params['wfm_t0_monitor_name'] = args.instrument_wfm_t0_monitor_name

    if args.instrument_wfm_virtual_source_distance is not None:
      instr_params['wfm_virtual_source_distance'] = args.instrument_wfm_virtual_source_distance

    # Handle detector overrides
    if 'detector' not in instr_params:
      from .instrument_defaults import default_detector
      instr_params['detector'] = copy.deepcopy(default_detector)

    det_params = instr_params['detector']

    if args.instrument_detector_size is not None:
      det_params['size'] = list(args.instrument_detector_size)

    if args.instrument_detector_centre_offset is not None:
      det_params['direct_beam_centre_offset'] = list(args.instrument_detector_centre_offset)

    if args.instrument_detector_pixels is not None:
      det_params['pixels'] = list(args.instrument_detector_pixels)

    if args.instrument_detector_resolution is not None:
      det_params['resolution'] = list(args.instrument_detector_resolution)

    # Replace the dict in instrument_defaults
    instrument_defaults[instr_name] = instr_params

  if args.filename is None and args.nxs is None:
    parser.error('No input file provided! This is only allowed when the --nxs option is used.')

  if args.label and len(args.label) != len(args.filename):
    parser.error(f"The number of labels(${len(args.label)}) doesn't agree with the number of files(${len(args.filename)})")

  if (args.experiment_time is not None) and args.experiment_time <= 0:
    parser.error('The --experiment_time must be a positive integer.')

  # if args.minimum_count_number < 0:
  #   parser.error('The --minimum_count_number must be a non-negative integer.')

  # if args.iterate and not args.find_experiment_time:
  #   parser.error('The --iterate option can only be used when --find_experiment_time is also in use.')

  if args.normalise_to_nxs and not args.nxs:
    parser.error('The --normalise_to_nxs option can only be used when --nxs is also in use.')
  
  return args