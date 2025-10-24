
"""
Create and run argparse command line interface for the plot script
"""
import argparse

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
  parser.add_argument('--background', default=0, type=float, help = 'Flat background value added to each bins.')
  parser.add_argument('-v', '--verbose', action='store_true', help = 'Verbose output.')
  parser.add_argument('--csv', action='store_true', help = 'Output the resulting histograms in csv format.')

  plotParamGroup = parser.add_argument_group('Control plotting', 'Parameters and options for plotting.')
  plotParamGroup.add_argument('-d', '--dual_plot', default=False, action='store_true', help = 'Create a dual plot in a single figure.')
  plotParamGroup.add_argument('-m', '--intensity_min', default=None, help = 'Intensity minimum for the 2D q plot colorbar.')
  plotParamGroup.add_argument('--individual_colorbars', default=False, action='store_true', help = 'Allow different individual colorbars for multiple 2D q plots.')
  plotParamGroup.add_argument('-q', '--q_min', default=0.09, type=float, help = 'Vertical component of the Q values of interest. Used as the minimum of the range is q_max is provided as well.')
  plotParamGroup.add_argument('--q_max', default=0.10, type=float, help = 'Maximum of the vertical component of the Q range of interest.')
  plotParamGroup.add_argument('--y_plot_range', nargs=2, type=float, help = 'Plot y range.')
  plotParamGroup.add_argument('--z_plot_range', nargs=2, type=float, help = 'Plot z range.')

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
  storedDataParamGroup.add_argument('--nxs', default=None, help = 'Full path to the D22 Nexus file. (Using automatic D22 measurement label for it.)')
  storedDataParamGroup.add_argument('--overlay', action='store_true', help = 'Overlay stored data with simulated data.') #TODO isn't it more general than that?
  storedDataParamGroup.add_argument('--normalise_to_nxs', action='store_true', help = 'Normalise simulated data to the total intensity in the Nexus file.')

  return parser

def parse_args(parser):
  args = parser.parse_args()

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