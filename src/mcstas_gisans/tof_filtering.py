

from pathlib import Path

from .instruments import instrumentParameters
from .input_output import print_tof_limits

def get_tof_filtering_limits(args):
  """
  Get TOF (time-of-flight) limits that can be used for filtering neutrons from
  the MCPL input file. The options are:
    1) No filtering ([-inf, inf])
    2) Using input values (args.input_tof_limits)
    3) Derive limits from a 1D TOF spectrum corresponding to a selected
       wavelength that is retrieved from a 2D McStas TOFLambda_monitor spectrum
       (that is assumed to represent the MCPL file content). The limits are
       defined by fitting a Gaussian function and getting a single FWHM range
       centred around the mean TOF value.
  """
  instParams = instrumentParameters[args.instrument]
  tofLimits = [float('-inf'), float('inf')]
  if instParams['tof_instrument'] and not args.no_mcpl_filtering and (args.input_tof_limits or args.wavelength):
    if args.input_tof_limits:
      tofLimits = args.input_tof_limits
    else:
      if args.figure_output == 'png' or args.figure_output == 'pdf':
        figureOutput = f"{args.savename}.{args.figure_output}"
      else:
        figureOutput = args.figure_output # None or 'show'
      mcstasDir = Path(args.filename).resolve().parent
      from .fit_monitor import fitGaussianToMcstasMonitor
      fit = fitGaussianToMcstasMonitor(dirname=mcstasDir, monitor=instParams['mcpl_monitor_name'], wavelength=args.wavelength, wavelength_rebin=args.input_wavelength_rebin, figureOutput=figureOutput, tofRangeFactor=args.input_tof_range_factor)
      tofLimits[0] = (fit['mean'] - fit['fwhm'] * 0.5 * args.input_tof_range_factor) * 1e-3
      tofLimits[1] = (fit['mean'] + fit['fwhm'] * 0.5 * args.input_tof_range_factor) * 1e-3
      if args.figure_output is not None:
      # Terminate the script execution because 'only plotting' has been selected by the user
        import sys
        print_tof_limits(tofLimits)
        sys.exit()
  return tofLimits