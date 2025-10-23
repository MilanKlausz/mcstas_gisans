

from pathlib import Path

from .instrument_defaults import instrument_defaults
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
  instParams = instrument_defaults[args.instrument]
  tof_limits = [float('-inf'), float('inf')]
  if instParams['tof_instrument'] and not args.no_mcpl_filtering and (args.input_tof_limits or args.wavelength):
    if args.input_tof_limits:
      tof_limits = args.input_tof_limits
    else:
      figure_output = f"{args.savename}_tof_filtering.{args.tof_filtering_figure}" if args.tof_filtering_figure in ['png', 'pdf'] else args.tof_filtering_figure
      mcstas_dir = Path(args.filename).resolve().parent
      from .fit_monitor import fit_gaussian_to_mcstas_monitor
      fit = fit_gaussian_to_mcstas_monitor(dirname=mcstas_dir, monitor=instParams['mcpl_monitor_name'], wavelength=args.wavelength, wavelength_rebin=args.input_wavelength_rebin, figure_output=figure_output, tof_range_factor=args.input_tof_range_factor)
      tof_limits[0] = (fit['mean'] - fit['fwhm'] * 0.5 * args.input_tof_range_factor) * 1e-3
      tof_limits[1] = (fit['mean'] + fit['fwhm'] * 0.5 * args.input_tof_range_factor) * 1e-3
      if args.tof_filtering_figure is not None:
      # Terminate the script execution because 'only plotting' has been selected by the user
        import sys
        print_tof_limits(tof_limits)
        sys.exit()
  return tof_limits