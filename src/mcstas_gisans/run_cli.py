
"""
Create and run argparse command line interface for the run script
"""
import argparse
from .instruments import instrumentParameters
from .get_samples import getSampleModels

def create_argparser():
  parser = argparse.ArgumentParser(description = 'Execute BornAgain simulation of a GISANS sample with incident neutrons taken from an input file. The output of the script is a .npz file (or files) containing the derived Q values for each outgoing neutron. The default Q value calculated is aiming to be as close as possible to the Q value from a measurement.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('filename',  help = 'Input filename. (Preferably MCPL file from the McStas MCPL_output component, but .dat file from McStas Virtual_output works as well)')
  parser.add_argument('--intensity_factor', default=1.0, type=float, help = 'A multiplication factor to modify the beam intensity. (Applied to the Monte Carlo weight of each particle in the input file.)')
  parser.add_argument('-i','--instrument', required=True, choices=list(instrumentParameters.keys()), help = 'Instrument (from instruments.py).')
  parser.add_argument('-p','--parallel_processes', required=False, type=int, help = 'Number of processes to be used for parallel processing.')
  parser.add_argument('--no_parallel', default=False, action='store_true', help = 'Do not use multiprocessing. This makes the simulation significantly slower, but enables profiling. Uses --raw_output implicitly.')
  parser.add_argument('-n','--pixel_number', default=10, type=int, help = 'Number of pixels in x and y direction of the "detector".')
  parser.add_argument('--wavelengthSelected', default=6.0, type=float, help = 'Wavelength (mean) in Angstrom selected by the velocity selector. Only used for non-time-of-flight instruments.')
  parser.add_argument('--angle_range', default=1.7, type=float, help = 'Scattering angle covered by the simulation. [deg]')

  outputGroup = parser.add_argument_group('Output', 'Control the generated outputs. By default a histogram (and corresponding uncertainty) is generated as an output, saved in a npz file, loadable with the plotQ script.')
  outputGroup.add_argument('-s', '--savename', default='', required=False, help = 'Output filename (can be full path).')
  outputGroup.add_argument('--raw_output', default=False, action='store_true', help = 'Create a raw list of Q events as output instead of the default histogrammed data. Warning: this option may require too much memory for high incident event and pixel numbers.')
  outputGroup.add_argument('--bins', nargs=3, type=int, default=[256, 128, 1], help='Number of histogram bins in x,y,z directions.')
  outputGroup.add_argument('--x_range', nargs=2, type=float, default=[-0.55, 0.55], help='Qx range of the histogram. (In horizontal plane right to left)')
  outputGroup.add_argument('--y_range', nargs=2, type=float, default=[-0.5, 0.6], help='Qy range of the histogram. (In vertical plane bottom to top)')
  outputGroup.add_argument('--z_range', nargs=2, type=float, default=[-1000, 1000], help='Qz range of the histogram. (In horizontal plane back to forward)')
  outputGroup.add_argument('--quick_plot', default=False, action='store_true', help='Show a quick Qx-Qz plot from the histogram result.')
  outputGroup.add_argument('--all_q', default=False, action='store_true', help = 'Calculate and save multiple Q values, each with different levels of approximation (from real Q calculated from all simulation parameters to the default output value, that is Q calculated at the detector surface). This results in significantly slower simulations (especially due to the lack of parallelisation), but can shed light on the effect of e.g. divergence and TOF to lambda conversion on the derived Q value, in order to gain confidence in the results.')

  sampleGroup = parser.add_argument_group('Sample', 'Sample related parameters and options.')
  sampleGroup.add_argument('-a', '--alpha', default=0.24, type=float, help = 'Incident angle on the sample. [deg] (Could be thought of as a sample rotation, but it is actually achieved by an incident beam coordinate transformation.)')
  sampleGroup.add_argument('--model', default="silica_100nm_air", choices=getSampleModels(), help = 'BornAgain model to be used.')
  sampleGroup.add_argument('--sample_arguments', help = 'Input arguments of the sample model in format: "arg1=value1;arg2=value2"')
  sampleGroup.add_argument('--sample_xwidth', default=0.06, type=float, help = 'Size of sample perpendicular to beam. [m]')
  sampleGroup.add_argument('--sample_zheight', default=0.08, type=float, help = 'Size of sample along the beam. [m]')
  sampleGroup.add_argument('--allow_sample_miss', default=False, action='store_true', help = 'Allow incident neutrons to miss the sample, and being directly propagated to the detector surface. This option can be used to simulate overillumination, or direct beam simulation by also setting one of the sample sizes to zero.')

  mcplFilteringGroup = parser.add_argument_group('MCPL filtering', 'Parameters and options to control which neutrons are used from the MCPL input file. By default no filtering is applied, but if a (central) wavelength is provided, an accepted TOF range is defined based on a McStas TOFLambda monitor (defined as mcpl_monitor_name for each instrument in instruments.py) that is assumed to correspond to the input MCPL file. The McStas monitor is looked for in the directory of the MCPL input file, and after fitting a Gaussian function, neutrons within a single FWHM range centred around the selected wavelength are used for the BornAgain simulation.')
  mcplFilteringGroup.add_argument('-w', '--wavelength', type=float, default=None, help = 'Central wavelength used for filtering based on the McStas TOFLambda monitor. (Also used for t0 correction.)')
  mcplFilteringGroup.add_argument('--input_tof_range_factor', default=1.0, type=float, help = 'Modify the accepted TOF range of neutrons by this multiplication factor.')
  mcplFilteringGroup.add_argument('--input_wavelength_rebin', default=1, type=int, help = 'Rebin the TOFLambda monitor along the wavelength axis by the provided factor (only if no extrapolation is needed).')
  mcplFilteringGroup.add_argument('--input_tof_limits', nargs=2, type=float, help = 'TOF limits for selecting neutrons from the MCPL file [millisecond]. When provided, fitting to the McStas monitor is not attempted.')
  mcplFilteringGroup.add_argument('--no_mcpl_filtering', action='store_true', help = 'Disable MCPL TOF filtering. Use all neutrons from the MCPL input file.')
  mcplFilteringGroup.add_argument('--figure_output', default=None, choices=['show', 'png', 'pdf'], help = 'Show or save the figure of the selected input TOF range and exit without doing the simulation. Only works with McStas monitor fitting.')

  t0correctionGroup = parser.add_argument_group('T0 correction', 'Parameters and options to control t0 TOF correction. Currently only works if the wavelength parameter in the MCPL filtering is provided.')
  t0correctionGroup.add_argument('--t0_fixed', default=None, type=float, help = 'Fix t0 correction value that is subtracted from the neutron TOFs. [s]')
  t0correctionGroup.add_argument('--t0_wavelength_rebin', default=None, type=int, help = 'Rebinning factor for the McStas TOFLambda monitor based t0 correction. Rebinning is applied along the wavelength axis. Only integer divisors are allowed.')
  t0correctionGroup.add_argument('--wfm', default=False, action='store_true', help = 'Wavelength Frame Multiplication (WFM) mode.')
  t0correctionGroup.add_argument('--no_t0_correction', action='store_true', help = 'Disable t0 correction. (Allows using McStas simulations which lack the supported monitors.)')

  return parser

def parse_args(parser):
  args = parser.parse_args()

  if args.wfm and any(key not in instrumentParameters[args.instrument] for key in wfmRequiredKeys):
    parser.error(f"wfm option is not enabled for the {args.instrument} instrument! Set the required instrument parameters in instruments.py!")

  if args.figure_output:
    if not args.wavelength:
      parser.error(f"The --figure_output option can only be used if a central wavelength (--wavelength) for fitting is provided.")
    if args.input_tof_limits:
      parser.error(f"The --figure_output option can not be used when the TOF range is provided with --input_tof_limits.")
    if args.no_mcpl_filtering:
      parser.error(f"The --figure_output option can not be used when no TOF filtering is selected with --no_mcpl_filtering.")

  if args.no_t0_correction:
    if args.t0_fixed:
      parser.error(f"The --no_t0_correction option can not be used together with --t0_fixed ")
    if args.t0_wavelength_rebin:
      parser.error(f"The --no_t0_correction option can not be used together with --t0_wavelength_rebin ")
    if args.wfm:
      parser.error(f"The --no_t0_correction option can not be used together with --wfm ")
  else:
    if not args.wavelength:
      parser.error(f"The --wavelength must be provided for T0 correction. Alternatively, the --no_t0_correction option can be used to skip T0 correction.")

  if not args.no_mcpl_filtering:
    if not args.wavelength:
      parser.error(f"The --wavelength must be provided for MCPL TOF filtering. Alternatively, the --no_mcpl_filtering option can be used to skip TOF filtering")

  if args.t0_fixed:
    if args.t0_wavelength_rebin:
      parser.error(f"The --t0_fixed option can not be used together with --t0_wavelength_rebin ")

  if args.all_q and args.wfm:
    parser.error(f"The --all_q option can not be used together with --wfm. (Simply because --all_q is not well maintained, but it can be changed on demand.)")

  if (args.sample_xwidth == 0 or args.sample_zheight == 0) and not args.allow_sample_miss:
    parser.error(f"One of the sample sizes is zero. Direct beam simulation also requires the --allow_sample_miss option to be set True.")
  if (args.sample_xwidth < 0 or args.sample_zheight < 0):
    parser.error(f"The sample sizes can not be negative. (For direct beam simulation, set either of the sample sizes to zero.)")

  if args.sample_arguments:
    pairs = args.sample_arguments.split(';')
    for pair in pairs:
        if '=' not in pair:
            parser.error(f"Invalid argument format for --sample_arguments: {pair}. Should be arg=value")

  if args.intensity_factor <= 0.0:
    parser.error(f"The intensity multiplication factor (--intensity_factor) must have a positive value.")

  return args