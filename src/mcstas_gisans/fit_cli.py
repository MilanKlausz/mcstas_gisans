from .run_cli import create_argparser as create_run_parser

def create_fit_parser():
  parser = create_run_parser()

  # Make filename optional for the run parser, since we may only want to view masks without running a simulation
  for action in parser._actions:
    if action.dest == 'filename':
      action.nargs = '?'
      action.default = ''

  scan_group = parser.add_argument_group('Parameter scan options')
  scan_group.add_argument('--scan', action='append', nargs='+', required=False,
                          help='Parameter name followed by values to scan, e.g., --scan radius 10 12 15')
  scan_group.add_argument('--nxs', type=str, nargs='+', required=True, help='Path(s) to experimental NeXus file(s) to match. If multiple files are given (e.g. segmented measurements), their counts are summed; --experiment_time should then be the cumulative experiment time across all given files.')
  scan_group.add_argument('--experiment_time', type=float, default=None, help='Virtual experiment time in seconds for upscaling the simulation. If --nxs specifies multiple files, this should be their cumulative experiment time.')
  scan_group.add_argument('--background', type=float, default=0.0, help='Flat background level added during upscaling.')
  scan_group.add_argument('--poisson_sampling', action='store_true', help='Enable random Poisson noise sampling on the simulated data. (Off by default during scans/fits to ensure deterministic, smooth objective function evaluation for optimizer convergence.)')
  scan_group.add_argument('--output_dir', type=str, default='scan_results', help='Directory to save scan results.')

  scan_plot_group = parser.add_argument_group('Plotting options for scan results')
  scan_plot_group.add_argument('--png', action='store_true', help='Generate comparison PNG plot for each simulation configuration.')
  scan_plot_group.add_argument('--y_plot_range', nargs=2, type=float, help='Plot y range.')
  scan_plot_group.add_argument('--z_plot_range', nargs=2, type=float, help='Plot z range.')
  scan_plot_group.add_argument('--q_min', type=float, default=0.0, help='Minimum Qz value for 1D slice comparison [1/nm].')
  scan_plot_group.add_argument('--q_max', type=float, default=0.0, help='Maximum Qz value for 1D slice comparison [1/nm].')
  scan_plot_group.add_argument('-m', '--intensity_min', default=None, help='Intensity minimum for the 2D q plot colorbar.')


  scan_mask_group = parser.add_argument_group('Masking options to exclude data ranges for the fitness calculation')
  scan_mask_group.add_argument('--mask_view', action='store_true', help='Only view the applied masks on the experimental NeXus data, then exit.')
  scan_mask_group.add_argument('--mask_qy_range', nargs=2, type=float, default=None, help='Qy range to mask out from the fitness calculation (e.g., -0.05 0.05) [1/nm].')
  scan_mask_group.add_argument('--mask_qy_min_cut', type=float, default=None, help='Lower Qy cut option. Any data below this Qy value is disregarded [1/nm].')
  scan_mask_group.add_argument('--mask_qy_max_cut', type=float, default=None, help='Upper Qy cut option. Any data above this Qy value is disregarded [1/nm].')
  scan_mask_group.add_argument('--mask_qz_min_cut', type=float, default=None, help='Lower Qz cut option. Any data below this Qz value is disregarded [1/nm].')
  scan_mask_group.add_argument('--mask_qz_max_cut', type=float, default=None, help='Upper Qz cut option. Any data above this Qz value is disregarded [1/nm].')
  scan_mask_group.add_argument('--mask_exclude_q_box', action='append', nargs=4, type=float, default=None,
                               help='Exclude rectangular Q-region defined by 4 numbers: qy_min qy_max qz_min qz_max [1/nm]. (Can be specified multiple times).')
  scan_mask_group.add_argument('--mask_include_q_box', action='append', nargs=4, type=float, default=None,
                               help='Include rectangular Q-region defined by 4 numbers: qy_min qy_max qz_min qz_max [1/nm]. Applied after exclusions. (Can be specified multiple times).')
  scan_mask_group.add_argument('--simulate_mask_angle_range', action='store_true',
                               help='Calculate minimum simulation angle range enclosing the unmasked detector pixels to optimize performance.')
  scan_mask_group.add_argument('--simulate_mask_angle_range_factor', type=float, default=1.0,
                               help='Expansion factor for --simulate_mask_angle_range (default: 1.0). Use e.g. 1.05 for a 5%% safety margin around the ROI.')
  fit_group = parser.add_argument_group('Automated optimization / fitting options')
  fit_group.add_argument('--fit', action='append', nargs='+', required=False,
                         help='Parameter to fit with initial guess and optional min/max bounds, e.g., --fit radius 51 40 60')
  fit_group.add_argument('--fit_integer', action='append', nargs='+', required=False, default=None,
                         help='Specify parameter names to fit as integers (e.g. --fit_integer layerNumber). These parameters will be constrained to integer values during optimization (rounded for Nelder-Mead and Powell, and natively handled for Differential Evolution).')
  fit_group.add_argument('--optimizer', type=str, default='nelder-mead', choices=['nelder-mead', 'powell', 'differential-evolution'],
                         help='Optimization algorithm to use (default: nelder-mead).')
  fit_group.add_argument('--popsize', type=int, default=15,
                         help='Population size multiplier for Differential Evolution (default: 15). The total population is popsize * number_of_parameters. A smaller value reduces evaluations per generation but reduces search diversity.')
  fit_group.add_argument('--max_evals', type=int, default=10,
                         help='Maximum number of objective function evaluations for the optimizer (default: 10).')
  fit_group.add_argument('--loss_function', type=str, default='reduced_chi2', choices=['reduced_chi2', 'log_residual'],
                         help='Metric to minimize during optimization (default: reduced_chi2).')
  fit_group.add_argument('--xatol', type=float, default=0.01,
                         help='Absolute parameter convergence tolerance. (SciPy default: 1e-4. Suggested for Monte Carlo simulations: 0.01).')
  fit_group.add_argument('--fatol', type=float, default=0.05,
                         help='Absolute loss function convergence tolerance. (SciPy default: 1e-4. Suggested for Monte Carlo simulations: 0.05 matching MC Poisson noise floor).')
  fit_group.add_argument('--gif', action='store_true',
                         help='Generate animated GIF showing the evolution of the fitting process.')

  joint_fit_group = parser.add_argument_group('Joint / Dual-sample fitting options (Secondary Sample)')
  joint_fit_group.add_argument('--nxs2', type=str, default=None,
                               help='Path to secondary experimental NeXus file to match for joint sample fitting.')
  joint_fit_group.add_argument('--fit2', action='append', nargs='+', required=False, default=None,
                               help='Parameter to fit specifically for secondary sample (sample 2), e.g., --fit2 latticeParameter 120 100 130')
  joint_fit_group.add_argument('--fit_common', action='append', nargs='+', required=False, default=None,
                               help='Parameter to fit in common across both samples (sample 1 and 2), e.g., --fit_common radius 51 40 60')
  joint_fit_group.add_argument('--sample_arguments2', type=str, default=None,
                               help='Non-fitted sample arguments specifically for secondary sample (sample 2), e.g., --sample_arguments2 "radius=51;interferenceRange=5"')
  joint_fit_group.add_argument('--filename2', type=str, default=None,
                               help='Optional secondary particle file for sample 2. Defaults to main --filename if omitted.')
  joint_fit_group.add_argument('--intensity_factor2', type=float, default=None,
                               help='Optional secondary intensity factor for sample 2. Defaults to main --intensity_factor if omitted.')
  joint_fit_group.add_argument('--alpha2', type=float, default=None,
                               help='Optional incident angle alpha for sample 2 [deg]. Defaults to main --alpha if omitted.')
  joint_fit_group.add_argument('--experiment_time2', type=float, default=None,
                               help='Optional virtual experiment time for sample 2 [s]. Defaults to main --experiment_time if omitted.')
  joint_fit_group.add_argument('--background2', type=float, default=None,
                               help='Optional flat background level for sample 2. Defaults to main --background if omitted.')

  return parser

def parse_scan_arguments(scan_args):
  scanned_params = {}
  for item in scan_args:
    if len(item) < 2:
      raise ValueError(f"Scan parameter must have at least one value: {item}")
    name = item[0]
    values = []
    for val_str in item[1:]:
      try:
        val = int(val_str)
      except ValueError:
        try:
          val = float(val_str)
        except ValueError:
          val = val_str
      values.append(val)
    scanned_params[name] = values
  return scanned_params

def parse_fit_arguments(fit_args):
  """
  Parses --fit arguments.

  Supported formats per parameter:
    * ``--fit name x0``
    * ``--fit name min max``
    * ``--fit name x0 min max``

  Returns
  -------
  tuple
      param_names: list of parameter names
      x0_list: list of initial values (float)
      bounds_list: list of (min_val, max_val) tuples or None
  """
  param_names = []
  x0_list = []
  bounds_list = []

  for item in fit_args:
    if len(item) < 2:
      raise ValueError(f"Fit parameter must specify at least name and initial value: {item}")
    name = item[0]
    param_names.append(name)

    if len(item) == 2:
      x0 = float(item[1])
      bounds = (None, None)
    elif len(item) == 3:
      b_min = float(item[1])
      b_max = float(item[2])
      x0 = (b_min + b_max) / 2.0
      bounds = (b_min, b_max)
    else:
      x0 = float(item[1])
      b_min = float(item[2])
      b_max = float(item[3])
      bounds = (b_min, b_max)

    x0_list.append(x0)
    bounds_list.append(bounds)

  return param_names, x0_list, bounds_list
