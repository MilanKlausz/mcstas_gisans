import argparse

from .run_cli import create_argparser as create_run_parser

def _angle_range_factor(value):
    """--simulate_mask_angle_range_factor: 'auto' or a positive number."""
    if value == 'auto':
        return value
    try:
        factor = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a positive number, got {value!r}")
    if factor <= 0:
        raise argparse.ArgumentTypeError(f"the factor must be positive, got {factor}")
    return factor

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
    scan_group.add_argument('--nxs_data_path', type=str, default=None, help='Explicit HDF5 path to the detector data inside the --nxs file(s), e.g. "entry0/data1/MultiDetector1_data". Overrides the default paths that are otherwise tried automatically.')
    scan_group.add_argument('--experiment_time', type=float, default=None, help='Measurement time in seconds: the simulated rates are scaled to expected counts over this time before comparison with the measured counts (required for scans and fits). If --nxs specifies multiple files, this should be their cumulative experiment time. If the NeXus file(s) report their own measurement duration, it is compared against this value and a warning (not an error) is printed on a mismatch.')
    scan_group.add_argument('--background', type=float, default=None, help='Fixed flat background [expected counts per detector pixel over --experiment_time] added to the simulated counts (default: 0). Cannot be combined with --fit_background.')
    scan_group.add_argument('--fit_background', action='store_true', help='Fit the flat background instead of setting it with --background (the two cannot be combined): for each simulated pattern, the background [counts per pixel] that minimises --loss_function over the unmasked pixels is found (a fast 1D minimisation between 0 and the mean measured counts, no extra simulation) and printed with the evaluation. The value is conditional on the simulated pattern (a too intense pattern gets a too low background), so it is meaningful near the best fit. Use it with the poisson_deviance loss: minimising reduced_chi2 overestimates it by ~0.5 counts.')
    scan_group.add_argument('--output_dir', type=str, default='scan_results', help='Directory to save scan results.')

    scan_plot_group = parser.add_argument_group('Plotting options for scan results')
    scan_plot_group.add_argument('--png', action='store_true', help='Generate comparison PNG plot for each simulation configuration.')
    scan_plot_group.add_argument('--y_plot_range', nargs=2, type=float, help='Plot y range.')
    scan_plot_group.add_argument('--z_plot_range', nargs=2, type=float, help='Plot z range.')
    scan_plot_group.add_argument('--q_min', type=float, default=0.0, help='Minimum Qz value for 1D slice comparison [1/nm].')
    scan_plot_group.add_argument('--q_max', type=float, default=0.0, help='Maximum Qz value for 1D slice comparison [1/nm].')
    scan_plot_group.add_argument('-m', '--intensity_min', default=None, help='Intensity minimum for the 2D q plot colorbar.')
    scan_plot_group.add_argument('--split_view', action='store_true', help='Comparison plots (--png, --gif): one 2D Q map with the measurement for Qy < 0 and the simulation for Qy > 0 (common colour scale) instead of two separate maps.')


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
                                 help='Simulate only the outgoing angles that can reach the unmasked detector pixels (faster), see --simulate_mask_angle_range_factor.')
    scan_mask_group.add_argument('--simulate_mask_angle_range_factor', type=_angle_range_factor, default='auto',
                                 help="Outgoing angles simulated with --simulate_mask_angle_range. 'auto' (default): every neutron hitting the sample gets its own window of the directions from which its rays can reach the range enclosing the unmasked pixels (as seen from the sample centre), i.e. that range shifted by the neutron's incident horizontal direction, its hit point on the sample and its gravity drop, plus 2 sigma of the detector resolution on every side; the windows have the size of that range (plus the resolution margins), and the outgoing-direction grid covers the window of each neutron. A number: that range scaled by this factor about its centre, simulated for every neutron; a warning is printed if it does not contain the union of the windows (rays reaching the edge of the unmasked region are then missing).")
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
    fit_group.add_argument('--loss_function', type=str, default='poisson_deviance', choices=['poisson_deviance', 'reduced_chi2', 'log_residual'],
                           help='Metric to minimize (default: poisson_deviance). poisson_deviance (recommended): Poisson likelihood-ratio statistic per pixel, also known as the Poisson deviance or Cash statistic (m: expected simulated counts incl. background, N: measured counts) with the Monte Carlo uncertainty of the simulation folded in; without Monte Carlo uncertainty it is 2/n*sum[m - N + N*ln(N/m)]; unbiased also at low counts, about 1 for a perfect model at high counts (a warning is printed if the Monte Carlo variance is not small). reduced_chi2: 1/n*sum[(N - m)^2 / (m + sigma_MC^2)], includes the Monte Carlo uncertainty but is biased at low counts. log_residual: mean squared difference of log10 intensities over pixels where both are positive (shape-oriented, ignores counting statistics). See "Choosing the loss function" in the fitting guide of the documentation for the definitions and which one to use.')
    fit_group.add_argument('--xatol', type=float, default=0.01,
                           help='Parameter convergence tolerance for Nelder-Mead and Powell (not used by Differential Evolution), relative to each parameter\'s scale: its bound range if bounded, otherwise the absolute initial value (default: 0.01, i.e. 1%%).')
    fit_group.add_argument('--fatol', type=float, default=0.05,
                           help='Absolute loss convergence tolerance. Nelder-Mead/Powell: change of the loss; Differential Evolution: spread (standard deviation) of the losses of the population (default: 0.05).')
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
      * ``--fit name min max``   (x0 = midpoint)
      * ``--fit name x0 min max``

    Returns
    -------
    tuple
        param_names: list of parameter names
        x0_list: list of initial values (float)
        bounds_list: list of (min_val, max_val) tuples ((None, None) if unbounded)
    """
    param_names = []
    x0_list = []
    bounds_list = []

    for item in fit_args:
        if len(item) < 2 or len(item) > 4:
            raise ValueError(f"A fit parameter must be given as 'name x0', 'name min max' or 'name x0 min max', got: {item}")
        name = item[0]
        if name in param_names:
            raise ValueError(f"Fit parameter '{name}' is given more than once.")
        param_names.append(name)
        values = [float(v) for v in item[1:]]

        if len(values) == 1:
            x0, bounds = values[0], (None, None)
        elif len(values) == 2:
            bounds = (values[0], values[1])
            x0 = (values[0] + values[1]) / 2.0
        else:
            x0, bounds = values[0], (values[1], values[2])

        if bounds[0] is not None:
            if not bounds[0] < bounds[1]:
                raise ValueError(f"Fit parameter '{name}': the lower bound ({bounds[0]}) must be smaller than the upper bound ({bounds[1]}).")
            if not bounds[0] <= x0 <= bounds[1]:
                raise ValueError(f"Fit parameter '{name}': the initial value {x0} is outside the bounds [{bounds[0]}, {bounds[1]}].")

        x0_list.append(x0)
        bounds_list.append(bounds)

    return param_names, x0_list, bounds_list
