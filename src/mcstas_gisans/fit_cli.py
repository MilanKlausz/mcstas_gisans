import argparse

from .run_cli import create_argparser as create_run_parser
from .nexus_reader import NXS_FORMATS

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
    scan_group.add_argument('--nxs_format', type=str, default='auto', choices=list(NXS_FORMATS), help="Format of the --nxs file(s). image: a 2D detector image [x, y] (e.g. D22, see --nxs_data_path). isis_sans2d: a raw ISIS SANS2D NeXus file with the TOF histograms of the rear tube detector (for --wavelength_slice). auto (default): raw SANS2D files are recognised by raw_data_1/beamline, every other file is read as an image.")
    scan_group.add_argument('--experiment_time', type=float, default=None, help='Measurement time in seconds: the simulated rates are scaled to expected counts over this time before comparison with the measured counts (required for scans and fits). If --nxs specifies multiple files, this should be their cumulative experiment time. If the NeXus file(s) report their own measurement duration, it is compared against this value and a warning (not an error) is printed on a mismatch.')
    scan_group.add_argument('--background', type=float, default=None, help='Fixed flat background [expected counts per detector pixel over --experiment_time] added to the simulated counts (default: 0). Cannot be combined with --fit_background.')
    scan_group.add_argument('--fit_background', action='store_true', help='Fit the flat background instead of setting it with --background (the two cannot be combined): for each simulated pattern, the background [counts per pixel] that minimises --loss_function over the unmasked pixels (with --fit_objective 1d: over the Qy profile of the Qz band, whose bins get the number of summed pixels times the background) is found (a fast 1D minimisation between 0 and the mean measured counts, no extra simulation) and printed with the evaluation. The value is conditional on the simulated pattern (a too intense pattern gets a too low background), so it is meaningful near the best fit. Use it with the poisson_deviance loss: minimising reduced_chi2 overestimates it by ~0.5 counts.')
    scan_group.add_argument('--output_dir', type=str, default='scan_results', help='Directory to save scan results.')

    scan_plot_group = parser.add_argument_group('Plotting options for scan results')
    scan_plot_group.add_argument('--png', action='store_true', help='Generate comparison PNG plot for each simulation configuration.')
    scan_plot_group.add_argument('--y_plot_range', nargs=2, type=float, help='Plot y range.')
    scan_plot_group.add_argument('--z_plot_range', nargs=2, type=float, help='Plot z range.')
    scan_plot_group.add_argument('--q_min', type=float, default=0.0, help='Minimum Qz of the Qz band [1/nm] whose Qy profile (the counts of the unmasked pixels summed over the band, per Qy bin) is shown below the 2D maps of the comparison plots. If --q_min < --q_max, the profiles of the measurement and the simulation are also compared: the 1D losses (poisson_deviance_1d, reduced_chi2_1d, log_residual_1d) are reported with every evaluation, and --fit_objective 1d minimises them.')
    scan_plot_group.add_argument('--q_max', type=float, default=0.0, help='Maximum Qz of the Qz band [1/nm], see --q_min.')
    scan_plot_group.add_argument('--band_exit_angle', nargs=2, type=float, metavar=('LOW', 'HIGH'), default=None,
                                 help='The band of the 1D Qy profile (its losses, --fit_objective 1d and the 1D panel of the plots) defined by the exit angle instead of Qz (instead of --q_min/--q_max): the detector pixels whose exit angle alpha_f above the sample surface has LOW <= alpha_f - alpha <= HIGH [deg], i.e. relative to the specular exit angle alpha (--alpha). The exit angle of a pixel centre is that of the Q conversion (Qz = k (sin alpha_f + sin alpha), with the gravity correction) at the wavelength (--wavelength_selected; for TOF the centre of each --wavelength_slice), so the band follows the specular Qz = 2 k sin(alpha) from slice to slice, and every TOF bin of a slice uses the same pixels. The profile sums the counts of the pixels of the band that are in unmasked Q bins, per Qy bin. Example: one tube (8.1 mm at 12 m: 0.0387 deg) below to six tubes above the specular: --band_exit_angle -0.0387 0.232. Not for joint fits.')
    scan_plot_group.add_argument('-m', '--intensity_min', default=None, help='Intensity minimum for the 2D q plot colorbar.')
    scan_plot_group.add_argument('--split_view', action='store_true', help='Comparison plots (--png, --gif): one 2D Q map with the measurement for Qy < 0 and the simulation for Qy > 0 (common colour scale) instead of two separate maps.')


    scan_mask_group = parser.add_argument_group('Masking options to exclude data ranges for the fitness calculation')
    scan_mask_group.add_argument('--mask_view', action='store_true', help='Only view the applied masks on the experimental NeXus data, then exit.')
    scan_mask_group.add_argument('--mask_view_style', choices=['grey', 'black'], default='grey', help="With --mask_view: show the masked pixels of the masked map as the measured data on a grey scale ('grey', default) or black them out ('black').")
    scan_mask_group.add_argument('--mask_qy_range', nargs=2, type=float, default=None, help='Qy range to mask out from the fitness calculation (e.g., -0.05 0.05) [1/nm].')
    scan_mask_group.add_argument('--mask_qy_min_cut', type=float, default=None, help='Lower Qy cut option. Any data below this Qy value is disregarded [1/nm].')
    scan_mask_group.add_argument('--mask_qy_max_cut', type=float, default=None, help='Upper Qy cut option. Any data above this Qy value is disregarded [1/nm].')
    scan_mask_group.add_argument('--mask_qz_min_cut', type=float, default=None, help='Lower Qz cut option. Any data below this Qz value is disregarded [1/nm].')
    scan_mask_group.add_argument('--mask_qz_max_cut', type=float, default=None, help='Upper Qz cut option. Any data above this Qz value is disregarded [1/nm].')
    scan_mask_group.add_argument('--mask_exclude_q_box', action='append', nargs=4, type=float, default=None,
                                 help='Exclude rectangular Q-region defined by 4 numbers: qy_min qy_max qz_min qz_max [1/nm]. (Can be specified multiple times).')
    scan_mask_group.add_argument('--mask_include_q_box', action='append', nargs=4, type=float, default=None,
                                 help='Include rectangular Q-region defined by 4 numbers: qy_min qy_max qz_min qz_max [1/nm]. Applied after exclusions. (Can be specified multiple times).')
    scan_mask_group.add_argument('--mask_exit_angle_band', nargs=2, type=float, metavar=('LOW', 'HIGH'), default=None,
                                 help='Keep only the detector pixels whose exit angle alpha_f above the sample surface has LOW <= alpha_f - alpha <= HIGH [deg], i.e. a band relative to the specular exit angle alpha (--alpha); applied together with the other --mask_* options. The exit angle of a pixel centre is that of the Q conversion (Qz = k (sin alpha_f + sin alpha), with the gravity correction) at the wavelength (--wavelength_selected; for TOF the centre of each --wavelength_slice). Unlike the Q masks, which are the same for all wavelength slices, the band follows the specular Qz = 2 k sin(alpha) from slice to slice, and every TOF bin of a slice uses the same pixels. Example: one tube (8.1 mm at 12 m: 0.0387 deg) below to six tubes above the specular, without the specular core: --mask_exit_angle_band -0.0387 0.232 --mask_qy_range -0.03 0.03.')
    scan_mask_group.add_argument('--simulate_mask_angle_range', action='store_true',
                                 help='Simulate only the outgoing angles that can reach the unmasked detector pixels (faster), see --simulate_mask_angle_range_factor.')
    scan_mask_group.add_argument('--keep_masked_directions', action='store_true',
                                 help='With --simulate_mask_angle_range: simulate also the outgoing directions whose rays can only reach masked pixels. By default they are skipped (masked in BornAgain, which does not compute them): a direction is simulated if its ray, widened by 2 sigma of the detector resolution along each detector axis, touches an unmasked pixel. The unmasked pixels get the same counts either way (up to the rays smeared by more than 2 sigma, as for the windows); the masked pixels of the simulated image are then mostly empty. The time saved is about the fraction of skipped directions (e.g. the corners of a T-shaped region, the hole of an excluded specular box), printed after the first evaluation.')
    scan_mask_group.add_argument('--simulate_mask_angle_range_factor', type=_angle_range_factor, default='auto',
                                 help="Outgoing angles simulated with --simulate_mask_angle_range. 'auto' (default): every neutron hitting the sample gets its own window of the directions from which its rays can reach the range enclosing the unmasked pixels (as seen from the sample centre), i.e. that range shifted by the neutron's incident horizontal direction, its hit point on the sample and its gravity drop, plus 2 sigma of the detector resolution on every side; the windows have the size of that range (plus the resolution margins), and the outgoing-direction grid covers the window of each neutron. A number: that range scaled by this factor about its centre, simulated for every neutron; a warning is printed if it does not contain the union of the windows (rays reaching the edge of the unmasked region are then missing).")
    tof_group = parser.add_argument_group('Time-of-flight (TOF) instruments', 'The measured and the simulated counts of every detector pixel and TOF bin are compared in wavelength slices: in a slice, the counts of the TOF bins whose centre wavelength is in the slice are converted to (Qy, Qz) at that wavelength and summed into the Q grid of the slice (the pixel edges of the detector at the centre wavelength of the slice). The wavelength of a TOF bin of a pixel is h t / (m (L1 + L2)), with L1 the nominal source-sample distance of the instrument (--instrument_nominal_source_sample_distance; the TOF of the simulated neutrons start at the McStas source, those of the measurement at the proton pulse, and both use this L1) and L2 the sample-pixel distance. The same sample parameters are evaluated for all slices, and the loss is the sum of the losses of the slices (reported per slice as well). The masks (--mask_*), the Qz band (--q_min/--q_max) and --background apply to every slice (the exit-angle bands --mask_exit_angle_band and --band_exit_angle are converted for each slice); --fit_background fits the background of every slice.')
    tof_group.add_argument('--wavelength_slice', action='append', nargs=2, type=float, metavar=('MIN', 'MAX'), default=None,
                           help='Wavelength slice [Angstrom] compared with the measurement (required for TOF instruments; can be given several times). The MCPL input is filtered for the slices (the TOF of the slices over L1, widened by 10%%, unless --input_tof_limits or --no_mcpl_filtering is given), and only the neutrons that can be detected in a TOF bin of a slice are simulated. No T0 correction may be applied to the simulated TOF (--no_t0_correction, or --t0_fixed for a known offset), since they are compared with the raw measured TOF. Requires a measurement with TOF bins (--nxs_format isis_sans2d).')
    tof_group.add_argument('--intensity_factor_spectrum', type=str, default=None, metavar='FILE',
                           help="Wavelength-dependent intensity calibration of the simulation (TOF only): a text file with the columns wavelength [Angstrom] and factor (lines starting with '#' are comments). The simulated count of every (pixel, TOF bin) is multiplied by the factor linearly interpolated at its wavelength (h t / (m (L1 + L2)), as for the slices), in addition to --intensity_factor. The file must cover the wavelength slices. Typically the ratio of the measured to the simulated direct beam (the counts of a direct-beam run per wavelength bin, divided by those of a simulation of the same McStas beam without sample over the same time, both wavelengths from the TOF in the same way), see the fitting guide.")

    fit_group = parser.add_argument_group('Automated optimization / fitting options')
    fit_group.add_argument('--fit', action='append', nargs='+', required=False,
                           help='Parameter to fit with initial guess and optional min/max bounds, e.g., --fit radius 51 40 60')
    fit_group.add_argument('--fit_integer', action='append', nargs='+', required=False, default=None,
                           help='Specify parameter names to fit as integers (e.g. --fit_integer layerNumber). These parameters will be constrained to integer values during optimization (rounded for Nelder-Mead and Powell, and natively handled for Differential Evolution).')
    fit_group.add_argument('--optimizer', type=str, default='nelder-mead', choices=['nelder-mead', 'powell', 'differential-evolution'],
                           help='Optimization algorithm to use (default: nelder-mead).')
    fit_group.add_argument('--popsize', type=int, default=15,
                           help='Population size multiplier for Differential Evolution (default: 15). The total population is popsize * number_of_parameters. A smaller value reduces evaluations per generation but reduces search diversity.')
    fit_group.add_argument('--de_continue', type=str, default=None, metavar='FIT_SUMMARY',
                           help='Differential Evolution only: start a new DE fit (in its own --output_dir) from the population of another DE fit, given by its '
                                'fit_summary.csv or output directory (e.g. a fit of an older mg_fit without fit_state.json). Its evaluations are replayed with the '
                                'update rules of scipy (immediate updating) to recover the population at the point where it stopped; the new fit starts from that '
                                'population without simulating it again, copies the earlier evaluations into its fit_summary.csv and continues their numbers. Use '
                                'the same --fit parameters, bounds, --popsize and --seed as the earlier fit. --max_evals counts the new evaluations only. Not needed '
                                'to continue an unfinished fit in the same --output_dir: that is automatic (see --overwrite).')
    fit_group.add_argument('--max_evals', type=int, default=10,
                           help='Maximum number of objective function evaluations for the optimizer (default: 10).')
    fit_group.add_argument('--loss_function', type=str, default='poisson_deviance', choices=['poisson_deviance', 'reduced_chi2', 'log_residual'],
                           help='Metric to minimize (default: poisson_deviance). poisson_deviance (recommended): Poisson likelihood-ratio statistic per pixel, also known as the Poisson deviance or Cash statistic (m: expected simulated counts incl. background, N: measured counts) with the Monte Carlo uncertainty of the simulation folded in; without Monte Carlo uncertainty it is 2/n*sum[m - N + N*ln(N/m)]; unbiased also at low counts, about 1 for a perfect model at high counts (a warning is printed if the Monte Carlo variance is not small). reduced_chi2: 1/n*sum[(N - m)^2 / (m + sigma_MC^2)], includes the Monte Carlo uncertainty but is biased at low counts. log_residual: mean squared difference of log10 intensities over pixels where both are positive (shape-oriented, ignores counting statistics). See "Choosing the loss function" in the fitting guide of the documentation for the definitions and which one to use.')
    fit_group.add_argument('--fit_objective', type=str, default='2d', choices=['2d', '1d'],
                           help='Comparison whose --loss_function is minimised (fits) and sorts the summaries (fits and scans); --fit_background fits the background for it. 2d (default): the unmasked detector pixels. 1d: the Qy profile of the Qz band [--q_min, --q_max] (requires --q_min < --q_max, or --band_exit_angle), i.e. the counts of the unmasked pixels within the band summed per Qy bin, as in the 1D panel of the comparison plots. If the band is given, the losses of both comparisons are reported (columns poisson_deviance, ... and poisson_deviance_1d, ... of the summary CSV).')
    fit_group.add_argument('--xatol', type=float, default=0.01,
                           help='Parameter convergence tolerance for Nelder-Mead and Powell (not used by Differential Evolution), relative to each parameter\'s scale: its bound range if bounded, otherwise the absolute initial value (default: 0.01, i.e. 1%%).')
    fit_group.add_argument('--fatol', type=float, default=0.05,
                           help='Absolute loss convergence tolerance. Nelder-Mead/Powell: change of the loss; Differential Evolution: spread (standard deviation) of the losses of the population (default: 0.05).')
    fit_group.add_argument('--overwrite', action='store_true',
                           help='Start the fit from scratch even if --output_dir holds an earlier fit: its fit_summary.csv, fit_evaluations.jsonl and '
                                'fit_state.json are removed (the comparison plots are overwritten as they are made). Without --overwrite, mg_fit looks at '
                                '--output_dir first: an unfinished fit with the same settings (e.g. stopped by a job time limit) is CONTINUED automatically '
                                '(its evaluations are retraced from fit_evaluations.jsonl without simulation, then new ones follow; the result is that of an '
                                'uninterrupted run), while a finished fit, an unfinished fit with other settings, or a fit_summary.csv without fit_state.json '
                                'stop mg_fit with a message. The budgets (--max_evals, --polish_max_evals) and the parallel and plot options may change '
                                'between the runs of one fit; without --seed, the seed of the fit is used.')
    fit_group.add_argument('--polish', type=str, default=None, choices=['nelder-mead', 'powell'],
                           help='Differential Evolution only: refine the best point of the DE with this local optimiser (in the same fit: same '
                                '--output_dir and fit_summary.csv, phase column "polish"), with the integer parameters fixed at their DE values. DE '
                                'stops when the losses of its population agree within --fatol, which can leave its best point a few standard errors '
                                'from the bottom of the minimum; the polish finds it (global search, then local refinement). See --polish_max_evals, '
                                '--polish_xatol, --polish_fatol.')
    fit_group.add_argument('--polish_max_evals', type=int, default=100,
                           help='Maximum number of evaluations of the --polish (default: 100).')
    fit_group.add_argument('--polish_xatol', type=float, default=0.002,
                           help='Parameter tolerance of the --polish, relative to the bound range of each parameter (default: 0.002).')
    fit_group.add_argument('--polish_fatol', type=float, default=0.002,
                           help='Loss tolerance of the --polish (absolute change of the loss per pixel; default: 0.002).')
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
