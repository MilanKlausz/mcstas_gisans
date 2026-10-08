"""
Create and run argparse command line interface for the run script
"""
import argparse
from typing import List

from .instrument_defaults import instrument_defaults, required_keys_for_wfm, set_instrument_parameters
from .sample import Sample

builtin_samples: List[str] = Sample.describe_builtin_samples()
builtin_str: str = ', '.join(builtin_samples)
DEFAULT_OUTGOING_DIRECTIONS: int = 20
# Sampling presets: target number of rays a detector pixel collects over the run, counted with the effective number of
# neutrons for their weights (noise of the direction sampling about 1.3/sqrt(rays) per pixel: 13%, 3%, 1.3%)
SAMPLING_PRESETS = {'quick': 100, 'standard': 2000, 'long': 10000}
# Used when no outgoing-direction or sampling option is given: a SAMPLING_PRESETS key, or None for DEFAULT_OUTGOING_DIRECTIONS
DEFAULT_SAMPLING = 'quick'

def create_argparser() -> argparse.ArgumentParser:
    """
    Parse command line arguments for the run script.
    """
    parser = argparse.ArgumentParser(
        description="Calculate DWBA GISANS scattering from a McStas MCPL output. "
                    "The output of the script is a .h5 Scipp file (or files) containing the "
                    "derived Q values (or spatial grid/TOF events for plotting). "
                    "Use the mg_plot script to visualize the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filename', help='Input filename. (Preferably MCPL file from the McStas MCPL_output component, but .dat file from McStas Virtual_output works as well)')
    parser.add_argument('--intensity_factor', default=1.0, type=float, help='A multiplication factor to modify the beam intensity. (Applied to the Monte Carlo weight of each particle in the input file.)')
    parser.add_argument('-i','--instrument', required=True, type=str.lower, choices=list(instrument_defaults.keys()), help='Instrument (defined in instrument_defaults.py).')
    parser.add_argument('-p','--parallel_processes', required=False, type=int, help='Number of processes to be used for parallel processing.')
    parser.add_argument('--no_parallel', default=False, action='store_true', help='Do not use multiprocessing (single process; useful for profiling). Results are identical to a parallel run with the same --seed.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for the Monte Carlo sampling (outgoing-direction jitter, detector resolution). Results are reproducible and independent of the number of parallel processes for a given seed. Default: a random seed, which is printed and stored in the output file. mg_fit uses the same seed for every evaluation.')
    parser.add_argument('--wavelength_selected', type=float, help='Wavelength (mean) in Angstrom selected by the monochromator. Only used for non-time-of-flight instruments.')
    parser.add_argument('--no_gravity', default=False, action='store_true', help='Do not take into account gravity.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Enable verbose logging.')

    bornagainGroup = parser.add_argument_group('BornAgain', 'Control the BornAgain simulation options.')
    bornagainGroup.add_argument('-a', '--alpha', default=0.24, type=float, help='Incident angle on the sample. [deg] (Could be thought of as a sample rotation, but it is actually achieved by an incident beam coordinate transformation.)')
    bornagainGroup.add_argument('-n', '--outgoing_directions', type=int, default=argparse.SUPPRESS, help=f'Number of outgoing directions (both horizontally and vertically) within the sampled angle range of the BornAgain simulation. Overrides the sampling preset (default: none, the grid is chosen by --sampling {DEFAULT_SAMPLING}).')
    bornagainGroup.add_argument('--outgoing_directions_horizontal', type=int, help='Number of outgoing directions in the horizontal direction (with --outgoing_directions_vertical; overrides the sampling preset).')
    bornagainGroup.add_argument('--outgoing_directions_vertical', type=int, help='Number of outgoing directions in the vertical direction.')
    presets_str = ', '.join(f'{name}: {rays}' for name, rays in SAMPLING_PRESETS.items())
    bornagainGroup.add_argument('--sampling', choices=list(SAMPLING_PRESETS.keys()), type=str.lower, help=f'Sampling preset: choose the number of outgoing directions so that every detector pixel collects about this many rays over the run ({presets_str}), from the (effective) number of neutrons hitting the sample and the simulated angle range. The noise of the direction sampling is about 1.3/sqrt(rays) per pixel; the statistical noise of the MCPL file adds to it. Default: {DEFAULT_SAMPLING}. The grid of every neutron is shifted randomly, so any grid is unbiased. The chosen numbers are printed with the options that reproduce them. Cannot be combined with --outgoing_directions/_horizontal/_vertical.')
    bornagainGroup.add_argument('--rays_per_pixel', type=float, help='Custom target number of rays per detector pixel instead of a --sampling preset.')
    bornagainGroup.add_argument('--angle_range', nargs=4, type=float, help='Horizontal min/max and vertical min/max scattering angles covered by the simulation: horiz_min horiz_max vert_min vert_max [deg]')
    bornagainGroup.add_argument('--use_avg_materials', default=False, action='store_true', help='BornAgain - use average materials option: "the refractive properties of material layers are computed by taking the average of the matrix material and the embedded particles".')
    bornagainGroup.add_argument('--specular', default='none', choices=['none', 'include_specular', 'specular_simulation'], type=str.lower, help="Control specular reflection in the simulation. NONE: Disables specular beam intensity in the GISAS ScatteringSimulation (setIncludeSpecular(False)). INCLUDE_SPECULAR: Adds specular beam intensity to the GISAS ScatteringSimulation (setIncludeSpecular(True)): BornAgain puts the reflectivity into the outgoing-direction bin containing the specular direction, replacing the diffuse intensity of that bin, so the specular spot is smeared over one grid bin. SPECULAR_SIMULATION: the reflectivity from a separate SpecularSimulation (with the same average-material, polarisation and analyzer options) as one extra ray per particle hitting the sample, in the exact mirror direction: the specular spot keeps the shape of the beam, independent of the grid (recommended).")
    bornagainGroup.add_argument('--bornagain_number_of_threads', type=int, default=None, help='Number of internal threads BornAgain should use. If None, uses BornAgain default.')
    
    outputGroup = parser.add_argument_group('Output', 'Control the generated outputs. The simulated detector image (with Monte Carlo uncertainties) and the metadata are saved in a Scipp HDF5 (.h5) file, loadable with mg_plot.')
    outputGroup.add_argument('-s', '--savename', default='', required=False, help='Output filename (can be full path).')
    outputGroup.add_argument('--temp_read_chunk_size', type=int, default=1000000, help='Chunk size for reading temporary intermediate files at the end of the simulation (default: 1000000)')

    sampleGroup = parser.add_argument_group('Sample', 'Sample related parameters and options.')
    sampleGroup.add_argument('--model', default="silica_100nm_air", help=(f"BornAgain model to use. Can be: the name of a built-in model (e.g. 'silica_100nm_air'), or a path to custom a Python file defining a sample model. Built-in models (and the BornAgain major versions they are tested with): {builtin_str}"))
    sampleGroup.add_argument('--allow_untested_bornagain_version', default=False, action='store_true', help='Use a built-in sample model also if it has no implementation for the installed BornAgain version (all its implementations declare other BORNAGAIN_VERSIONS) (the implementation for the newest older version is used, with a warning; its results may be wrong). Without this option such a model stops the run with an error. The tested versions of each built-in model are listed with --model.')
    sampleGroup.add_argument('--sample_arguments', help='Input arguments of the sample model in format: "arg1=value1;arg2=value2"')
    sampleGroup.add_argument('--sample_orientation', default=1, choices=[0,1,2], type=int, help='Orientation of the sample, by the direction of its surface normal (looking along the beam): 1 - horizontal sample, normal up (reflection goes up); 0 - vertical sample, normal pointing right (reflection goes right, towards lower raw detector x index); 2 - vertical sample, normal pointing left (reflection goes left, towards higher raw detector x index).')
    sampleGroup.add_argument('--sample_size_y', default=0.06, type=float, help='Size of sample perpendicular to beam (along y-axis in BornAgain geometry). [m]')
    sampleGroup.add_argument('--sample_size_x', default=0.08, type=float, help='Size of sample along the beam (along x-axis in BornAgain geometry). [m]')
    sampleGroup.add_argument('--allow_sample_miss', default=False, action='store_true', help='Allow incident neutrons to miss the sample, and be directly propagated to the detector surface. This option can be used to simulate overillumination, or direct beam simulation by also setting one of the sample sizes to zero.')

    polarizationGroup = parser.add_argument_group('Polarization', 'Control polarization and analyzer settings.')
    polarizationGroup.add_argument('--use_polarization', default=False, action='store_true', help='Simulate polarized scattering.')
    polarizationGroup.add_argument('--analyzer_direction', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='The polarization analysis direction (typically a unit vector).')
    polarizationGroup.add_argument('--analyzer_efficiency', type=float, default=1.0, help='The polarization efficiency of the analyzer, representing the polarizing power or the ability to select a specific spin state (0 <= eta <= 1).')
    polarizationGroup.add_argument('--analyzer_transmission', type=float, default=0.5, help='The total transmission factor of the analyzer (fraction of beam intensity that passes through, 0 <= T <= 0.5).')

    mcplFilteringGroup = parser.add_argument_group('MCPL filtering', 'Parameters and options to control which neutrons are used from the MCPL input file. By default no filtering is applied, but if a (central) wavelength is provided, an accepted TOF range is defined based on a McStas TOFLambda monitor (defined as mcpl_monitor_name for each instrument in instrument_defaults.py) that is assumed to correspond to the input MCPL file. The McStas monitor is looked for in the directory of the MCPL input file, and after fitting a Gaussian function, neutrons within a single FWHM range centred around the selected wavelength are used for the BornAgain simulation.')
    mcplFilteringGroup.add_argument('-w', '--wavelength', type=float, default=None, help='Central wavelength used for filtering based on the McStas TOFLambda monitor. (Also used for t0 correction.)')
    mcplFilteringGroup.add_argument('--input_tof_range_factor', default=1.0, type=float, help='Modify the accepted TOF range of neutrons by this multiplication factor.')
    mcplFilteringGroup.add_argument('--input_wavelength_rebin', default=1, type=int, help='Rebin the TOFLambda monitor along the wavelength axis by the provided factor (only if no extrapolation is needed).')
    mcplFilteringGroup.add_argument('--input_tof_limits', nargs=2, type=float, help='TOF limits for selecting neutrons from the MCPL file [millisecond]. When provided, fitting to the McStas monitor is not attempted.')
    mcplFilteringGroup.add_argument('--input_weight_limit', type=float, default=0.0, help='Monte Carlo particle weight limit to exclude particles of low importance.')
    mcplFilteringGroup.add_argument('--no_mcpl_filtering', action='store_true', help='Disable MCPL TOF filtering. Use all neutrons from the MCPL input file.')
    mcplFilteringGroup.add_argument('--tof_filtering_figure', default=None, choices=['show', 'png', 'pdf'], help='Show or save the figure of the selected input TOF range and exit without doing the simulation. Only works with McStas monitor fitting.')

    t0correctionGroup = parser.add_argument_group('T0 correction', 'Parameters and options to control t0 TOF correction. Currently only works if the wavelength parameter in the MCPL filtering is provided.')
    t0correctionGroup.add_argument('--t0_fixed', default=None, type=float, help='Fix t0 correction value that is subtracted from the neutron TOFs. [s]')
    t0correctionGroup.add_argument('--t0_wavelength_rebin', default=None, type=int, help='Rebinning factor for the McStas TOFLambda monitor based t0 correction. Rebinning is applied along the wavelength axis. Only integer divisors are allowed.')
    t0correctionGroup.add_argument('--wfm', default=False, action='store_true', help='Wavelength Frame Multiplication (WFM) mode.')
    t0correctionGroup.add_argument('--no_t0_correction', action='store_true', help='Disable t0 correction. (Allows using McStas simulations which lack the supported monitors.)')
    t0correctionGroup.add_argument('--t0_correction_figure', default=None, choices=['show', 'png', 'pdf'], help='Show or save the figure of the t0 correction and exit without doing the simulation. Only works with McStas monitor fitting.')

    instrumentGroup = parser.add_argument_group('Instrument overrides', 'Override default parameters for the selected instrument.')
    instrumentGroup.add_argument('--instrument_nominal_source_sample_distance', type=float, help='Override nominal source to sample distance. [m]')
    instrumentGroup.add_argument('--instrument_sample_detector_distance', type=float, help='Override sample to detector distance. [m]')
    instrumentGroup.add_argument('--instrument_detector_size', nargs=2, type=float, help='Override detector dimensions [size_x, size_y] in meters.')
    instrumentGroup.add_argument('--instrument_detector_centre_offset', nargs=2, type=float, help='Detector centre position [offset_x, offset_y] in metres relative to the undeflected nominal beam axis through the sample (NeXus frame: x horizontal, y up). A property of the detector position only (independent of sample orientation and wavelength); determine it with mg_beam_centre_correction.')
    instrumentGroup.add_argument('--instrument_detector_pixels', nargs=2, type=int, help='Override detector pixel counts [pixels_x, pixels_y].')
    instrumentGroup.add_argument('--instrument_detector_resolution', nargs=2, type=float, help='Override detector resolution FWHM [res_x, res_y] in meters.')
    instrumentGroup.add_argument('--instrument_tof_instrument', type=str.lower, choices=['true', 'false'], help='Override whether the instrument is a Time-of-Flight (TOF) instrument.')
    instrumentGroup.add_argument('--instrument_t0_monitor_name', type=str, help='Override t0 monitor name.')
    instrumentGroup.add_argument('--instrument_wfm_t0_monitor_name', type=str, help='Override WFM t0 monitor name.')
    instrumentGroup.add_argument('--instrument_wfm_virtual_source_distance', type=float, help='Override WFM virtual source distance. [m]')
    instrumentGroup.add_argument('--instrument_beam_angle', type=float, help='Override the beam angle. Angle of the incident beam above the nominal beam axis [deg], in the plane of incidence, positive towards the sample surface normal (opposite sign to the former beam declination angle). It must describe the mean direction of the simulated (MCPL) beam at the sample; it cancels in the reduction of measured data. If not provided, defaults to 0.0 (or the instrument\'s configured default in instrument_defaults.py). An independent estimate from the MCPL file\'s particle velocities is printed and compared against the value actually used, purely as a sanity check (it is never used as a fallback value, since it must match whatever value was assumed for mg_beam_centre_correction, which has no MCPL data to estimate it from).')
    instrumentGroup.add_argument('--instrument_beam_declination_angle', type=float, default=None, help=argparse.SUPPRESS)  # removed, see parse_args
    instrumentGroup.add_argument('--nexus_y_shift', type=float, default=0.0, help='Shift the beam slightly upwards (in NeXus frame) to ensure it hits the sample surface. E.g. 0.0065')

    return parser

def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse arguments from the command line and validate their combinations.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The configured argument parser.

    Returns
    -------
    argparse.Namespace
        The parsed and validated arguments.
    """
    args: argparse.Namespace = parser.parse_args()
    
    # Validate outgoing directions options
    has_outgoing_directions: bool = hasattr(args, 'outgoing_directions')
    if has_outgoing_directions and (args.outgoing_directions_horizontal is not None or args.outgoing_directions_vertical is not None):
        parser.error("Cannot specify --outgoing_directions together with --outgoing_directions_horizontal or --outgoing_directions_vertical")
    has_explicit_directions = has_outgoing_directions or args.outgoing_directions_horizontal is not None or args.outgoing_directions_vertical is not None
    if args.sampling is not None and args.rays_per_pixel is not None:
        parser.error("Cannot specify --sampling together with --rays_per_pixel")
    if (args.sampling is not None or args.rays_per_pixel is not None) and has_explicit_directions:
        parser.error("Cannot specify --sampling or --rays_per_pixel together with --outgoing_directions, --outgoing_directions_horizontal or --outgoing_directions_vertical")
    if args.rays_per_pixel is not None and args.rays_per_pixel <= 0:
        parser.error("--rays_per_pixel must be positive")
    if (args.outgoing_directions_horizontal is not None) != (args.outgoing_directions_vertical is not None):
        parser.error("Both --outgoing_directions_horizontal and --outgoing_directions_vertical must be specified together")
    if args.sampling is None and args.rays_per_pixel is None and not has_explicit_directions and DEFAULT_SAMPLING is not None:
        args.sampling = DEFAULT_SAMPLING
    if args.sampling is not None:
        args.rays_per_pixel = SAMPLING_PRESETS[args.sampling]
    if args.rays_per_pixel is not None:
        args.outgoing_directions = None  # set from the particles by set_outgoing_directions_from_sampling
    elif not has_outgoing_directions and args.outgoing_directions_horizontal is None and args.outgoing_directions_vertical is None:
        args.outgoing_directions = DEFAULT_OUTGOING_DIRECTIONS
    elif not has_outgoing_directions:
        args.outgoing_directions = None

    if args.instrument_beam_declination_angle is not None:
        parser.error("--instrument_beam_declination_angle was renamed to --instrument_beam_angle, with the OPPOSITE sign "
                     f"(positive = beam rising towards the sample normal): use --instrument_beam_angle {-args.instrument_beam_declination_angle}")
    if args.parallel_processes is not None and args.parallel_processes < 1:
        parser.error("--parallel_processes must be at least 1.")
    if args.seed is None:
        import numpy as np
        args.seed = int(np.random.SeedSequence().entropy % 2**63)
    print(f"Random seed: {args.seed}")

    # Resolve the instrument parameters (defaults + CLI overrides) once: args.instrument_params
    instr_params = set_instrument_parameters(args)

    if args.wfm and any(key not in instr_params for key in required_keys_for_wfm):
        parser.error(f"wfm option is not enabled for the {args.instrument} instrument. Set the required instrument parameters in instrument_defaults.py.")

    if args.tof_filtering_figure:
        if not args.wavelength:
            parser.error("The --tof_filtering_figure option can only be used if a central wavelength (--wavelength) for fitting is provided.")
        if args.input_tof_limits:
            parser.error("The --tof_filtering_figure option can not be used when the TOF range is provided with --input_tof_limits.")
        if args.no_mcpl_filtering:
            parser.error("The --tof_filtering_figure option can not be used when no TOF filtering is selected with --no_mcpl_filtering.")

    if instr_params['tof_instrument']: # tof instrument
        if args.wavelength_selected:
            parser.error("The --wavelength_selected parameter should not be used for TOF instruments. Use the --wavelength parameter instead.")
    else:
        if args.wavelength:
            parser.error("The --wavelength parameter should not be used for non-TOF instruments. Use the --wavelength_selected parameter instead.")
        if not args.wavelength_selected:
            parser.error("For non-TOF instruments the --wavelength_selected parameter is required.")

    if args.no_t0_correction:
        if args.t0_fixed is not None:
            parser.error("The --no_t0_correction option can not be used together with --t0_fixed.")
        if args.t0_wavelength_rebin is not None:
            parser.error("The --no_t0_correction option can not be used together with --t0_wavelength_rebin.")
        if args.wfm:
            parser.error("The --no_t0_correction option can not be used together with --wfm.")
    elif instr_params['tof_instrument']:
        if not args.wavelength:
            parser.error("The --wavelength must be provided for T0 correction. Alternatively, the --no_t0_correction option can be used to skip T0 correction.")

    if not args.no_mcpl_filtering and instr_params['tof_instrument']:
        if not args.wavelength and not args.input_tof_limits:
            parser.error("The --wavelength or --input_tof_limits must be provided for MCPL TOF filtering. Alternatively, the --no_mcpl_filtering option can be used to skip TOF filtering.")

    if instr_params['tof_instrument']:
        name = args.instrument
        if not args.no_mcpl_filtering and not args.input_tof_limits and 'mcpl_monitor_name' not in instr_params:
            parser.error(f"The '{name}' instrument defines no 'mcpl_monitor_name' (McStas TOFLambda monitor for MCPL TOF filtering): "
                         "use --input_tof_limits or --no_mcpl_filtering.")
        if not args.no_t0_correction and args.t0_fixed is None and not args.wfm and 't0_monitor_name' not in instr_params:
            parser.error(f"The '{name}' instrument defines no 't0_monitor_name' (McStas TOFLambda monitor for T0 correction): "
                         "use --t0_fixed or --no_t0_correction.")

    if args.t0_fixed is not None:
        if args.t0_wavelength_rebin is not None:
            parser.error("The --t0_fixed option can not be used together with --t0_wavelength_rebin.")

    if (args.sample_size_y == 0 or args.sample_size_x == 0) and not args.allow_sample_miss:
        parser.error("One of the sample sizes is zero. Direct beam simulation also requires the --allow_sample_miss option to be set True.")
    if (args.sample_size_y < 0 or args.sample_size_x < 0):
        parser.error("The sample sizes can not be negative. (For direct beam simulation, set either of the sample sizes to zero.)")

    if args.sample_arguments:
        pairs = [p for p in args.sample_arguments.split(';') if p.strip()]
        for pair in pairs:
            if '=' not in pair:
                parser.error(f"Invalid argument format for --sample_arguments: {pair}. Should be arg=value.")

    if args.intensity_factor <= 0.0:
        parser.error("The intensity multiplication factor (--intensity_factor) must have a positive value.")

    # Validate analyzer parameters
    if not (0.0 <= args.analyzer_transmission <= 0.5):
        parser.error(f"analyzer_transmission must be between 0.0 and 0.5 (got {args.analyzer_transmission}).")
    if not (0.0 <= args.analyzer_efficiency <= 1.0):
        parser.error(f"analyzer_efficiency must be between 0.0 and 1.0 (got {args.analyzer_efficiency}).")

    direction_norm = (args.analyzer_direction[0]**2 + args.analyzer_direction[1]**2 + args.analyzer_direction[2]**2) ** 0.5
    bloch_vector_len = abs(args.analyzer_efficiency) * direction_norm
    if bloch_vector_len > 1.0:
        parser.error(
            f"The analyzer Bloch vector (efficiency * direction) must have a length <= 1.0. "
            f"Current length: {bloch_vector_len:.4f} (efficiency: {args.analyzer_efficiency}, direction norm: {direction_norm:.4f})."
        )

    return args