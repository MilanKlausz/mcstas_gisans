
import argparse
from .instrument_defaults import instrument_defaults, set_instrument_parameters

def zero_to_one(x):
    """Argparser type check function for float number in range [0.0, 1.0]"""
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} not a floating-point literal")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range [0.0, 1.0]")
    return x

def create_argparser():
    parser = argparse.ArgumentParser(description = 'Create Q plots from an .h5 file containing the derived Q values for each outgoing neutron from the BornAgain simulation.')
    parser.add_argument('-f', '--filename', nargs = '*', help = 'Input filename[s].')
    parser.add_argument('-l', '--label', nargs = '*', help = 'Label for input[s].')
    parser.add_argument('-s', '--savename', default='qPlot', required=False, help = 'Output image filename.')
    parser.add_argument('--pdf', action='store_true', help = 'Export figure as pdf.')
    parser.add_argument('--png', action='store_true', help = 'Export figure as png.')
    parser.add_argument('-t', '--experiment_time', default=None, type=int, help = 'Experiment time in seconds to scale the results up to. (e.g. 10800). Must be a positive integer. If a --nxs file reports its own measurement duration, it is compared against this value and a warning (not an error) is printed on a mismatch.')
    parser.add_argument('--background', default=0, type=float, help = 'Add Poisson background to each bin.')
    parser.add_argument('-v', '--verbose', action='store_true', help = 'Verbose output.')
    parser.add_argument('--csv', action='store_true', help = 'Output the resulting histograms in csv format.')
    # The instrument option was moved to instrumentGroup

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

    storedDataParamGroup = parser.add_argument_group('Stored data', 'Use stored data files for plotting or comparison.')
    storedDataParamGroup.add_argument('--nxs', nargs = '*', help = 'Full path to the D22 Nexus file.')
    storedDataParamGroup.add_argument('--nxs_label', nargs = '*', help = 'Label for Nexus input[s]. Must be used together with --nxs if a label is desired. If not provided, the label will be generated from the Nexus file name.')
    storedDataParamGroup.add_argument('--nxs_data_path', type=str, default=None, help='Explicit HDF5 path to the detector data inside the --nxs file(s), e.g. "entry0/data1/MultiDetector1_data". Overrides the default paths that are otherwise tried automatically.')
    storedDataParamGroup.add_argument('--overlay', action='store_true', help = 'Overlay stored data with simulated data.') #TODO isn't it more general than that?
    storedDataParamGroup.add_argument('--normalise_to_nxs', action='store_true', help = 'Normalise simulated data to the total intensity in the Nexus file.')
    storedDataParamGroup.add_argument('--sample_orientation', default=1, choices=[0,1,2], type=float, help = 'Orientation of the sample. 1 - horizontal sample, 0/2 - vertical sample with the beam hitting it from left/right.')
    storedDataParamGroup.add_argument('-a', '--alpha', default=0.0, type=float, help = 'Incident angle on the sample. [deg] (Could be thought of as a sample rotation, but it is actually achieved by an incident beam coordinate transformation.)')
    storedDataParamGroup.add_argument('--wavelength', type=float, default=6.0, help = 'Wavelength in Angstroms.')

    instrumentGroup = parser.add_argument_group('Instrument overrides', 'Override default parameters for the selected instrument.')
    instrumentGroup.add_argument('-i', '--instrument', default='d22', type=str.lower, choices=list(instrument_defaults.keys()), help = 'Instrument (from instruments.py).')
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
    instrumentGroup.add_argument('--instrument_beam_angle', type=float, help='Override the instrument beam angle [deg]. This is the angle of the incident beam relative to the nominal horizontal axis. If not provided, it is automatically calculated from the simulation events using arcsin(mean(v_transverse) / mean(v_total)).')

    nxsInstrumentGroup = parser.add_argument_group('NeXus Instrument overrides', 'Override parameters specifically for the NeXus instrument used to parse the measured data.')
    nxsInstrumentGroup.add_argument('--nxs_instrument_name', type=str.lower, choices=list(instrument_defaults.keys()), help='NeXus instrument name. Defaults to the simulated instrument if not provided.')
    nxsInstrumentGroup.add_argument('--nxs_sample_orientation', choices=[0,1,2], type=float, help = 'Orientation of the sample in the NeXus experiment. 1 - horizontal sample, 0/2 - vertical sample with the beam hitting it from left/right. Defaults to the simulated sample orientation.')
    nxsInstrumentGroup.add_argument('--nxs_instrument_nominal_source_sample_distance', type=float, help='Override NeXus nominal source to sample distance. [m]')
    nxsInstrumentGroup.add_argument('--nxs_instrument_sample_detector_distance', type=float, help='Override NeXus sample to detector distance. [m]')
    nxsInstrumentGroup.add_argument('--nxs_instrument_detector_size', nargs=2, type=float, help='Override NeXus detector dimensions [size_x, size_y] in meters.')
    nxsInstrumentGroup.add_argument('--nxs_instrument_detector_centre_offset', nargs=2, type=float, help='Override NeXus detector centre offset [offset_x, offset_y] in meters.')
    nxsInstrumentGroup.add_argument('--nxs_instrument_detector_pixels', nargs=2, type=int, help='Override NeXus detector pixel counts [pixels_x, pixels_y].')
    nxsInstrumentGroup.add_argument('--nxs_instrument_detector_resolution', nargs=2, type=float, help='Override NeXus detector resolution FWHM [res_x, res_y] in meters.')
    nxsInstrumentGroup.add_argument('--nxs_instrument_tof_instrument', type=str.lower, choices=['true', 'false'], help='Override whether the NeXus instrument is a Time-of-Flight (TOF) instrument.')
    nxsInstrumentGroup.add_argument('--nxs_instrument_t0_monitor_name', type=str, help='Override NeXus t0 monitor name.')
    nxsInstrumentGroup.add_argument('--nxs_instrument_wfm_t0_monitor_name', type=str, help='Override NeXus WFM t0 monitor name.')
    nxsInstrumentGroup.add_argument('--nxs_instrument_wfm_virtual_source_distance', type=float, help='Override NeXus WFM virtual source distance. [m]')
    nxsInstrumentGroup.add_argument('--nxs_instrument_beam_angle', type=float, help='Override the NeXus instrument beam angle [deg].')

    # Data slicing options
    sliceGroup = parser.add_argument_group('Data Slicing', 'Options for slicing event data (e.g. TOF).')
    sliceGroup.add_argument('--wavelength_slice', nargs=2, type=float, metavar=('MIN', 'MAX'), help='Slice TOF event data by wavelength range [angstrom] before plotting.')

    return parser

def parse_args(parser):
    args = parser.parse_args()

    # Apply instrument parameter overrides in instrument_defaults
    set_instrument_parameters(args)

    if args.filename is None and args.nxs is None:
        parser.error('No input file provided! This is only allowed when the --nxs option is used.')

    if args.label and len(args.label) != len(args.filename):
        parser.error(f"The number of labels ({len(args.label)}) doesn't agree with the number of files ({len(args.filename)})")

    if (args.experiment_time is not None) and args.experiment_time <= 0:
        parser.error('The --experiment_time must be a positive integer.')



    if args.normalise_to_nxs and not args.nxs:
        parser.error('The --normalise_to_nxs option can only be used when --nxs is also in use.')

    return args