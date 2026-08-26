"""
Time-of-Flight (TOF) Filtering Utilities.

This module provides helper functions to calculate appropriate Time-of-Flight (TOF)
cuts based on the requested neutron wavelength range and the source-to-detector distance.
"""

import sys
from pathlib import Path
from typing import Any, List

from .instrument_defaults import instrument_defaults
from .input_output import print_tof_limits

def get_tof_filtering_limits(args: Any) -> List[float]:
    """
    Get TOF (time-of-flight) limits that can be used for filtering neutrons from
    the MCPL input file.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments. Must contain at least:
        
        * instrument: str
        * no_mcpl_filtering: bool
        * input_tof_limits: list of float or None
        * wavelength: float or None
        * savename: str
        * tof_filtering_figure: str or None
        * filename: str
        * input_wavelength_rebin: float or None
        * input_tof_range_factor: float

    Returns
    -------
    list of float
        A two-element list representing the lower and upper TOF limits.

    Notes
    -----
    The options for filtering are:

    1. No filtering ([-inf, inf])
    2. Using input values (args.input_tof_limits)
    3. Derive limits from a 1D TOF spectrum corresponding to a selected
       wavelength that is retrieved from a 2D McStas TOFLambda_monitor spectrum
       (that is assumed to represent the MCPL file content). The limits are
       defined by fitting a Gaussian function and getting a single FWHM range
       centred around the mean TOF value.
    """
    inst_params = instrument_defaults[args.instrument]
    tof_limits = [float('-inf'), float('inf')]

    # Check if we should apply TOF filtering based on instrument parameters and user arguments
    should_filter = inst_params.get('tof_instrument', False) and not args.no_mcpl_filtering
    has_filtering_args = args.input_tof_limits or args.wavelength

    if should_filter and has_filtering_args:
        if args.input_tof_limits:
            # Use explicitly provided limits
            tof_limits = args.input_tof_limits
        else:
            # Derive limits from 1D TOF spectrum fit
            figure_output = f"{args.savename}_tof_filtering.{args.tof_filtering_figure}" if args.tof_filtering_figure in ['png', 'pdf'] else args.tof_filtering_figure
            mcstas_dir = Path(args.filename).resolve().parent
            
            from .fit_monitor import fit_gaussian_to_mcstas_monitor
            fit = fit_gaussian_to_mcstas_monitor(
                dirname=mcstas_dir,
                monitor=inst_params['mcpl_monitor_name'],
                wavelength=args.wavelength,
                wavelength_rebin=args.input_wavelength_rebin,
                figure_output=figure_output,
                tof_range_factor=args.input_tof_range_factor
            )
            
            # Convert fit mean and fwhm to seconds (assuming input is in milliseconds)
            half_width = fit['fwhm'] * 0.5 * args.input_tof_range_factor
            tof_limits[0] = (fit['mean'] - half_width) * 1e-3
            tof_limits[1] = (fit['mean'] + half_width) * 1e-3
            
            if args.tof_filtering_figure is not None:
                # Terminate the script execution because 'only plotting' has been selected by the user
                print_tof_limits(tof_limits)
                sys.exit()

    return tof_limits