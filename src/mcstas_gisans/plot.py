#!/usr/bin/env python3

"""
Main plotting script to create 2D/1D Q plots from simulation results
"""

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any, Dict

from .plotting_utils import plot_q_1d, log_plot_2d, create_2d_histogram, extract_range_to_1d, show_or_save
from .experiment_time import upscale_simple
from .input_output import load_scipp_file

def get_plot_ranges(datasets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]], 
                    y_plot_range: Optional[List[float]], 
                    z_plot_range: Optional[List[float]]) -> Tuple[List[float], List[float], float, float]:
    """
    Get plot ranges. 
    
    Return ranges if provided, otherwise find the minimum and maximum from the datasets.
    
    Parameters
    ----------
    datasets : list of tuple
        List of dataset tuples, each containing (hist, hist_error, y_edges, z_edges, label).
    y_plot_range : list of float, optional
        Predefined Qy plot range [min, max].
    z_plot_range : list of float, optional
        Predefined Qz plot range [min, max].

    Returns
    -------
    tuple
        y_plot_range, z_plot_range, min_value, max_value
    """
    if not y_plot_range:
        y_edge_min = min([y_edges[0] for _, _, y_edges, _, _ in datasets])
        y_edge_max = max([y_edges[-1] for _, _, y_edges, _, _ in datasets])
        y_plot_range = [y_edge_min, y_edge_max]
    if not z_plot_range:
        z_edge_min = min([z_edges[0] for _, _, _, z_edges, _ in datasets])
        z_edge_max = max([z_edges[-1] for _, _, _, z_edges, _ in datasets])
        z_plot_range = [z_edge_min, z_edge_max]
    min_value = min([float(hist.min().min()) for hist, _, _, _, _ in datasets])
    max_value = max([float(hist.max().max()) for hist, _, _, _, _ in datasets])
    return y_plot_range, z_plot_range, min_value, max_value

def get_overlay_plot_axes(column: int = 2) -> Tuple[List[plt.Axes], plt.Axes]:
    """
    Get axes for special subplot layout for dataset comparison. 
    
    Create a grid of subplots, replacing the bottom row with a single larger subplot.

    Parameters
    ----------
    column : int, optional
        Number of columns for the top row, by default 2.

    Returns
    -------
    tuple
        Tuple containing (axes_top, axes_bottom) where axes_top is a list of Axes 
        for the top row and axes_bottom is the single Axes for the bottom row.
    """
    fig, axes = plt.subplots(2, column, figsize=(16, 12))

    #Replace the bottom row of the grid with a single new subplot
    gs = axes[1, 0].get_gridspec() #Get GridSpec from the bottom left subplot
    for i in range(column): #Remove all bottom subplots
        axes[1, i].remove()
    axes_bottom = fig.add_subplot(gs[1:, :]) #cover the row with a new subplot

    axes = axes.flatten()
    axes_top = [axes[i] for i in range(column)]
    return axes_top, axes_bottom

def _extract_metadata_from_scipp(scipp_da_meta: Any) -> Dict[str, Any]:
    """
    Helper to extract metadata from a Scipp DataArray.
    """
    import scipp as sc
    metadata = {}
    if scipp_da_meta is not None:
        if isinstance(scipp_da_meta, sc.DataGroup) and 'instrument' in scipp_da_meta:
            inst = scipp_da_meta['instrument']
            if 'sample_orientation' in inst:
                metadata['sample_orientation'] = inst['sample_orientation'].value
            if 'alpha_inc_deg' in inst:
                metadata['alpha'] = inst['alpha_inc_deg'].value
            if 'beam_angle' in inst:
                metadata['beam_angle'] = inst['beam_angle'].value
            if 'detector_centre_offset_x' in inst and 'detector_centre_offset_y' in inst:
                metadata['instrument_detector_centre_offset'] = [
                    inst['detector_centre_offset_x'].value,
                    inst['detector_centre_offset_y'].value
                ]
            if 'name' in inst:
                name = inst['name'].value
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                metadata['instrument_name'] = name
    return metadata

def setup_global_instrument(args: Any) -> Tuple[Optional[Any], str, float, Any]:
    """
    Setup the global instrument based on provided arguments and NeXus/Scipp metadata.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    tuple
        (instrument, instr_name, alpha, sample_orientation)
    """
    from .instrument import Instrument
    from .instrument_defaults import instrument_defaults

    if getattr(args, 'filename', None):
        for filename in args.filename:
            if filename.endswith('.h5'):
                scipp_da_meta = load_scipp_file(filename)
                metadata = _extract_metadata_from_scipp(scipp_da_meta)

                cli_instrument = getattr(args, 'instrument_name', getattr(args, 'instrument', 'd22'))
                instr_name = metadata.get('instrument_name', cli_instrument)
                if instr_name == 'unknown':
                    instr_name = cli_instrument

                if '--instrument' in sys.argv or '-i' in sys.argv:
                    instr_name = cli_instrument

                alpha = metadata.get('alpha', args.alpha)
                if '--alpha' in sys.argv or '-a' in sys.argv:
                    alpha = args.alpha

                sample_orientation = metadata.get('sample_orientation', args.sample_orientation)
                if '--sample_orientation' in sys.argv:
                    sample_orientation = args.sample_orientation

                beam_angle = metadata.get('beam_angle', 0.0)
                if getattr(args, 'instrument_beam_angle', None) is not None:
                    beam_angle = args.instrument_beam_angle

                centre_offset = metadata.get('instrument_detector_centre_offset', None)
                if getattr(args, 'instrument_detector_centre_offset', None) is not None:
                    centre_offset = args.instrument_detector_centre_offset

                instr_params = instrument_defaults[instr_name]
                instr_params['beam_angle'] = beam_angle
                if centre_offset is not None:
                    instr_params['detector']['direct_beam_centre_offset'] = centre_offset

                return Instrument(instr_params, alpha, args.wavelength, sample_orientation), instr_name, alpha, sample_orientation
    return None, getattr(args, 'instrument_name', getattr(args, 'instrument', 'd22')), args.alpha, args.sample_orientation

def _load_nexus_datasets(args: Any, instr_name: str, alpha: float, sample_orientation: Any) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]], float]:
    """
    Load Nexus datasets based on arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    instr_name : str
        Instrument name.
    alpha : float
        Incident angle alpha.
    sample_orientation : Any
        Sample orientation.
        
    Returns
    -------
    tuple
        (datasets, nxs_sum) where datasets is a list of dataset tuples.
    """
    datasets = []
    nxs_sum = 0.0
    if args.nxs:
        nxs_labels = args.nxs_label if args.nxs_label else args.nxs #default to filename if no label provided
        for nxs_filename, nxs_label in zip(args.nxs, nxs_labels):
            from .instrument_defaults import get_nxs_instrument_parameters
            from .instrument import Instrument
            nxs_instr_params = get_nxs_instrument_parameters(args, default_instr_name=instr_name)
            nxs_sample_orient = sample_orientation
            if getattr(args, 'nxs_sample_orientation', None) is not None:
                nxs_sample_orient = args.nxs_sample_orientation
            nxs_instrument = Instrument(nxs_instr_params, alpha, args.wavelength, nxs_sample_orient)

            from .nexus_reader import read_nexus_data, warn_if_duration_mismatch
            hist, hist_error, y_edges, z_edges = read_nexus_data(nxs_filename, nxs_instrument, data_path=getattr(args, 'nxs_data_path', None))
            warn_if_duration_mismatch([nxs_filename], getattr(args, 'experiment_time', None), label=nxs_filename)
            nxs_sum = np.sum(hist)
            if args.verbose:
                print(f"{nxs_filename} sum: {nxs_sum}")
            datasets.append((hist, hist_error, y_edges, z_edges, nxs_label))
    return datasets, nxs_sum

def _load_sim_datasets(args: Any, global_instrument: Any, instr_name: str, alpha: float, sample_orientation: Any, nxs_sum: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """
    Load simulation datasets from h5/npz files.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    global_instrument : Instrument
        The global instrument instance.
    instr_name : str
        Instrument name.
    alpha : float
        Incident angle alpha.
    sample_orientation : Any
        Sample orientation.
    nxs_sum : float
        Sum of nexus data for normalisation.
        
    Returns
    -------
    list
        List of dataset tuples.
    """
    datasets = []
    if args.filename:
        labels = args.label if args.label else args.filename #default to filename if no label is provided
        from .instrument import Instrument
        for filename, label in zip(args.filename, labels):
            if filename.endswith('.h5'):
                scipp_da = load_scipp_file(filename)
                import scipp as sc
                
                from .instrument_defaults import instrument_defaults
                sim_instr_params = copy.deepcopy(instrument_defaults[instr_name])
                has_offset = False

                if isinstance(scipp_da, sc.DataGroup) and 'instrument' in scipp_da:
                    dg_inst = scipp_da['instrument']
                    if 'detector_centre_offset_x' in dg_inst:
                        sim_instr_params['detector']['direct_beam_centre_offset_x_nexus'] = dg_inst['detector_centre_offset_x'].value
                        has_offset = True
                    if 'detector_centre_offset_y' in dg_inst:
                        sim_instr_params['detector']['direct_beam_centre_offset_y_nexus'] = dg_inst['detector_centre_offset_y'].value
                        has_offset = True
                    scipp_da = scipp_da['data']  # Extract underlying DataArray for plotting
                else:
                    raise ValueError("The loaded file does not appear to be a valid mcstas_gisans sc.DataGroup output.")

                if has_offset:
                    instrument = Instrument(sim_instr_params, alpha, args.wavelength, sample_orientation)
                else:
                    instrument = global_instrument

                scipp_da = instrument.compute_q_scipp(scipp_da)

                y_edges, z_edges = instrument.get_q_pixel_limits(args.wavelength)
                y_min, y_max = (args.y_plot_range[0], args.y_plot_range[1]) if getattr(args, 'y_plot_range', None) else (y_edges[0], y_edges[-1])
                z_min, z_max = (args.z_plot_range[0], args.z_plot_range[1]) if getattr(args, 'z_plot_range', None) else (z_edges[0], z_edges[-1])
                bins_y, bins_z = len(y_edges) - 1, len(z_edges) - 1

                import scipp as sc

                # Bin into Qy, Qz
                if scipp_da.bins is not None:
                    # For TOF / event data
                    if getattr(args, 'wavelength_slice', None) is not None:
                        min_w = args.wavelength_slice[0]
                        max_w = args.wavelength_slice[1]
                        scipp_da = scipp_da.bin(wavelength=sc.array(dims=['wavelength'], values=[min_w, max_w], unit='angstrom'))

                    scipp_da = scipp_da.bins.concat()

                    scipp_binned = scipp_da.bin(
                        Qy=sc.linspace(dim='Qy', start=y_min/10, stop=y_max/10, num=bins_y + 1, unit='1/angstrom'),
                        Qz=sc.linspace(dim='Qz', start=z_min/10, stop=z_max/10, num=bins_z + 1, unit='1/angstrom')
                    )
                    scipp_hist = scipp_binned.bins.sum()
                    scipp_hist = scipp_hist.transpose(['Qy', 'Qz'])
                    hist = scipp_hist.values
                    if scipp_hist.variances is not None:
                        hist_error = np.sqrt(scipp_hist.variances)
                    else:
                        hist_error = np.sqrt(hist)

                    y_edges = scipp_hist.coords['Qy'].values * 10.0
                    z_edges = scipp_hist.coords['Qz'].values * 10.0
                else:
                    # For non-TOF / flattened pixel data, preserve the exact pixel grid to match NeXus perfectly
                    raw_hist = scipp_da.values.reshape((instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus))
                    if scipp_da.variances is not None:
                        raw_err = np.sqrt(scipp_da.variances.reshape((instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus)))
                    else:
                        raw_err = np.sqrt(raw_hist)

                    hist = instrument.detector.coords.rotate_detector_image(raw_hist)
                    hist_error = instrument.detector.coords.rotate_detector_image(raw_err)
                    y_edges, z_edges = instrument.get_q_pixel_limits(args.wavelength)

            elif filename.endswith('.npz'):
                # Legacy NPZ support
                data = np.load(filename)
                hist = data['hist']
                if 'error' in data:
                    hist_error = data['error']
                else:
                    hist_error = np.sqrt(hist)

                if len(hist.shape) == 3:
                    hist = np.sum(hist, axis=0)
                    hist_error = np.sqrt(np.sum(hist_error**2, axis=0))

                y_edges = data['yEdges']
                z_edges = data['zEdges']
            else:
                sys.exit("Unsupported file extension. Only .h5 and .npz files are supported.")
            
            if args.experiment_time:
                hist, hist_error = upscale_simple(hist, hist_error, args.experiment_time, args.background)

            hist_sum = np.sum(hist)
            if args.verbose:
                print(f"{filename} sum: {hist_sum}")
            if args.normalise_to_nxs and nxs_sum > 0:
                hist *= nxs_sum / hist_sum #normalise total intensity of the sim to the nxs data
                hist_error *= nxs_sum / hist_sum
            datasets.append((hist, hist_error, y_edges, z_edges, label))

            if args.csv:
                csv_filename = f"{filename.rsplit('.', 1)[0]}.csv"
                np.savetxt(csv_filename, hist, delimiter=',')
                print(f"Created {csv_filename}")
    return datasets

def get_datasets(args: Any) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """
    Prepare the datasets to be plotted.
    
    Loads data from NeXus measurement files (.nxs) or McStas/BornAgain simulation
    results (.h5). For TOF simulations, data is loaded as Scipp event data arrays,
    Q-values are calculated per event using the instrument geometry, and binned
    into 2D histograms.
    Scales intensities to experiment time if required.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    list
        List of dataset tuples.
    """
    global_instrument, instr_name, alpha, sample_orientation = setup_global_instrument(args)

    if global_instrument is None:
        from .instrument import Instrument
        from .instrument_defaults import instrument_defaults
        instr_name = getattr(args, 'instrument_name', getattr(args, 'instrument', 'd22'))
        instr_params = instrument_defaults[instr_name]
        beam_angle = getattr(args, 'instrument_beam_angle', 0.0)
        if beam_angle is None: beam_angle = 0.0
        instr_params['beam_angle'] = beam_angle
        alpha = args.alpha
        sample_orientation = args.sample_orientation
        global_instrument = Instrument(instr_params, alpha, args.wavelength, sample_orientation)

    nxs_datasets, nxs_sum = _load_nexus_datasets(args, instr_name, alpha, sample_orientation)
    sim_datasets = _load_sim_datasets(args, global_instrument, instr_name, alpha, sample_orientation, nxs_sum)
    
    return nxs_datasets + sim_datasets

def _plot_differences(args: Any, 
                      dataset_index: int, 
                      datasets: List[Any], 
                      all_1d_values: List[np.ndarray], 
                      all_1d_errors: List[np.ndarray], 
                      axes_top: List[plt.Axes], 
                      axes_bottom: plt.Axes, 
                      y_bins: np.ndarray, 
                      y_plot_range: List[float], 
                      z_plot_range: List[float], 
                      line_colors: List[str]) -> None:
    """
    Plot differences between two datasets.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    dataset_index : int
        Index of the current dataset.
    datasets : list
        List of all datasets.
    all_1d_values : list of ndarray
        Extracted 1D values for each dataset.
    all_1d_errors : list of ndarray
        Extracted 1D errors for each dataset.
    axes_top : list of Axes
        Axes for the 2D plots.
    axes_bottom : Axes
        Axes for the 1D plots.
    y_bins : ndarray
        Qy bin centers.
    y_plot_range : list of float
        Qy plot range.
    z_plot_range : list of float
        Qz plot range.
    line_colors : list of str
        List of colors for the datasets.
    """
    if args.plot_differences > 0 and dataset_index == 1:
        meas = all_1d_values[0]
        meas_err = all_1d_errors[0]
        sim = all_1d_values[1]
        sim_err = all_1d_errors[1]
        res_err = np.sqrt(meas_err**2 + sim_err**2)

        ax_int = axes_bottom
        ax_rel = ax_int.twinx()

        with np.errstate(divide='ignore', invalid='ignore'):
            if args.plot_differences == 1:
                rel_abs_diff = np.abs(sim - meas) / meas
                ax_rel.plot(y_bins, rel_abs_diff, color='gray', linestyle='dashdot', linewidth=1.5, label='Relative absolute difference')
                ax_rel.set_ylabel("Relative absolute difference")
                ax_rel.set_ylim(0, 1.6)
            if args.plot_differences == 2:
                rel_diff = (sim - meas) / meas
                ax_rel.plot(y_bins, rel_diff, color='gray', linestyle='dashdot', linewidth=1.5, label='Relative difference')
                ax_rel.set_ylabel("Relative difference")
                ax_rel.set_ylim(-1.6, 1.6)
            if args.plot_differences == 3:
                norm_residuals = (sim - meas) / res_err
                ax_rel.plot(y_bins, norm_residuals, color='gray', linestyle='dashdot', linewidth=1.5, label='Normalized residuals')
                ax_rel.set_ylabel("Normalized residuals")
                ax_rel.set_ylim(-20, 20)
        ax_rel.legend(loc=0)

        plot_2d_axes = axes_top[2]
        line_color = line_colors[2]
        hist_0, hist_error_0, _, _, _ = datasets[0]
        hist = datasets[1][0]
        hist_error = datasets[1][1]
        y_edges = datasets[1][2]
        z_edges = datasets[1][3]

        with np.errstate(divide='ignore', invalid='ignore'):
            if args.plot_differences == 1:
                diff2d = np.abs(hist - hist_0) / hist_0
                title_text = 'Relative absolute difference'
            elif args.plot_differences == 2:
                diff2d = (hist - hist_0) / hist_0
                title_text = 'Relative difference'
            elif args.plot_differences == 3:
                res_err_2d = np.sqrt(hist_error**2 + hist_error_0**2)
                diff2d = (hist - hist_0) / res_err_2d
                title_text = 'Normalized residuals'

        cmap = plt.get_cmap('jet')
        cmap.set_bad('k')
        ax = plot_2d_axes
        quadmesh = ax.pcolormesh(y_edges, z_edges, diff2d.T, cmap=cmap)
        fig = ax.figure
        fig.colorbar(quadmesh, ax=ax, orientation='vertical')

        ax.set_xlim(y_plot_range)
        ax.set_ylim(z_plot_range)
        ax.set_xlabel('Qy [1/nm]')
        ax.set_ylabel('Qz [1/nm]')
        ax.set_title(title_text)

def _setup_main_plot(args: Any, datasets: List[Any]) -> Tuple[Any, Any, Any, Any, str, Any, Any]:
    """Helper to setup the initial plot variables."""
    match_horizontal_axes = False
    ax1, ax2 = None, None
    axes_top, axes_bottom = None, None
    axes_multi2d = None

    if args.dual_plot:
        _, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))
        plot_output = 'none'
        match_horizontal_axes = True
    else:
        if args.overlay:
            top_row_plot_number = len(datasets) + (1 if args.plot_differences > 0 else 0)
            axes_top, axes_bottom = get_overlay_plot_axes(top_row_plot_number)
        elif args.multi2d:
            n_plots = len(datasets)
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
            axes_multi2d = axes.flatten()
            for i in range(n_plots, rows * cols):
                fig.delaxes(axes_multi2d[i])
            
        if args.pdf:
            plot_output = ".pdf"
        elif args.png:
            plot_output = ".png"
        else:
            plot_output = 'show'

    return match_horizontal_axes, ax1, ax2, axes_top, axes_bottom, plot_output, axes_multi2d

def main() -> None:
    """
    Main plotting routine.
    
    Parses arguments, loads datasets, and routes to appropriate plotting logic.
    """
    from .plot_cli import create_argparser, parse_args
    parser = create_argparser()
    args = parse_args(parser)

    # set global font size for axis labels, titles, and tick marks
    plt.rcParams.update({'font.size': args.font_size})

    datasets = get_datasets(args)
    match_horizontal_axes, ax1, ax2, axes_top, axes_bottom, plot_output, axes_multi2d = _setup_main_plot(args, datasets)

    if args.intensity_min is not None:
        intensity_min = float(args.intensity_min)
    else:
        is_upscaled = args.experiment_time
        intensity_min = 1e-9 if not is_upscaled else 1

    if args.overlay:
        y_plot_range, z_plot_range, _, max_value = get_plot_ranges(datasets, args.y_plot_range, args.z_plot_range)
        line_colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'brown']
        all_1d_values = []
        all_1d_errors = []
        for dataset_index, dataset in enumerate(datasets):
            plot_2d_axes = axes_top[dataset_index]
            line_color = line_colors[dataset_index]
            hist, hist_error, y_edges, z_edges, label = dataset

            common_maximum = max_value if args.individual_colorbars is False else None

            if plot_2d_axes is not None:
                log_plot_2d(hist, y_edges, z_edges, label, ax=plot_2d_axes, intensity_min=intensity_min, intensity_max=common_maximum, y_range=y_plot_range, z_range=z_plot_range, savename=args.savename, output='none')

            qz_min_index = np.digitize(args.q_min, z_edges) - 1
            qz_max_index = np.digitize(args.q_max, z_edges)
            qz_max_index_clamped = min(qz_max_index, len(z_edges) - 1)
            values, errors, y_bins, z_limits = extract_range_to_1d(hist, hist_error, y_edges, z_edges, [qz_min_index, qz_max_index])
            all_1d_values.append(values)
            all_1d_errors.append(errors)
            title_text = f" Qz=[{z_limits[0]:.4f} 1/nm, {z_limits[1]:.4f} 1/nm]"
            horizontal_axis_label = 'Qy [1/nm]'
            plot_q_1d(values, errors, y_bins, horizontal_axis_label, color=line_color, title_text=title_text, label=label, ax=axes_bottom, limits=y_plot_range, savename=args.savename, output='none')
            plot_2d_axes.axhline(z_edges[min(qz_min_index, len(z_edges)-1)], color='magenta', linestyle='--', label=f'q_z = {z_edges[min(qz_min_index, len(z_edges)-1)]:.3f}')
            plot_2d_axes.axhline(z_edges[qz_max_index_clamped], color='magenta', linestyle='--', label=f'q_z = {z_edges[qz_max_index_clamped]:.3f}')

            _plot_differences(args, dataset_index, datasets, all_1d_values, all_1d_errors, axes_top, axes_bottom, y_bins, y_plot_range, z_plot_range, line_colors)

        axes_bottom.set_ylim(bottom=intensity_min)
        axes_bottom.grid()
        axes_bottom.legend(loc='upper left')
        plt.tight_layout()

        show_or_save(plot_output, args.savename)

    if not args.overlay:
        if args.multi2d:
            y_plot_range, z_plot_range, _, max_value = get_plot_ranges(datasets, args.y_plot_range, args.z_plot_range)
            mappable = None
            for dataset_index, dataset in enumerate(datasets):
                plot_2d_axes = axes_multi2d[dataset_index]
                hist, hist_error, y_edges, z_edges, label = dataset
                common_maximum = max_value if args.individual_colorbars is False else None
                mappable = log_plot_2d(hist, y_edges, z_edges, label, ax=plot_2d_axes, intensity_min=intensity_min, intensity_max=common_maximum, y_range=y_plot_range, z_range=z_plot_range, savename=args.savename, output='none', add_colorbar=args.individual_colorbars)

            plt.tight_layout()
            if not args.individual_colorbars and mappable is not None:
                fig = axes_multi2d[0].figure
                fig.colorbar(mappable, ax=axes_multi2d[:len(datasets)], orientation='vertical', label='Intensity', fraction=0.075, pad=0.03)

            show_or_save(plot_output, args.savename)
        else:
            for hist, hist_error, y_edges, z_edges, label in datasets:
                y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges[0], y_edges[-1]]
                z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges[0], z_edges[-1]]
                log_plot_2d(hist, y_edges, z_edges, '', ax=ax1, intensity_min=intensity_min, y_range=y_plot_range, z_range=z_plot_range, savename=args.savename, match_horizontal_axes=match_horizontal_axes, output=plot_output)

                qz_min_index_exp = np.digitize(args.q_min, z_edges) - 1
                qz_max_index_exp = np.digitize(args.q_max, z_edges)
                values, errors, y_bins, z_limits = extract_range_to_1d(hist, hist_error, y_edges, z_edges, [qz_min_index_exp, qz_max_index_exp])
                title_text = f" Qz=[{z_limits[0]:.4f}1/nm, {z_limits[1]:.4f}1/nm]"
                horizontal_axis_label = 'Qy [1/nm]'
                plot_q_1d(values, errors, y_bins, horizontal_axis_label, color='blue', title_text=title_text, label=label, ax=ax2, limits=y_plot_range, savename=args.savename, output=plot_output)

                if ax1 is not None:
                    if 0 <= qz_min_index_exp < len(z_edges):
                        ax1.axhline(z_edges[qz_min_index_exp], color='magenta', linestyle='--')
                    if 0 <= qz_max_index_exp < len(z_edges):
                        ax1.axhline(z_edges[qz_max_index_exp], color='magenta', linestyle='--')

                if ax2 is not None:
                    ax2.grid(True)

    if args.dual_plot:
        if not args.pdf and not args.png:
            plt.show()
        else:
            if args.pdf:
                filename = f"{args.savename}.pdf"
            elif args.png:
                filename = f"{args.savename}.png"
            plt.savefig(filename, dpi=300)
            print(f"Created {filename}")

if __name__ == '__main__':
    main()
