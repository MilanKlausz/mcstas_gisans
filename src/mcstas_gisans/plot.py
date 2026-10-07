#!/usr/bin/env python3

"""
Main plotting script to create 2D/1D Q plots from simulation results
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any, Dict

from .plotting_utils import plot_q_1d, log_plot_2d, extract_range_to_1d, show_or_save, link_axes
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
    The Qy and Qz axes of the top row (2D maps) are shared, and the Qy axis of the bottom
    row (1D slice) with them, so that zooming one of them zooms all.

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
    fig, axes = plt.subplots(2, column, figsize=(16, 12), squeeze=False)  # 2D array also for one column

    #Replace the bottom row of the grid with a single new subplot
    gs = axes[1, 0].get_gridspec() #Get GridSpec from the bottom left subplot
    for i in range(column): #Remove all bottom subplots
        axes[1, i].remove()
    axes_bottom = fig.add_subplot(gs[1:, :]) #cover the row with a new subplot

    axes = axes.flatten()
    axes_top = [axes[i] for i in range(column)]
    link_axes(axes_top, axes_bottom)
    return axes_top, axes_bottom

PLOT_DEFAULTS = {'instrument': 'd22', 'alpha': 0.0, 'sample_orientation': 1, 'wavelength': 6.0}


def _extract_metadata_from_scipp(scipp_dg: Any) -> Dict[str, Any]:
    """
    Extract the instrument configuration stored by mg_run in a Scipp DataGroup.
    Missing entries (older files) are simply absent from the returned dict.
    """
    import json
    import scipp as sc
    metadata: Dict[str, Any] = {}
    if not (isinstance(scipp_dg, sc.DataGroup) and 'instrument' in scipp_dg):
        return metadata
    inst = scipp_dg['instrument']

    def value(key):
        v = inst[key].value
        return v.decode('utf-8') if isinstance(v, bytes) else v

    for key, meta_key in (('name', 'instrument'), ('alpha_inc_deg', 'alpha'), ('sample_orientation', 'sample_orientation'),
                          ('beam_angle', 'beam_angle'), ('no_gravity', 'no_gravity'), ('wfm', 'wfm')):
        if key in inst:
            metadata[meta_key] = value(key)
    if 'wavelength_selected' in inst and value('wavelength_selected'):
        metadata['wavelength'] = float(value('wavelength_selected'))
    if 'detector_centre_offset_x' in inst and 'detector_centre_offset_y' in inst:
        metadata['detector_centre_offset'] = [float(value('detector_centre_offset_x')), float(value('detector_centre_offset_y'))]
    if 'parameters_json' in inst:
        metadata['parameters'] = json.loads(value('parameters_json'))
    if metadata.get('instrument') == 'unknown':
        del metadata['instrument']
    return metadata


def _cli_or_metadata(args: Any, name: str, metadata: Dict[str, Any], label: str) -> Any:
    """Explicit command line value > value stored in the simulation file > plotting default."""
    cli_value = getattr(args, name, None)
    stored = metadata.get(name)
    if cli_value is not None:
        if stored is not None and not _same(cli_value, stored):
            print(f"WARNING: --{name} {cli_value} overrides the value stored in {label} ({stored}).")
        return cli_value
    return stored if stored is not None else PLOT_DEFAULTS[name]


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return a == b


def build_plot_instrument(args: Any, metadata: Dict[str, Any], label: str = 'the simulation file') -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Build the Instrument for one simulation file from its stored configuration, with
    explicitly given command line values taking precedence.

    Returns (instrument, instrument_parameters, settings) where settings holds alpha,
    sample_orientation, wavelength, no_gravity and wfm.
    """
    from .instrument import Instrument
    from .instrument_defaults import resolve_instrument_parameters

    name = _cli_or_metadata(args, 'instrument', metadata, label)
    stored_params = metadata.get('parameters') if metadata.get('instrument') == name else None
    params = resolve_instrument_parameters(name, args, prefix='instrument_', base=stored_params)
    if stored_params is None and metadata.get('instrument') == name:
        # older files: only the offset and beam angle were stored
        if metadata.get('detector_centre_offset') is not None and getattr(args, 'instrument_detector_centre_offset', None) is None:
            params['detector']['direct_beam_centre_offset'] = list(metadata['detector_centre_offset'])
        if metadata.get('beam_angle') is not None and getattr(args, 'instrument_beam_angle', None) is None:
            params['beam_angle'] = metadata['beam_angle']

    settings = {
        'alpha': float(_cli_or_metadata(args, 'alpha', metadata, label)),
        'sample_orientation': int(_cli_or_metadata(args, 'sample_orientation', metadata, label)),
        'wavelength': float(_cli_or_metadata(args, 'wavelength', metadata, label)),
        'no_gravity': bool(metadata.get('no_gravity', False)),
        'wfm': bool(metadata.get('wfm', False)),
    }
    instrument = Instrument(params, settings['alpha'], settings['wavelength'], settings['sample_orientation'],
                            wfm=settings['wfm'], no_gravity=settings['no_gravity'])
    return instrument, params, settings


def _load_nexus_datasets(args: Any, reference: Optional[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]], float]:
    """
    Load the measured NeXus datasets.

    The measurement is interpreted with the instrument configuration of the first
    simulation file (`reference` = (instrument_parameters, settings)) if one is given,
    otherwise with the command line / default configuration; --nxs_instrument_* and
    --nxs_sample_orientation override it. Measured data always include gravity.
    """
    from .instrument import Instrument
    from .instrument_defaults import get_nxs_instrument_parameters
    from .nexus_reader import read_nexus_data, warn_if_duration_mismatch

    datasets = []
    nxs_sum = 0.0
    if not args.nxs:
        return datasets, nxs_sum

    if reference is not None:
        base_params, settings = reference
    else:
        _, base_params, settings = build_plot_instrument(args, {})
    nxs_params = get_nxs_instrument_parameters(args, default_instr_name=base_params['name'], base=base_params)
    orientation = int(args.nxs_sample_orientation) if getattr(args, 'nxs_sample_orientation', None) is not None else settings['sample_orientation']
    nxs_instrument = Instrument(nxs_params, settings['alpha'], settings['wavelength'], orientation)

    nxs_labels = args.nxs_label if args.nxs_label else [os.path.basename(f) for f in args.nxs]  # default to the file name
    for nxs_filename, nxs_label in zip(args.nxs, nxs_labels):
        hist, hist_error, y_edges, z_edges = read_nexus_data(nxs_filename, nxs_instrument, data_path=getattr(args, 'nxs_data_path', None))
        warn_if_duration_mismatch([nxs_filename], getattr(args, 'experiment_time', None), label=nxs_filename)
        if not datasets:
            nxs_sum = np.sum(hist)  # --normalise_to_nxs uses the first NeXus file
        if args.verbose:
            print(f"{nxs_filename} sum: {np.sum(hist)}")
        datasets.append((hist, hist_error, y_edges, z_edges, nxs_label))
    return datasets, nxs_sum


def _load_sim_datasets(args: Any) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]], Optional[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """
    Load simulation datasets from .h5 (and legacy .npz) files, each with the instrument
    configuration stored in the file itself. Returns (datasets, reference) where reference
    is the (instrument_parameters, settings) of the first .h5 file (or None).
    """
    import scipp as sc
    datasets = []
    reference = None
    if not args.filename:
        return datasets, reference
    labels = args.label if args.label else args.filename  # default to filename if no label is provided
    for filename, label in zip(args.filename, labels):
        if filename.endswith('.h5'):
            scipp_dg = load_scipp_file(filename)
            if not (isinstance(scipp_dg, sc.DataGroup) and 'instrument' in scipp_dg):
                raise ValueError(f"{filename} does not appear to be a valid mcstas_gisans sc.DataGroup output.")
            instrument, params, settings = build_plot_instrument(args, _extract_metadata_from_scipp(scipp_dg), label=filename)
            if reference is None:
                reference = (params, settings)
            scipp_da = scipp_dg['data']
            wavelength = settings['wavelength']

            y_edges, z_edges = instrument.get_q_pixel_limits(wavelength)
            if scipp_da.bins is not None:
                # TOF / event data: Q per event
                scipp_da = instrument.compute_q_scipp(scipp_da)
                if getattr(args, 'wavelength_slice', None) is not None:
                    min_w, max_w = args.wavelength_slice
                    scipp_da = scipp_da.bin(wavelength=sc.array(dims=['wavelength'], values=[min_w, max_w], unit='angstrom'))
                scipp_da = scipp_da.bins.concat()
                # default Q range: the extent of the events (all wavelengths), in 1/nm
                events = scipp_da.bins.constituents['data']
                event_qy, event_qz = events.coords['Qy'].values * 10.0, events.coords['Qz'].values * 10.0
                pad = lambda lo, hi: (lo - 1e-9 * max(1.0, abs(hi - lo)), hi + 1e-9 * max(1.0, abs(hi - lo)))
                y_min, y_max = tuple(args.y_plot_range) if getattr(args, 'y_plot_range', None) else pad(np.nanmin(event_qy), np.nanmax(event_qy))
                z_min, z_max = tuple(args.z_plot_range) if getattr(args, 'z_plot_range', None) else pad(np.nanmin(event_qz), np.nanmax(event_qz))
                bins_y, bins_z = len(y_edges) - 1, len(z_edges) - 1
                scipp_binned = scipp_da.bin(
                    Qy=sc.linspace(dim='Qy', start=y_min/10, stop=y_max/10, num=bins_y + 1, unit='1/angstrom'),
                    Qz=sc.linspace(dim='Qz', start=z_min/10, stop=z_max/10, num=bins_z + 1, unit='1/angstrom')
                )
                scipp_hist = scipp_binned.bins.sum().transpose(['Qy', 'Qz'])
                hist = scipp_hist.values
                hist_error = np.sqrt(scipp_hist.variances) if scipp_hist.variances is not None else np.sqrt(hist)
                y_edges = scipp_hist.coords['Qy'].values * 10.0
                z_edges = scipp_hist.coords['Qz'].values * 10.0
            else:
                # non-TOF pixel data: keep the exact pixel grid, identical to the NeXus treatment
                shape = (instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus)
                raw_hist = scipp_da.values.reshape(shape)
                raw_err = np.sqrt(scipp_da.variances.reshape(shape)) if scipp_da.variances is not None else np.sqrt(raw_hist)
                hist = instrument.detector.coords.rotate_detector_image(raw_hist)
                hist_error = instrument.detector.coords.rotate_detector_image(raw_err)

        elif filename.endswith('.npz'):
            # Legacy Q-histogram files of the dev branch: hist shape (Qy, Qz[, Qx])
            data = np.load(filename)
            hist = data['hist']
            hist_error = data['error'] if 'error' in data else np.sqrt(hist)
            if hist.ndim == 3:
                hist = np.sum(hist, axis=2)
                hist_error = np.sqrt(np.sum(hist_error**2, axis=2))
            y_edges = data['yEdges']
            z_edges = data['zEdges']
        else:
            sys.exit("Unsupported file extension. Only .h5 and .npz files are supported.")

        if args.experiment_time:
            hist, hist_error = upscale_simple(hist, hist_error, args.experiment_time, args.background)

        hist_sum = np.sum(hist)
        if args.verbose:
            print(f"{filename} sum: {hist_sum}")
        datasets.append((hist, hist_error, y_edges, z_edges, label))

        if args.csv:
            csv_filename = f"{filename.rsplit('.', 1)[0]}.csv"
            np.savetxt(csv_filename, hist, delimiter=',')
            print(f"Created {csv_filename}")
    return datasets, reference


def get_datasets(args: Any) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """
    Prepare the datasets to be plotted.

    Simulation files (.h5) are interpreted with the instrument configuration stored in
    each file (explicit command line values take precedence, with a warning). Measured
    NeXus data are interpreted with the configuration of the first simulation file
    (unless overridden with the command line or --nxs_instrument_* options), or with the
    command line configuration if no simulation file is given.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    list
        List of dataset tuples (hist, hist_error, y_edges, z_edges, label), NeXus datasets first.
    """
    sim_datasets, reference = _load_sim_datasets(args)  # first .h5 file = reference configuration
    nxs_datasets, nxs_sum = _load_nexus_datasets(args, reference)
    if args.normalise_to_nxs and nxs_sum > 0:
        # normalise the total intensity of each simulation to the NeXus data (float arithmetic:
        # Poisson-sampled histograms are integer arrays)
        sim_datasets = [(hist * (nxs_sum / np.sum(hist)), err * (nxs_sum / np.sum(hist)), y, z, label)
                        for hist, err, y, z, label in sim_datasets]
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
    Plot differences between the second and the first dataset (e.g. simulation vs measurement)
    in the extra panel after the dataset panels, and on a secondary axis of the 1D plot.

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

        plot_2d_axes = axes_top[len(datasets)]  # the extra panel after the dataset panels
        hist_0, hist_error_0, y_edges_0, z_edges_0, _ = datasets[0]
        hist = datasets[1][0]
        hist_error = datasets[1][1]
        y_edges = datasets[1][2]
        z_edges = datasets[1][3]
        if hist.shape != hist_0.shape or not (np.allclose(y_edges, y_edges_0) and np.allclose(z_edges, z_edges_0)):
            sys.exit("--plot_differences: the first two datasets have different Q binning and cannot be compared pixel by pixel.")

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

def _plot_split_view(args: Any, datasets: List[Any], intensity_min: float, plot_output: str) -> None:
    """
    One 2D map of the first two datasets: the first for Qy < 0, the second for Qy > 0
    (with --nxs: the measurement left, the simulation right), on a common colour scale.
    With --overlay, the 1D Qy slices of both datasets (--q_min..--q_max) below, as wide as the map.
    """
    if len(datasets) != 2:
        sys.exit(f"--split_view needs exactly two datasets, got {len(datasets)}.")
    from .plotting_utils import split_plot_2d
    (hist_left, _, y_edges_left, z_edges_left, label_left), (hist_right, _, y_edges_right, z_edges_right, label_right) = datasets
    y_plot_range, z_plot_range, _, _ = get_plot_ranges(datasets, args.y_plot_range, args.z_plot_range)
    if args.overlay:
        _, (ax, ax_slice) = plt.subplots(2, 1, figsize=(10, 12))
        link_axes([ax], ax_slice)
    else:
        _, ax = plt.subplots(figsize=(10, 8))
    split_plot_2d(hist_left, y_edges_left, z_edges_left, hist_right, y_edges_right, z_edges_right,
                  label_left=label_left, label_right=label_right, ax=ax, intensity_min=intensity_min,
                  y_range=y_plot_range, z_range=z_plot_range, match_horizontal_axes=args.overlay)
    if args.overlay:
        for line_color, (hist, hist_error, y_edges, z_edges, label) in zip(['blue', 'green'], datasets):
            qz_min_index = np.digitize(args.q_min, z_edges) - 1
            qz_max_index = np.digitize(args.q_max, z_edges) - 1
            values, errors, y_bins, z_limits = extract_range_to_1d(hist, hist_error, y_edges, z_edges, [qz_min_index, qz_max_index])
            title_text = f" Qz=[{z_limits[0]:.4f} 1/nm, {z_limits[1]:.4f} 1/nm]"
            plot_q_1d(values, errors, y_bins, 'Qy [1/nm]', color=line_color, title_text=title_text, label=label, ax=ax_slice, limits=y_plot_range, savename=args.savename, output='none')
        for z_limit in z_limits:  # the summed Qz range
            ax.axhline(z_limit, color='magenta', linestyle='--')
        ax_slice.set_ylim(bottom=intensity_min)
        ax_slice.grid()
        ax_slice.legend(loc='upper left')
    plt.tight_layout()
    show_or_save(plot_output, args.savename)

def _setup_main_plot(args: Any, datasets: List[Any]) -> Tuple[Any, Any, Any, Any, str, Any, Any]:
    """Helper to setup the initial plot variables."""
    match_horizontal_axes = False
    ax1, ax2 = None, None
    axes_top, axes_bottom = None, None
    axes_multi2d = None

    if args.dual_plot:
        _, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))
        link_axes([ax1], ax2)  # the Qy axis of the 1D slice follows the map
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
            link_axes(axes_multi2d[:n_plots])  # zooming one map zooms all
            
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
    if args.plot_differences > 0 and (not args.overlay or len(datasets) < 2):
        sys.exit("--plot_differences needs --overlay and at least two datasets (the second is compared with the first).")
    match_horizontal_axes, ax1, ax2, axes_top, axes_bottom, plot_output, axes_multi2d = _setup_main_plot(args, datasets)

    if args.intensity_min is not None:
        intensity_min = float(args.intensity_min)
    else:
        is_upscaled = args.experiment_time
        intensity_min = 1e-9 if not is_upscaled else 1

    if args.split_view:
        _plot_split_view(args, datasets, intensity_min, plot_output)
        return

    if args.overlay:
        y_plot_range, z_plot_range, _, max_value = get_plot_ranges(datasets, args.y_plot_range, args.z_plot_range)
        line_colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'brown']
        all_1d_values = []
        all_1d_errors = []
        for dataset_index, dataset in enumerate(datasets):
            plot_2d_axes = axes_top[dataset_index]
            line_color = line_colors[dataset_index % len(line_colors)]
            hist, hist_error, y_edges, z_edges, label = dataset

            common_maximum = max_value if args.individual_colorbars is False else None

            if plot_2d_axes is not None:
                # a single map spans the row: its colorbar is attached to its right instead of taking
                # its width, so that its Qy axis lines up with the 1D slice below
                log_plot_2d(hist, y_edges, z_edges, label, ax=plot_2d_axes, intensity_min=intensity_min, intensity_max=common_maximum, y_range=y_plot_range, z_range=z_plot_range, savename=args.savename, match_horizontal_axes=len(axes_top) == 1, output='none')

            qz_min_index = np.digitize(args.q_min, z_edges) - 1
            qz_max_index = np.digitize(args.q_max, z_edges) - 1
            values, errors, y_bins, z_limits = extract_range_to_1d(hist, hist_error, y_edges, z_edges, [qz_min_index, qz_max_index])
            all_1d_values.append(values)
            all_1d_errors.append(errors)
            title_text = f" Qz=[{z_limits[0]:.4f} 1/nm, {z_limits[1]:.4f} 1/nm]"
            horizontal_axis_label = 'Qy [1/nm]'
            plot_q_1d(values, errors, y_bins, horizontal_axis_label, color=line_color, title_text=title_text, label=label, ax=axes_bottom, limits=y_plot_range, savename=args.savename, output='none')
            for z_limit in z_limits:  # the summed Qz range
                plot_2d_axes.axhline(z_limit, color='magenta', linestyle='--', label=f'q_z = {z_limit:.3f}')

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
            for dataset_index, (hist, hist_error, y_edges, z_edges, label) in enumerate(datasets):
                # separate figures: one file per dataset, instead of overwriting the same file
                savename = args.savename if len(datasets) == 1 or args.dual_plot else f"{args.savename}_{dataset_index}"
                y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges[0], y_edges[-1]]
                z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges[0], z_edges[-1]]
                log_plot_2d(hist, y_edges, z_edges, '', ax=ax1, intensity_min=intensity_min, y_range=y_plot_range, z_range=z_plot_range, savename=savename, match_horizontal_axes=match_horizontal_axes, output=plot_output)

                qz_min_index_exp = np.digitize(args.q_min, z_edges) - 1
                qz_max_index_exp = np.digitize(args.q_max, z_edges) - 1
                values, errors, y_bins, z_limits = extract_range_to_1d(hist, hist_error, y_edges, z_edges, [qz_min_index_exp, qz_max_index_exp])
                title_text = f" Qz=[{z_limits[0]:.4f}1/nm, {z_limits[1]:.4f}1/nm]"
                horizontal_axis_label = 'Qy [1/nm]'
                plot_q_1d(values, errors, y_bins, horizontal_axis_label, color='blue', title_text=title_text, label=label, ax=ax2, limits=y_plot_range, savename=savename, output=plot_output)

                if ax1 is not None:
                    for z_limit in z_limits:  # the summed Qz range
                        ax1.axhline(z_limit, color='magenta', linestyle='--')

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
