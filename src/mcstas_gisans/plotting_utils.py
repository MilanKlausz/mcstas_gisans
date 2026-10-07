"""
Collection of plotting functions
"""

import numpy as np
# from neutron_utilities import calculate_wavelength
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Optional, List, Tuple, Any

def show_or_save(output: str, filename_base: str) -> None:
    """
    Handle the display or saving of a matplotlib figure.

    Parameters
    ----------
    output : str
        Action to perform: 'show' to display, 'none' to do nothing, or a string
        to append to the base filename for saving (e.g., '.png').
    filename_base : str
        The base path and filename to which `output` is appended.

    Returns
    -------
    None
    """
    if output == 'show':
        plt.show()
    elif output != 'none':
        filename = filename_base + output
        plt.savefig(filename, dpi=300)
        print(f"Created {filename}")

def log_plot_2d(
    hist: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    title_text: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    intensity_min: float = 1e-9,
    intensity_max: Optional[float] = None,
    y_range: Optional[List[float]] = None,
    z_range: Optional[List[float]] = None,
    savename: str = 'plotQ',
    match_horizontal_axes: bool = False,
    output: str = 'show',
    add_colorbar: bool = True
) -> Any:
    """
    Plot a 2D histogram with a logarithmic color scale.

    Parameters
    ----------
    hist : numpy.ndarray
        The 2D histogram data to plot.
    y_edges : numpy.ndarray
        Bin edges along the y-axis.
    z_edges : numpy.ndarray
        Bin edges along the z-axis.
    title_text : str, optional
        Title of the plot, by default None.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on, by default None (creates a new one).
    intensity_min : float, optional
        Minimum intensity for the logarithmic color scale, by default 1e-9.
    intensity_max : float, optional
        Maximum intensity for the color scale. If None, uses the maximum value of `hist`.
    y_range : list of float, optional
        Limits for the y-axis, by default [-0.55, 0.55].
    z_range : list of float, optional
        Limits for the z-axis, by default [-0.5, 0.6].
    savename : str, optional
        Base filename if saving the plot, by default 'plotQ'.
    match_horizontal_axes : bool, optional
        Whether to attach the colorbar to the right of the axes instead of taking its width
        from them (so that the Qy axis lines up with a plot of the same width above or below),
        by default False.
    output : str, optional
        Action for the plot ('show', 'none', or extension like '.png'), by default 'show'.
    add_colorbar : bool, optional
        Whether to append a colorbar to the plot, by default True.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The QuadMesh object created by pcolormesh.
    """
    if y_range is None: y_range = [-0.55, 0.55]
    if z_range is None: z_range = [-0.5, 0.6]
    if ax is None:
        _, ax = plt.subplots()

    # Get colormap and set the color for invalid values (like empty bins with LogNorm)
    cmap = plt.get_cmap('jet')
    cmap.set_bad('k')
    
    # Determine the maximum intensity if not provided
    intensity_max = intensity_max if intensity_max is not None else np.max(hist)

    quadmesh = ax.pcolormesh(
        y_edges, z_edges, hist.T, 
        norm=colors.LogNorm(vmin=intensity_min, vmax=intensity_max), 
        cmap=cmap
    )

    ax.set_xlim(y_range)
    ax.set_ylim(z_range)
    ax.set_xlabel('Qy [1/nm]')
    ax.set_ylabel('Qz [1/nm]')
    if title_text is not None:
        ax.set_title(title_text)

    # plt.gca().invert_xaxis() #optionally invert x-axis?

    if add_colorbar:
        add_map_colorbar(quadmesh, ax, match_horizontal_axes)

    # cbar.set_label('Intensity') # Optionally set the colorbar label

    show_or_save(output, savename + '_2D')
    return quadmesh

def add_map_colorbar(mappable: Any, ax: plt.Axes, match_horizontal_axes: bool = False) -> Any:
    """
    Add a vertical colorbar to a 2D map.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The plotted map (e.g. the QuadMesh of pcolormesh).
    ax : matplotlib.axes.Axes
        The axes of the map.
    match_horizontal_axes : bool, optional
        If False (default), the colorbar takes its width from the map's axes. If True, it is
        attached to the right of the axes instead, so the map keeps the full width of its slot and
        its Qy axis lines up with a plot of the same width above or below it (an inset axes, which
        moves with the map, also through tight_layout).

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar.
    """
    fig = ax.figure
    if not match_horizontal_axes:
        return fig.colorbar(mappable, ax=ax, orientation='vertical')
    # gap and width of 0.01 and 0.02 figure widths, in units of the axes width
    width = ax.get_position().width
    cax = ax.inset_axes([1.0 + 0.01 / width, 0.0, 0.02 / width, 1.0])
    return fig.colorbar(mappable, cax=cax)

def link_axes(map_axes: List[plt.Axes], slice_ax: Optional[plt.Axes] = None) -> None:
    """
    Share the Qy and Qz axes of 2D Q maps, and the Qy axis of a 1D Qy-slice plot with them, so that
    zooming or panning one map in the interactive viewer applies to all of them (axis limits set
    later on any of them are shared, too). Tick labels are not hidden.

    Parameters
    ----------
    map_axes : list of matplotlib.axes.Axes
        Axes of the 2D maps (Qy horizontal, Qz vertical).
    slice_ax : matplotlib.axes.Axes, optional
        Axes of a 1D plot with Qy on its horizontal axis (only that axis is shared).
    """
    first = map_axes[0]
    for ax in map_axes[1:]:
        ax.sharex(first)
        ax.sharey(first)
    if slice_ax is not None:
        slice_ax.sharex(first)

def _finite_max_on_side(hist: np.ndarray, y_edges: np.ndarray, split_qy: float, left: bool) -> float:
    """Largest finite value of the Qy bins (first axis) whose centre is on the given side of split_qy."""
    centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    side = hist[centres < split_qy] if left else hist[centres > split_qy]
    side = side[np.isfinite(side)]
    return float(side.max()) if side.size else np.nan

def split_plot_2d(
    hist_left: np.ndarray,
    y_edges_left: np.ndarray,
    z_edges_left: np.ndarray,
    hist_right: np.ndarray,
    y_edges_right: np.ndarray,
    z_edges_right: np.ndarray,
    label_left: str = 'Measurement',
    label_right: str = 'Simulation',
    title_text: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    intensity_min: float = 1e-9,
    intensity_max: Optional[float] = None,
    y_range: Optional[List[float]] = None,
    z_range: Optional[List[float]] = None,
    split_qy: float = 0.0,
    add_colorbar: bool = True,
    match_horizontal_axes: bool = False
) -> Any:
    """
    Plot two 2D histograms in one map with a common logarithmic color scale: the first one
    (e.g. the measurement) for Qy < split_qy, the second one (e.g. the simulation) for Qy > split_qy.

    Each histogram is clipped exactly at split_qy (a bin across it is cut, not dropped), a vertical
    line marks the split and the two halves are labelled.

    Parameters
    ----------
    hist_left, hist_right : numpy.ndarray
        The 2D histograms [Qy, Qz] shown left and right of split_qy.
    y_edges_left, z_edges_left, y_edges_right, z_edges_right : numpy.ndarray
        Their Qy and Qz bin edges.
    label_left, label_right : str, optional
        Labels written into the two halves.
    title_text : str, optional
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (default: a new figure).
    intensity_min : float, optional
        Minimum of the logarithmic color scale.
    intensity_max : float, optional
        Maximum of the color scale (default: the largest value shown in either half).
    y_range, z_range : list of float, optional
        Axis limits (default: the extent of both histograms).
    split_qy : float, optional
        Qy of the split, by default 0.
    add_colorbar : bool, optional
        Whether to add a colorbar, by default True.
    match_horizontal_axes : bool, optional
        Whether to attach the colorbar to the right of the axes instead of taking its width
        from them (see add_map_colorbar), by default False.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The QuadMesh of the left half (for a colorbar).
    """
    from matplotlib.path import Path
    if ax is None:
        _, ax = plt.subplots()
    if y_range is None:
        y_range = [min(y_edges_left[0], y_edges_right[0]), max(y_edges_left[-1], y_edges_right[-1])]
    if z_range is None:
        z_range = [min(z_edges_left[0], z_edges_right[0]), max(z_edges_left[-1], z_edges_right[-1])]
    if intensity_max is None:
        intensity_max = np.nanmax([_finite_max_on_side(hist_left, y_edges_left, split_qy, left=True),
                                   _finite_max_on_side(hist_right, y_edges_right, split_qy, left=False)])

    cmap = plt.get_cmap('jet').with_extremes(bad='k')
    norm = colors.LogNorm(vmin=intensity_min, vmax=intensity_max)
    quadmesh_left = ax.pcolormesh(y_edges_left, z_edges_left, hist_left.T, norm=norm, cmap=cmap)
    quadmesh_right = ax.pcolormesh(y_edges_right, z_edges_right, hist_right.T, norm=norm, cmap=cmap)
    # clip each half to its side of split_qy (within the extent of its histogram). A Path, unlike a
    # Rectangle patch, is applied in addition to the clipping to the axes, so the maps stay inside
    # the axes also when zoomed or panned in the interactive viewer.
    def side_clip_path(y_from, y_to, z_edges):
        z_lo, z_hi = np.min(z_edges), np.max(z_edges)
        return Path([(y_from, z_lo), (y_to, z_lo), (y_to, z_hi), (y_from, z_hi), (y_from, z_lo)], closed=True)
    quadmesh_left.set_clip_path(side_clip_path(min(np.min(y_edges_left), split_qy), split_qy, z_edges_left),
                                ax.transData)
    quadmesh_right.set_clip_path(side_clip_path(split_qy, max(np.max(y_edges_right), split_qy), z_edges_right),
                                 ax.transData)
    ax.axvline(split_qy, color='white', linewidth=1)

    text_style = dict(transform=ax.transAxes, va='top', color='white',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='none'))
    ax.text(0.02, 0.98, label_left, ha='left', **text_style)
    ax.text(0.98, 0.98, label_right, ha='right', **text_style)

    ax.set_xlim(y_range)
    ax.set_ylim(z_range)
    ax.set_xlabel('Qy [1/nm]')
    ax.set_ylabel('Qz [1/nm]')
    if title_text is not None:
        ax.set_title(title_text)
    if add_colorbar:
        add_map_colorbar(quadmesh_left, ax, match_horizontal_axes)
    return quadmesh_left

def plot_q_1d(
    values: np.ndarray,
    errors: np.ndarray,
    bin_edges: np.ndarray,
    horizontal_axis_label: str,
    color: str = 'blue',
    title_text: Optional[str] = None,
    label: str = '',
    ax: Optional[plt.Axes] = None,
    limits: Optional[List[float]] = None,
    savename: str = 'plotQ',
    output: str = 'show'
) -> None:
    """
    Plot a 1D histogram or slice with error bars on a logarithmic scale.

    Parameters
    ----------
    values : numpy.ndarray
        The intensity values to plot.
    errors : numpy.ndarray
        The uncertainties (errors) associated with `values`.
    bin_edges : numpy.ndarray
        The positions along the horizontal axis.
    horizontal_axis_label : str
        Label for the horizontal axis.
    color : str, optional
        Color of the data points and line, by default 'blue'.
    title_text : str, optional
        Title of the plot, by default None.
    label : str, optional
        Label for the legend, by default ''.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on, by default None (creates a new one).
    limits : list of float, optional
        Limits for the horizontal axis, by default [-0.55, 0.55].
    savename : str, optional
        Base filename if saving the plot, by default 'plotQ'.
    output : str, optional
        Action for the plot ('show', 'none', or extension like '.png'), by default 'show'.

    Returns
    -------
    None
    """
    if limits is None: limits = [-0.55, 0.55]
    if ax is None:
        _, ax = plt.subplots()

    ax.errorbar(bin_edges, values, yerr=errors, fmt='o-', capsize=5, ecolor='red', color=color, label=label)
    ax.set_xlabel(horizontal_axis_label)
    ax.set_ylabel('Intensity')
    if title_text is not None:
        ax.set_title(title_text)
    ax.set_yscale("log")
    ax.set_xlim(limits)

    show_or_save(output, savename + '_qSlice')


def extract_range_to_1d(
    hist: np.ndarray,
    hist_error: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    z_index_range: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Extract a slice from a 2D histogram into a 1D histogram by summing over a range.

    Handles the correct propagation of uncertainty from the 2D error histogram.

    Parameters
    ----------
    hist : numpy.ndarray
        The 2D histogram data.
    hist_error : numpy.ndarray
        The 2D array of statistical uncertainties corresponding to `hist`.
    y_edges : numpy.ndarray
        Bin edges along the y-axis.
    z_edges : numpy.ndarray
        Bin edges along the z-axis.
    z_index_range : list of int
        A two-element list specifying the start and end indices along the z-axis to extract.

    Returns
    -------
    values : numpy.ndarray
        The extracted 1D histogram values (summed along the z-axis slice).
    errors : numpy.ndarray
        The propagated uncertainties for the 1D histogram.
    y_bins : numpy.ndarray
        The bin centers along the y-axis.
    z_limits : list of float
        The physical limits [z_min, z_max] of the extracted range.
    """
    # Ensure indices are within valid bounds
    z_idx_0 = min(max(0, z_index_range[0]), len(z_edges) - 2)
    z_idx_1 = min(max(0, z_index_range[1]), len(z_edges) - 2)
    
    # Ensure correct ordering
    if z_idx_0 > z_idx_1:
        z_idx_0, z_idx_1 = z_idx_1, z_idx_0
        
    z_limits = [float(z_edges[z_idx_0]), float(z_edges[z_idx_1 + 1])]
    
    # Extract the requested region and sum over the secondary axis
    values_extracted = hist[:, z_idx_0:z_idx_1 + 1]
    values = np.sum(values_extracted, axis=1)
    
    # Propagate errors in quadrature
    errors_extracted = hist_error[:, z_idx_0:z_idx_1 + 1]
    errors = np.sqrt(np.sum(errors_extracted**2, axis=1))
    
    # Calculate bin centers from bin edges
    y_bins = (y_edges[:-1] + y_edges[1:]) / 2 
    
    return values, errors, y_bins, z_limits

### TODO in dev ###
def extract_range_to_1d_vertical(
    hist: np.ndarray,
    hist_error: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    y_index_range: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Extract a slice from a 2D histogram into a 1D histogram (vertical summing).

    Handles the propagation of error from the corresponding 2D error histogram.

    Parameters
    ----------
    hist : numpy.ndarray
        The 2D histogram data.
    hist_error : numpy.ndarray
        The 2D array of statistical uncertainties corresponding to `hist`.
    y_edges : numpy.ndarray
        Bin edges along the y-axis.
    z_edges : numpy.ndarray
        Bin edges along the z-axis.
    y_index_range : list of int
        A two-element list specifying the start and end indices along the y-axis to extract.

    Returns
    -------
    values : numpy.ndarray
        The extracted 1D histogram values (summed along the y-axis slice).
    errors : numpy.ndarray
        The propagated uncertainties for the 1D histogram.
    z_bins : numpy.ndarray
        The bin centers along the z-axis.
    y_limits : list of float
        The physical limits [y_min, y_max] of the extracted range.
    """
    # Ensure indices are within valid bounds
    y_idx_0 = min(max(0, y_index_range[0]), len(y_edges) - 2)
    y_idx_1 = min(max(0, y_index_range[1]), len(y_edges) - 2)
    
    # Ensure correct ordering
    if y_idx_0 > y_idx_1:
        y_idx_0, y_idx_1 = y_idx_1, y_idx_0
        
    y_limits = [float(y_edges[y_idx_0]), float(y_edges[y_idx_1 + 1])]
    
    # Extract the requested region and sum over the primary axis
    values_extracted = hist[y_idx_0:y_idx_1 + 1, :]
    values = np.sum(values_extracted, axis=0)
    
    # Propagate errors in quadrature
    errors_extracted = hist_error[y_idx_0:y_idx_1 + 1, :]
    errors = np.sqrt(np.sum(errors_extracted**2, axis=0))
    
    # Calculate bin centers from bin edges
    z_bins = (z_edges[:-1] + z_edges[1:]) / 2 
    
    return values, errors, z_bins, y_limits