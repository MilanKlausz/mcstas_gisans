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
        Whether to adjust colorbar to match horizontal axes, by default False.
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
    fig = ax.figure

    # plt.gca().invert_xaxis() #optionally invert x-axis?

    if add_colorbar:
        if not match_horizontal_axes:
            cbar = fig.colorbar(quadmesh, ax=ax, orientation='vertical')
        else:
            # Adjust the colorbar axis to exactly match the plot's vertical extent
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
            cbar = fig.colorbar(quadmesh, cax=cax)

    # cbar.set_label('Intensity') # Optionally set the colorbar label

    show_or_save(output, savename + '_2D')
    return quadmesh

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

def create_2d_histogram(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    y_bins: int = 256,
    z_bins: int = 128,
    y_range: Optional[List[float]] = None,
    z_range: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D histogram of weighted y-z values.

    Calculates the 2D histogram of the provided data and computes the corresponding
    statistical errors (uncertainties) based on the weights. The resulting
    histograms are transposed relative to the standard x-y plot axes if not 
    plotted accordingly.

    Parameters
    ----------
    x : numpy.ndarray
        Array of x-coordinates.
    y : numpy.ndarray
        Array of y-coordinates.
    weights : numpy.ndarray
        Array of weights (e.g., intensities).
    y_bins : int, optional
        Number of bins along the first dimension, by default 256.
    z_bins : int, optional
        Number of bins along the second dimension, by default 128.
    y_range : list of float, optional
        Limits for the first dimension, by default [-0.55, 0.55].
    z_range : list of float, optional
        Limits for the second dimension, by default [-0.5, 0.6].

    Returns
    -------
    hist : numpy.ndarray
        The 2D histogram values.
    hist_error : numpy.ndarray
        The calculated uncertainties for the 2D histogram.
    y_edges : numpy.ndarray
        The bin edges along the first dimension.
    z_edges : numpy.ndarray
        The bin edges along the second dimension.
    """
    if y_range is None: y_range = [-0.55, 0.55]
    if z_range is None: z_range = [-0.5, 0.6]
    
    # Generate the main 2D histogram
    hist, y_edges, z_edges = np.histogram2d(x, y, weights=weights, bins=[y_bins, z_bins], range=[y_range, z_range])
    
    # Calculate the histogram of squared weights to compute uncertainties
    hist_weight2, _, _ = np.histogram2d(x, y, weights=weights**2, bins=[y_bins, z_bins], range=[y_range, z_range])
    hist_error = np.sqrt(hist_weight2)

    return hist, hist_error, y_edges, z_edges

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