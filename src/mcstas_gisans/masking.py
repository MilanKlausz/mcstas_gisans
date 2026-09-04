"""
Detector masking utilities and visualization
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Any

def get_mask(y_edges: np.ndarray, 
             z_edges: np.ndarray, 
             mask_qy_range: Optional[Tuple[float, float]] = None,
             qy_min_cut: Optional[float] = None, 
             qy_max_cut: Optional[float] = None, 
             qz_min_cut: Optional[float] = None, 
             qz_max_cut: Optional[float] = None,
             exclude_q_box: Optional[List[Tuple[float, float, float, float]]] = None, 
             include_q_box: Optional[List[Tuple[float, float, float, float]]] = None,
             shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Calculates a boolean mask of shape matching the detector data.

    True = keep, False = mask out.

    Order of evaluation:
      1. Start with keep_mask = all True.
      2. Apply ALL exclusion masks (cuts, mask_qy_range, exclude_q_box).
      3. Apply ALL inclusion masks (include_q_box) after all exclusions.

    Parameters
    ----------
    y_edges : np.ndarray
        The bin edges in the y-direction (typically Qy).
    z_edges : np.ndarray
        The bin edges in the z-direction (typically Qz).
    mask_qy_range : tuple of float, optional
        A range (min, max) in Qy to be excluded.
    qy_min_cut : float, optional
        Minimum Qy value to keep; anything below is excluded.
    qy_max_cut : float, optional
        Maximum Qy value to keep; anything above is excluded.
    qz_min_cut : float, optional
        Minimum Qz value to keep; anything below is excluded.
    qz_max_cut : float, optional
        Maximum Qz value to keep; anything above is excluded.
    exclude_q_box : list of tuple of float, optional
        A list of bounding boxes (qy_min, qy_max, qz_min, qz_max) to exclude.
    include_q_box : list of tuple of float, optional
        A list of bounding boxes (qy_min, qy_max, qz_min, qz_max) to forcefully include.
    shape : tuple of int, optional
        The shape of the expected mask. Defaults to (len(y_edges) - 1, len(z_edges) - 1).

    Returns
    -------
    np.ndarray
        A boolean mask array of the specified shape.
    """
    if shape is None:
        shape = (len(y_edges) - 1, len(z_edges) - 1)

    # Initialize mask keeping all pixels by default
    keep_mask = np.ones(shape, dtype=bool)

    # Calculate bin centers
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    z_centres = (z_edges[:-1] + z_edges[1:]) / 2.0

    # Determine proper meshgrid indexing based on shape layout
    if shape[0] == len(y_centres):
        YY, ZZ = np.meshgrid(y_centres, z_centres, indexing='ij')
    else:
        YY, ZZ = np.meshgrid(y_centres, z_centres, indexing='xy')

    # PHASE 1: Apply ALL Exclusion Masks
    if mask_qy_range is not None:
        keep_mask[(YY >= mask_qy_range[0]) & (YY <= mask_qy_range[1])] = False

    if qy_min_cut is not None:
        keep_mask[YY < qy_min_cut] = False

    if qy_max_cut is not None:
        keep_mask[YY > qy_max_cut] = False

    if qz_min_cut is not None:
        keep_mask[ZZ < qz_min_cut] = False

    if qz_max_cut is not None:
        keep_mask[ZZ > qz_max_cut] = False

    # Exclude custom box regions
    if exclude_q_box:
        for box in exclude_q_box:
            if len(box) == 4:
                qy_min, qy_max, qz_min, qz_max = box
                box_mask = (YY >= qy_min) & (YY <= qy_max) & (ZZ >= qz_min) & (ZZ <= qz_max)
                keep_mask[box_mask] = False

    # PHASE 2: Apply ALL Inclusion Masks (AFTER all exclusions)
    if include_q_box:
        for box in include_q_box:
            if len(box) == 4:
                qy_min, qy_max, qz_min, qz_max = box
                box_mask = (YY >= qy_min) & (YY <= qy_max) & (ZZ >= qz_min) & (ZZ <= qz_max)
                keep_mask[box_mask] = True

    return keep_mask

def apply_mask(data: np.ndarray, mask: np.ndarray, fill_value: Union[float, int]) -> np.ndarray:
    """
    Applies a precalculated boolean mask to the data, replacing False values with fill_value.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    mask : np.ndarray
        The boolean mask to apply (True means keep, False means mask out).
    fill_value : float or int
        The value to substitute where the mask is False.

    Returns
    -------
    np.ndarray
        The masked data array.
    """
    # Ensure array is float64 if fill_value is NaN to prevent dtype issues
    res = data.astype(np.float64) if isinstance(fill_value, float) and np.isnan(fill_value) else data.copy()
    res[~mask] = fill_value
    return res

def save_view_masks_plot(hist_raw: np.ndarray, 
                         hist_raw_error: np.ndarray, 
                         hist_masked: np.ndarray, 
                         hist_masked_error: np.ndarray,
                         y_edges_nxs: np.ndarray, 
                         z_edges_nxs: np.ndarray, 
                         q_min: float, 
                         q_max: float, 
                         y_plot_range: Tuple[float, float], 
                         z_plot_range: Tuple[float, float],
                         savename: str, 
                         intensity_min: Optional[Union[float, int]] = None) -> None:
    """
    Generates a comparison plot showing raw data vs applied detector masks.

    Parameters
    ----------
    hist_raw : np.ndarray
        The raw 2D histogram data.
    hist_raw_error : np.ndarray
        The raw 2D histogram error data.
    hist_masked : np.ndarray
        The masked 2D histogram data.
    hist_masked_error : np.ndarray
        The masked 2D histogram error data.
    y_edges_nxs : np.ndarray
        The bin edges for the y-axis (Qy).
    z_edges_nxs : np.ndarray
        The bin edges for the z-axis (Qz).
    q_min : float
        The minimum Qz limit for the 1D slice extraction.
    q_max : float
        The maximum Qz limit for the 1D slice extraction.
    y_plot_range : tuple of float
        The plot limits for the y-axis.
    z_plot_range : tuple of float
        The plot limits for the z-axis.
    savename : str
        The file path to save the generated plot.
    intensity_min : float or int, optional
        The minimum intensity to use for the log-scaled 2D plots. Defaults to 1.0.
    """
    import matplotlib.pyplot as plt
    from .plotting_utils import plot_q_1d, log_plot_2d, extract_range_to_1d

    # Ensure minimum intensity for log plotting
    if intensity_min is not None:
        intensity_min = float(intensity_min)
    else:
        intensity_min = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot raw 2D
    log_plot_2d(hist_raw, y_edges_nxs, z_edges_nxs, "Raw NeXus data", ax=axes[0, 0],
                intensity_min=intensity_min, intensity_max=hist_raw.max(),
                y_range=y_plot_range, z_range=z_plot_range, output='none')

    # Plot masked 2D
    log_plot_2d(hist_masked, y_edges_nxs, z_edges_nxs, "Masked NeXus data", ax=axes[0, 1],
                intensity_min=intensity_min, intensity_max=hist_raw.max(),
                y_range=y_plot_range, z_range=z_plot_range, output='none')

    gs = axes[1, 0].get_gridspec()
    axes[1, 0].remove()
    axes[1, 1].remove()
    ax_bottom = fig.add_subplot(gs[1:, :])

    qz_min_index = np.digitize(q_min, z_edges_nxs) - 1
    qz_max_index = np.digitize(q_max, z_edges_nxs)

    # For 1D extraction, replace NaN with 0 so np.sum works properly
    hist_masked_1d = np.nan_to_num(hist_masked, nan=0.0)

    values_raw, errors_raw, y_bins_nxs, z_limits = extract_range_to_1d(
        hist_raw, hist_raw_error, y_edges_nxs, z_edges_nxs, [qz_min_index, qz_max_index]
    )
    plot_q_1d(values_raw, errors_raw, y_bins_nxs, 'Qy [1/nm]', color='blue',
              title_text='', label='Raw data', ax=ax_bottom, limits=y_plot_range, output='none')

    values_masked, errors_masked, y_bins_nxs, _ = extract_range_to_1d(
        hist_masked_1d, hist_masked_error, y_edges_nxs, z_edges_nxs, [qz_min_index, qz_max_index]
    )
    plot_q_1d(values_masked, errors_masked, y_bins_nxs, 'Qy [1/nm]', color='green',
              label='Masked data', ax=ax_bottom, limits=y_plot_range, output='none')

    axes[0, 0].axhline(z_edges_nxs[qz_min_index], color='magenta', linestyle='--')
    axes[0, 0].axhline(z_edges_nxs[qz_max_index], color='magenta', linestyle='--')
    axes[0, 1].axhline(z_edges_nxs[qz_min_index], color='magenta', linestyle='--')
    axes[0, 1].axhline(z_edges_nxs[qz_max_index], color='magenta', linestyle='--')

    # Format 1D overlay plot (grid only on the major ticks of this plot)
    ax_bottom.set_title(f"Qz=[{z_limits[0]:.4f} 1/nm, {z_limits[1]:.4f} 1/nm]")
    ax_bottom.grid(True, which='major')
    ax_bottom.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(savename, dpi=300)
    plt.close(fig)
    print(f"Created masks view plot: {savename}")
