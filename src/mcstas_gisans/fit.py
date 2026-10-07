#!/usr/bin/env python3

"""
Executes automated parameter fitting or parameter scans (grid sweeps) of BornAgain
simulations, evaluates the match against experimental NeXus data, and outputs fit metrics.
"""

import os
import copy
import csv
import itertools
import time
import argparse
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union

from .run_cli import parse_args as parse_run_args
from .input_output import get_particles
from .preconditioning import precondition
from .parameters import pack_parameters, set_outgoing_directions_from_sampling, build_instrument, angle_window_extent, angle_window_margin
from .run import process_particles, process_particles_parallelly
from .hardware import get_available_cores
from .nexus_reader import read_nexus_data, warn_if_duration_mismatch
from .experiment_time import upscale_simple
from .masking import get_mask, apply_mask, save_view_masks_plot

def format_time(seconds: Optional[float]) -> str:
    """
    Format time in seconds to a human-readable string (hours, minutes, seconds).

    Parameters
    ----------
    seconds : float or None
        Time in seconds.

    Returns
    -------
    str
        Formatted time string.
    """
    if seconds is None or seconds < 0:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{seconds:.2f}s"

from .fit_cli import create_fit_parser, parse_scan_arguments, parse_fit_arguments

def convert_val(value_str: str) -> Union[int, float, str]:
    """
    Convert a string value to integer or float if possible.

    Parameters
    ----------
    value_str : str
        The string to convert.

    Returns
    -------
    int, float, or str
        The converted value, or the original string if conversion fails.
    """
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str

def format_fit_value(v: float) -> str:
    """Format a fitted parameter value with 4 decimal places, falling back to
    scientific notation for nonzero values too small to show at that precision
    (e.g. an SLD on the order of 1e-6), which would otherwise print as 0.0000."""
    text = f"{v:.4f}"
    if v != 0 and float(text) == 0.0:
        return f"{v:.4e}"
    return text


LOSS_FUNCTIONS = ('poisson_deviance', 'reduced_chi2', 'log_residual')
MC_TO_POISSON_WARNING_RATIO = 1.0  # warn if the MC variance exceeds the counting variance in >5% of the pixels


def poisson_deviance_with_mc(counts: np.ndarray, expected: np.ndarray, mc_variance: np.ndarray) -> np.ndarray:
    """
    Per-pixel deviance 2 [ln P(N | N) - ln P(N | m, sigma^2)] of measured counts N against a
    simulated expectation m with Monte Carlo variance sigma^2.

    The finite simulation statistics are modelled by a gamma-distributed expectation (mean m,
    variance sigma^2), which makes N negative-binomially distributed with mean m and variance
    m + sigma^2 (effective likelihood for weighted Monte Carlo, cf. Arguelles, Schneider & Yuan,
    JHEP 06 (2019) 030). Limits: sigma -> 0 gives the Poisson deviance 2 [m - N + N ln(N/m)]
    (unbiased also at low counts); at high counts it approaches (N - m)^2 / (m + sigma^2).
    P(N | N) is the saturated Poisson term, so a perfect model gives ~1 per pixel.
    """
    from scipy.special import gammaln, betaln, xlogy
    counts = np.asarray(counts, dtype=float)
    m = np.maximum(np.asarray(expected, dtype=float), 1e-12)
    var = np.maximum(np.asarray(mc_variance, dtype=float), 0.0)

    saturated = xlogy(counts, counts) - counts - gammaln(counts + 1)
    log_p = xlogy(counts, m) - m - gammaln(counts + 1)          # Poisson

    alpha = np.divide(m * m, var, out=np.full_like(m, np.inf), where=var > 0)
    use_nb = np.isfinite(alpha) & (alpha < 1e14)
    if np.any(use_nb):
        n = counts[use_nb]
        a = alpha[use_nb]
        r = var[use_nb] / m[use_nb]                                 # 1 / beta
        # ln Gamma(a + n) - ln Gamma(a) = gammaln(n) - betaln(a, n) for n >= 1 (stable for large a)
        lgamma_ratio = np.where(n > 0, gammaln(np.maximum(n, 1)) - betaln(a, np.maximum(n, 1)), 0.0)
        log_nb = (lgamma_ratio - gammaln(n + 1)
                  - a * np.log1p(r)                                 # a ln(beta / (1 + beta))
                  - n * np.log1p(1.0 / r))                          # n ln(1 / (1 + beta))
        log_p = log_p.copy()
        log_p[use_nb] = log_nb
    return 2.0 * (saturated - log_p)


def calculate_fitness(
    hist_nxs: np.ndarray,
    hist_sim: np.ndarray,
    hist_sim_mc_error: np.ndarray
) -> Dict[str, float]:
    """
    Goodness-of-fit metrics between measured counts and simulated expected counts.

    Pixels where either input is NaN (masked) are excluded; n is the number of used pixels.

    Parameters
    ----------
    hist_nxs : np.ndarray
        Measured counts N (NaN outside the mask).
    hist_sim : np.ndarray
        Simulated expected counts m over the measurement time, including background.
    hist_sim_mc_error : np.ndarray
        Monte Carlo (statistical) uncertainty of hist_sim.

    Returns
    -------
    dict
        With the keys:

        - ``poisson_deviance``: mean per-pixel deviance of poisson_deviance_with_mc (Poisson
          likelihood, with the Monte Carlo uncertainty of the simulation folded in).
          ~1 for a perfect model; unbiased also at low counts.
        - ``reduced_chi2``: 1/n * sum[(N - m)^2 / (m + sigma_MC^2)] (Pearson chi^2 with the
          model's Poisson variance plus the Monte Carlo variance; biased at low counts).
        - ``log_residual``: mean of (log10 N - log10 m)^2 over pixels where both are positive.
        - ``mc_to_poisson_variance``: 95th percentile of sigma_MC^2 / m over the pixels, i.e. how
          large the Monte Carlo variance of the simulation is compared to the counting variance.
    """
    keep = np.isfinite(hist_nxs) & np.isfinite(hist_sim)
    n_valid = int(np.sum(keep))
    if n_valid == 0:
        return {key: np.nan for key in LOSS_FUNCTIONS + ('mc_to_poisson_variance',)}

    counts = hist_nxs[keep]
    expected = hist_sim[keep]
    mc_variance = hist_sim_mc_error[keep] ** 2

    poisson_deviance = float(np.sum(poisson_deviance_with_mc(counts, expected, mc_variance)) / n_valid)

    variance = expected + mc_variance
    variance = np.where(variance > 0, variance, 1.0)
    reduced_chi2 = float(np.sum((counts - expected) ** 2 / variance) / n_valid)

    both_positive = (counts > 0) & (expected > 0)
    if np.any(both_positive):
        log_residual = float(np.mean((np.log10(counts[both_positive]) - np.log10(expected[both_positive])) ** 2))
    else:
        log_residual = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = mc_variance / expected
    ratio = ratio[np.isfinite(ratio)]
    mc_to_poisson_variance = float(np.percentile(ratio, 95)) if ratio.size else np.nan

    return {
        'poisson_deviance': poisson_deviance,
        'reduced_chi2': reduced_chi2,
        'log_residual': log_residual,
        'mc_to_poisson_variance': mc_to_poisson_variance,
    }

def fit_flat_background(
    hist_nxs: np.ndarray,
    hist_signal: np.ndarray,
    hist_signal_mc_error: np.ndarray,
    loss_function: str
) -> Tuple[float, float]:
    """
    Flat background that minimises a loss between measured and simulated counts (``--fit_background``).

    The background b is searched by a bounded 1D minimisation between 0 and the mean measured
    counts of the used pixels (a flat level above the mean cannot fit), so it costs no extra
    simulation. Minimising over b for each parameter set gives the same optimum as fitting b as a
    parameter of the fit.

    Parameters
    ----------
    hist_nxs : np.ndarray
        Measured counts N (NaN outside the mask).
    hist_signal : np.ndarray
        Simulated expected counts over the measurement time, without background.
    hist_signal_mc_error : np.ndarray
        Monte Carlo uncertainty of hist_signal.
    loss_function : str
        Key of calculate_fitness to minimise (one of LOSS_FUNCTIONS).

    Returns
    -------
    tuple
        (b, loss at b), b in counts per pixel; (0, NaN) if no pixel can be used.
    """
    from scipy.optimize import minimize_scalar
    keep = np.isfinite(hist_nxs) & np.isfinite(hist_signal)
    counts, signal, error = hist_nxs[keep], hist_signal[keep], hist_signal_mc_error[keep]
    if counts.size == 0:
        return 0.0, np.nan
    loss = lambda b: calculate_fitness(counts, signal + b, error)[loss_function]
    b_max = max(float(np.mean(counts)), 1e-6)
    res = minimize_scalar(loss, bounds=(0.0, b_max), method='bounded', options={'xatol': 1e-3 * b_max})
    best_b, best_loss = float(res.x), float(res.fun)
    loss_at_zero = loss(0.0)  # the bounded method does not evaluate the end points themselves
    if loss_at_zero <= best_loss:
        best_b, best_loss = 0.0, loss_at_zero
    return best_b, best_loss


def save_comparison_plot(
    hist_nxs: np.ndarray,
    hist_nxs_error: np.ndarray,
    y_edges_nxs: np.ndarray,
    z_edges_nxs: np.ndarray,
    hist_sim: np.ndarray,
    hist_sim_error: np.ndarray,
    y_edges_sim: np.ndarray,
    z_edges_sim: np.ndarray,
    q_min: float,
    q_max: float,
    y_plot_range: List[float],
    z_plot_range: List[float],
    savename: str,
    label_sim: str,
    intensity_min: Optional[float] = None,
    split_view: bool = False
) -> None:
    """
    Save a 2x2 comparison plot of experimental and simulated data (with split_view: one 2D map
    with the measurement for Qy < 0 and the simulation for Qy > 0, above the 1D slice of the
    same width). The axes of the 2D maps and the Qy axis of the 1D slice are shared (zooming one
    zooms all in the interactive viewer).

    Parameters
    ----------
    hist_nxs : np.ndarray
        NeXus data histogram.
    hist_nxs_error : np.ndarray
        NeXus data errors.
    y_edges_nxs : np.ndarray
        NeXus y edges.
    z_edges_nxs : np.ndarray
        NeXus z edges.
    hist_sim : np.ndarray
        Simulated data histogram.
    hist_sim_error : np.ndarray
        Simulated data errors.
    y_edges_sim : np.ndarray
        Simulation y edges.
    z_edges_sim : np.ndarray
        Simulation z edges.
    q_min : float
        Minimum Qz for 1D slice extraction.
    q_max : float
        Maximum Qz for 1D slice extraction.
    y_plot_range : list of float
        Plotting range for y axis.
    z_plot_range : list of float
        Plotting range for z axis.
    savename : str
        Path to save the generated plot.
    label_sim : str
        Label for the simulated data plot.
    intensity_min : float, optional
        Minimum intensity for color scaling.
    split_view : bool, optional
        One split 2D map instead of two separate maps.
    """
    import matplotlib.pyplot as plt
    from .plotting_utils import plot_q_1d, log_plot_2d, extract_range_to_1d, split_plot_2d, link_axes

    if intensity_min is not None:
        intensity_min = float(intensity_min)
    else:
        intensity_min = 1.0
    intensity_max = hist_nxs[~np.isnan(hist_nxs)].max()

    if split_view:
        # one column: the 1D slice has the width of the map above it (the colorbar is attached
        # to the right of the map instead of taking its width, so that the Qy axes line up)
        fig, (ax_map, ax_bottom) = plt.subplots(2, 1, figsize=(10, 12))
        split_plot_2d(hist_nxs, y_edges_nxs, z_edges_nxs, hist_sim, y_edges_sim, z_edges_sim,
                      label_left="D22 measurement", label_right="Simulation", title_text=label_sim, ax=ax_map,
                      intensity_min=intensity_min, intensity_max=intensity_max,
                      y_range=y_plot_range, z_range=z_plot_range, match_horizontal_axes=True)
        map_axes = [ax_map]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 2D maps (no grid is added to these)
        log_plot_2d(hist_nxs, y_edges_nxs, z_edges_nxs, "D22 measurement", ax=axes[0, 0],
                    intensity_min=intensity_min, intensity_max=intensity_max,
                    y_range=y_plot_range, z_range=z_plot_range, output='none')

        log_plot_2d(hist_sim, y_edges_sim, z_edges_sim, label_sim, ax=axes[0, 1],
                    intensity_min=intensity_min, intensity_max=intensity_max,
                    y_range=y_plot_range, z_range=z_plot_range, output='none')
        map_axes = [axes[0, 0], axes[0, 1]]

        gs = axes[1, 0].get_gridspec()
        axes[1, 0].remove()
        axes[1, 1].remove()
        ax_bottom = fig.add_subplot(gs[1:, :])
    # zooming one map zooms all of them, and the Qy axis of the 1D slice
    link_axes(map_axes, ax_bottom)

    qz_min_index = np.digitize(q_min, z_edges_sim) - 1
    qz_max_index = np.digitize(q_max, z_edges_sim) - 1

    # For 1D extraction, replace NaN with 0 so np.sum works properly
    hist_nxs_1d = np.nan_to_num(hist_nxs, nan=0.0)
    hist_sim_1d = np.nan_to_num(hist_sim, nan=0.0)

    values_nxs, errors_nxs, y_bins_nxs, z_limits = extract_range_to_1d(
        hist_nxs_1d, hist_nxs_error, y_edges_nxs, z_edges_nxs, [qz_min_index, qz_max_index]
    )
    plot_q_1d(values_nxs, errors_nxs, y_bins_nxs, 'Qy [1/nm]', color='blue',
              title_text='', label='D22 measurement', ax=ax_bottom, limits=y_plot_range, output='none')

    values_sim, errors_sim, y_bins_sim, _ = extract_range_to_1d(
        hist_sim_1d, hist_sim_error, y_edges_sim, z_edges_sim, [qz_min_index, qz_max_index]
    )
    plot_q_1d(values_sim, errors_sim, y_bins_sim, 'Qy [1/nm]', color='green',
              label=label_sim, ax=ax_bottom, limits=y_plot_range, output='none')

    for ax in map_axes:
        for z_limit in z_limits:  # the summed Qz range
            ax.axhline(z_limit, color='magenta', linestyle='--')

    # Format 1D overlay plot (grid only on the major ticks of this plot)
    ax_bottom.set_title(f"Qz=[{z_limits[0]:.4f} 1/nm, {z_limits[1]:.4f} 1/nm]")
    ax_bottom.grid(True, which='major')
    ax_bottom.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(savename, dpi=300)
    plt.close(fig)
    print(f"Created comparison plot: {savename}")

def parse_joint_fit_arguments(args: Any) -> Tuple[List[str], List[float], List[Tuple[Optional[float], Optional[float]]], Dict[str, int], Dict[str, int]]:
    """
    Parse parameter fit definitions for single or joint fitting.

    Combines --fit_common (common to both samples), --fit (sample 1), and --fit2 (sample 2).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    tuple
        param_names : list of str
            List of parameter display names in optimization vector x.
        x0_list : list of float
            List of initial values.
        bounds_list : list of tuple
            List of (min, max) bounds.
        s1_map : dict
            Mapping of internal sample 1 parameter name to index in x.
        s2_map : dict
            Mapping of internal sample 2 parameter name to index in x.
    """
    common_names, common_x0, common_bounds = parse_fit_arguments(args.fit_common) if args.fit_common else ([], [], [])
    s1_names, s1_x0, s1_bounds = parse_fit_arguments(args.fit) if args.fit else ([], [], [])
    s2_names, s2_x0, s2_bounds = parse_fit_arguments(args.fit2) if args.fit2 else ([], [], [])

    param_names = []
    x0_list = []
    bounds_list = []

    s1_map = {}
    s2_map = {}

    # 1. Common parameters apply to both sample 1 and sample 2
    for name, x0, b in zip(common_names, common_x0, common_bounds):
        idx = len(param_names)
        param_names.append(name)
        x0_list.append(x0)
        bounds_list.append(b)
        s1_map[name] = idx
        s2_map[name] = idx

    # 2. Sample 1 specific parameters
    for name, x0, b in zip(s1_names, s1_x0, s1_bounds):
        idx = len(param_names)
        display_name = f"s1_{name}" if (name in s2_names or name in common_names) else name
        param_names.append(display_name)
        x0_list.append(x0)
        bounds_list.append(b)
        s1_map[name] = idx

    # 3. Sample 2 specific parameters
    for name, x0, b in zip(s2_names, s2_x0, s2_bounds):
        idx = len(param_names)
        display_name = f"s2_{name}" if (name in s1_names or name in common_names) else name
        param_names.append(display_name)
        x0_list.append(x0)
        bounds_list.append(b)
        s2_map[name] = idx

    return param_names, x0_list, bounds_list, s1_map, s2_map

def make_secondary_args(args: Any) -> Any:
    """
    Create secondary arguments object for Sample 2 evaluation.

    Parameters
    ----------
    args : argparse.Namespace
        Original parsed arguments.

    Returns
    -------
    argparse.Namespace
        A deep copy of args, modified for sample 2.
    """
    args2 = copy.deepcopy(args)
    if getattr(args, 'nxs2', None):
        args2.nxs = args.nxs2
    if getattr(args, 'sample_arguments2', None):
        args2.sample_arguments = args.sample_arguments2
    if getattr(args, 'filename2', None):
        args2.filename = args.filename2
    if getattr(args, 'intensity_factor2', None) is not None:
        args2.intensity_factor = args.intensity_factor2
    if getattr(args, 'alpha2', None) is not None:
        args2.alpha = args.alpha2
    if getattr(args, 'experiment_time2', None) is not None:
        args2.experiment_time = args.experiment_time2
    if getattr(args, 'background2', None) is not None:
        args2.background = args.background2
    return args2

def save_joint_comparison_plot(
    hist_nxs1: np.ndarray,
    hist_nxs_error1: np.ndarray,
    y_edges_nxs1: np.ndarray,
    z_edges_nxs1: np.ndarray,
    hist_sim_masked1: np.ndarray,
    hist_sim_error_masked1: np.ndarray,
    edges_sim1_0: np.ndarray,
    edges_sim1_1: np.ndarray,
    hist_nxs2: np.ndarray,
    hist_nxs_error2: np.ndarray,
    y_edges_nxs2: np.ndarray,
    z_edges_nxs2: np.ndarray,
    hist_sim_masked2: np.ndarray,
    hist_sim_error_masked2: np.ndarray,
    edges_sim2_0: np.ndarray,
    edges_sim2_1: np.ndarray,
    q_min: float,
    q_max: float,
    y_plot_range: List[float],
    z_plot_range: List[float],
    savename: str,
    label_sim1: str = "Sample 1 Sim",
    label_sim2: str = "Sample 2 Sim",
    split_view: bool = False
) -> None:
    """
    Save a 3x2 joint comparison plot of experimental and simulated data for two samples (with
    split_view: one 2D map per sample, the measurement for Qy < 0 and the simulation for Qy > 0).
    The axes of the 2D maps and the Qy axis of the 1D slice are shared.

    Parameters
    ----------
    hist_nxs1 : np.ndarray
        NeXus data histogram for sample 1.
    hist_nxs_error1 : np.ndarray
        NeXus data errors for sample 1.
    y_edges_nxs1 : np.ndarray
        NeXus y edges for sample 1.
    z_edges_nxs1 : np.ndarray
        NeXus z edges for sample 1.
    hist_sim_masked1 : np.ndarray
        Simulated data histogram for sample 1.
    hist_sim_error_masked1 : np.ndarray
        Simulated data errors for sample 1.
    edges_sim1_0 : np.ndarray
        Simulation y edges for sample 1.
    edges_sim1_1 : np.ndarray
        Simulation z edges for sample 1.
    hist_nxs2 : np.ndarray
        NeXus data histogram for sample 2.
    hist_nxs_error2 : np.ndarray
        NeXus data errors for sample 2.
    y_edges_nxs2 : np.ndarray
        NeXus y edges for sample 2.
    z_edges_nxs2 : np.ndarray
        NeXus z edges for sample 2.
    hist_sim_masked2 : np.ndarray
        Simulated data histogram for sample 2.
    hist_sim_error_masked2 : np.ndarray
        Simulated data errors for sample 2.
    edges_sim2_0 : np.ndarray
        Simulation y edges for sample 2.
    edges_sim2_1 : np.ndarray
        Simulation z edges for sample 2.
    q_min : float
        Minimum Qz for 1D slice extraction.
    q_max : float
        Maximum Qz for 1D slice extraction.
    y_plot_range : list of float
        Plotting range for y axis.
    z_plot_range : list of float
        Plotting range for z axis.
    savename : str
        Path to save the generated plot.
    label_sim1 : str, optional
        Label for the sample 1 simulated data plot.
    label_sim2 : str, optional
        Label for the sample 2 simulated data plot.
    split_view : bool, optional
        One split 2D map per sample instead of two separate maps.
    """
    import matplotlib.pyplot as plt
    from .plotting_utils import plot_q_1d, log_plot_2d, extract_range_to_1d, split_plot_2d, link_axes

    intensity_min = 1.0
    vmax1 = hist_nxs1[~np.isnan(hist_nxs1)].max() if np.any(~np.isnan(hist_nxs1)) else 100.0
    vmax2 = hist_nxs2[~np.isnan(hist_nxs2)].max() if np.any(~np.isnan(hist_nxs2)) else 100.0
    if split_view:
        # Row 0: one split map per sample; row 1: the 1D Q-slice overlay
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        split_plot_2d(hist_nxs1, y_edges_nxs1, z_edges_nxs1, hist_sim_masked1, edges_sim1_0, edges_sim1_1,
                      label_left="Sample 1 Measurement", label_right=label_sim1, ax=axes[0, 0],
                      intensity_min=intensity_min, intensity_max=vmax1, y_range=y_plot_range, z_range=z_plot_range)
        split_plot_2d(hist_nxs2, y_edges_nxs2, z_edges_nxs2, hist_sim_masked2, edges_sim2_0, edges_sim2_1,
                      label_left="Sample 2 Measurement", label_right=label_sim2, ax=axes[0, 1],
                      intensity_min=intensity_min, intensity_max=vmax2, y_range=y_plot_range, z_range=z_plot_range)
        bottom_row = 1
        map_axes_1, map_axes_2 = [(axes[0, 0], edges_sim1_1)], [(axes[0, 1], edges_sim2_1)]
    else:
        fig, axes = plt.subplots(3, 2, figsize=(16, 16))

        # 1. Row 0: Sample 1 NeXus & Sim
        log_plot_2d(hist_nxs1, y_edges_nxs1, z_edges_nxs1, "Sample 1 Measurement", ax=axes[0, 0],
                    intensity_min=intensity_min, intensity_max=vmax1,
                    y_range=y_plot_range, z_range=z_plot_range, output='none')
        log_plot_2d(hist_sim_masked1, edges_sim1_0, edges_sim1_1, label_sim1, ax=axes[0, 1],
                    intensity_min=intensity_min, intensity_max=vmax1,
                    y_range=y_plot_range, z_range=z_plot_range, output='none')

        # 2. Row 1: Sample 2 NeXus & Sim
        log_plot_2d(hist_nxs2, y_edges_nxs2, z_edges_nxs2, "Sample 2 Measurement", ax=axes[1, 0],
                    intensity_min=intensity_min, intensity_max=vmax2,
                    y_range=y_plot_range, z_range=z_plot_range, output='none')
        log_plot_2d(hist_sim_masked2, edges_sim2_0, edges_sim2_1, label_sim2, ax=axes[1, 1],
                    intensity_min=intensity_min, intensity_max=vmax2,
                    y_range=y_plot_range, z_range=z_plot_range, output='none')
        bottom_row = 2
        map_axes_1 = [(axes[0, 0], z_edges_nxs1), (axes[0, 1], edges_sim1_1)]
        map_axes_2 = [(axes[1, 0], z_edges_nxs2), (axes[1, 1], edges_sim2_1)]

    # Last row: merged for the 1D Q-slice overlay
    gs = axes[bottom_row, 0].get_gridspec()
    axes[bottom_row, 0].remove()
    axes[bottom_row, 1].remove()
    ax_bottom = fig.add_subplot(gs[bottom_row, :])
    # zooming one map zooms all of them, and the Qy axis of the 1D slice
    link_axes([ax for ax, _ in map_axes_1 + map_axes_2], ax_bottom)

    qz_min_index1 = np.digitize(q_min, edges_sim1_1) - 1
    qz_max_index1 = np.digitize(q_max, edges_sim1_1) - 1

    qz_min_index2 = np.digitize(q_min, edges_sim2_1) - 1
    qz_max_index2 = np.digitize(q_max, edges_sim2_1) - 1

    # Sample 1 1D curves
    hist_nxs1_1d = np.nan_to_num(hist_nxs1, nan=0.0)
    hist_sim1_1d = np.nan_to_num(hist_sim_masked1, nan=0.0)
    val_nxs1, err_nxs1, y_bins_nxs1, z_limits1 = extract_range_to_1d(
        hist_nxs1_1d, hist_nxs_error1, y_edges_nxs1, z_edges_nxs1, [qz_min_index1, qz_max_index1]
    )
    plot_q_1d(val_nxs1, err_nxs1, y_bins_nxs1, 'Qy [1/nm]', color='blue',
              title_text='', label='Sample 1 Measurement', ax=ax_bottom, limits=y_plot_range, output='none')

    val_sim1, err_sim1, y_bins_sim1, _ = extract_range_to_1d(
        hist_sim1_1d, hist_sim_error_masked1, edges_sim1_0, edges_sim1_1, [qz_min_index1, qz_max_index1]
    )
    plot_q_1d(val_sim1, err_sim1, y_bins_sim1, 'Qy [1/nm]', color='cyan',
              label=label_sim1, ax=ax_bottom, limits=y_plot_range, output='none')

    # Sample 2 1D curves
    hist_nxs2_1d = np.nan_to_num(hist_nxs2, nan=0.0)
    hist_sim2_1d = np.nan_to_num(hist_sim_masked2, nan=0.0)
    val_nxs2, err_nxs2, y_bins_nxs2, _ = extract_range_to_1d(
        hist_nxs2_1d, hist_nxs_error2, y_edges_nxs2, z_edges_nxs2, [qz_min_index2, qz_max_index2]
    )
    plot_q_1d(val_nxs2, err_nxs2, y_bins_nxs2, 'Qy [1/nm]', color='red',
              label='Sample 2 Measurement', ax=ax_bottom, limits=y_plot_range, output='none')

    val_sim2, err_sim2, y_bins_sim2, _ = extract_range_to_1d(
        hist_sim2_1d, hist_sim_error_masked2, edges_sim2_0, edges_sim2_1, [qz_min_index2, qz_max_index2]
    )
    plot_q_1d(val_sim2, err_sim2, y_bins_sim2, 'Qy [1/nm]', color='orange',
              label=label_sim2, ax=ax_bottom, limits=y_plot_range, output='none')

    # Highlight Qz slice on 2D plots
    for ax, z_edges, qz_min_idx, qz_max_idx in (
        [(ax, z_edges, qz_min_index1, qz_max_index1) for ax, z_edges in map_axes_1] +
        [(ax, z_edges, qz_min_index2, qz_max_index2) for ax, z_edges in map_axes_2]
    ):
        ax.axhline(z_edges[min(max(qz_min_idx, 0), len(z_edges) - 2)], color='magenta', linestyle='--')
        ax.axhline(z_edges[min(max(qz_max_idx, 0), len(z_edges) - 2) + 1], color='magenta', linestyle='--')

    ax_bottom.set_title(f"Qz=[{z_limits1[0]:.4f} 1/nm, {z_limits1[1]:.4f} 1/nm]")
    ax_bottom.grid(True, which='major')
    ax_bottom.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(savename, dpi=300)
    plt.close(fig)
    print(f"Created joint comparison plot: {savename}")

def validate_fit_args(args: Any, parser: argparse.ArgumentParser) -> None:
    """
    Validate command-line arguments specific to fitting.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    parser : argparse.ArgumentParser
        The argument parser instance for raising errors.
    """
    has_fit = bool(args.fit or args.fit_common or args.fit2)
    if not args.mask_view:
        if not args.filename:
            parser.error("the following arguments are required: filename")
        if not args.scan and not has_fit:
            parser.error("Either --scan, --fit, --fit2, or --fit_common must be specified.")
        if args.scan and has_fit:
            parser.error("--scan cannot be combined with --fit/--fit2/--fit_common.")
        if args.scan and (args.nxs2 or args.sample_arguments2):
            parser.error("Joint (two-sample) options are only supported for fits, not for --scan.")
        if not args.experiment_time:
            parser.error("--experiment_time is required: the simulated rates must be scaled to expected counts to be compared with the measured counts.")
        if getattr(args, 'instrument_params', {}).get('tof_instrument', False):
            parser.error("Fitting is not implemented for TOF instruments yet.")
    if not args.nxs:
        parser.error("the following arguments are required: --nxs")
    if getattr(args, 'fit_background', False) and (args.background is not None or getattr(args, 'background2', None) is not None):
        parser.error("--fit_background fits the flat background for each evaluation: it cannot be combined with a fixed --background/--background2.")

    if has_fit:
        try:
            param_names, _, _, _, _ = parse_joint_fit_arguments(args)
        except ValueError as err:
            parser.error(str(err))
        if getattr(args, 'fit_integer', None):
            base_names = {name[3:] if name.startswith(('s1_', 's2_')) else name for name in param_names}
            for item in args.fit_integer:
                for name in item:
                    if name not in param_names and name not in base_names:
                        parser.error(f"--fit_integer {name}: '{name}' is not a fitted parameter ({param_names}).")

    if (args.fit2 or args.sample_arguments2) and not args.nxs2:
        args.nxs2 = args.nxs  # Default secondary NeXus dataset to primary NeXus dataset if omitted

    if (args.fit or args.fit2 or args.fit_common) and args.optimizer.lower() == 'differential-evolution':
        param_names, _, bounds, _, _ = parse_joint_fit_arguments(args)
        for name, (low, high) in zip(param_names, bounds):
            if low is None or high is None:
                parser.error(f"Differential Evolution optimizer requires finite bounds for all fitted parameters. Please specify bounds in --fit/--fit2/--fit_common for '{name}' (e.g. --fit {name} <initial_guess> <min_bound> <max_bound>).")

def prepare_experimental_data(args: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare and load experimental NeXus data and apply masks.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    tuple
        (hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw)
    """
    wavelength_val = args.wavelength_selected if args.wavelength_selected else (args.wavelength if args.wavelength else 6.0)

    from .instrument import Instrument
    from .instrument_defaults import get_instrument_parameters
    instr_params = get_instrument_parameters(args)
    instrument = Instrument(instr_params, args.alpha, wavelength_val, args.sample_orientation, args.wfm, args.no_gravity)

    nxs_paths = args.nxs if isinstance(args.nxs, list) else [args.nxs]
    nxs_data_path = getattr(args, 'nxs_data_path', None)

    print(f"Loading experimental NeXus data from {len(nxs_paths)} file(s):")
    hist_list = []
    y_edges_nxs, z_edges_nxs = None, None
    for path in nxs_paths:
        hist, _, y_edges_nxs, z_edges_nxs = read_nexus_data(path, instrument, data_path=nxs_data_path)
        print(f"  {path}: shape={hist.shape}, counts={np.sum(hist):.0f}")
        hist_list.append(hist)

    if len(hist_list) > 1:
        shapes = {h.shape for h in hist_list}
        if len(shapes) > 1:
            raise ValueError(f"Incompatible NeXus data shapes across files: {shapes}")

    hist_nxs_raw = np.sum(hist_list, axis=0)
    hist_nxs_error_raw = np.sqrt(hist_nxs_raw)
    print(f"Summed NeXus dataset of shape {hist_nxs_raw.shape}, total counts={np.sum(hist_nxs_raw):.0f}")

    warn_if_duration_mismatch(nxs_paths, args.experiment_time, label=f"{len(nxs_paths)} --nxs file(s)")

    mask = get_mask(
        y_edges_nxs, z_edges_nxs,
        mask_qy_range=args.mask_qy_range,
        qy_min_cut=args.mask_qy_min_cut, qy_max_cut=args.mask_qy_max_cut,
        qz_min_cut=args.mask_qz_min_cut, qz_max_cut=args.mask_qz_max_cut,
        exclude_q_box=args.mask_exclude_q_box,
        include_q_box=args.mask_include_q_box,
        shape=hist_nxs_raw.shape
    )

    hist_nxs = apply_mask(hist_nxs_raw, mask, np.nan)
    hist_nxs_error = apply_mask(hist_nxs_error_raw, mask, 0.0)
    n_unmasked = int(np.sum(np.isfinite(hist_nxs)))
    print(f"Unmasked detector pixels used for the comparison: {n_unmasked}")
    if n_unmasked == 0 and not args.mask_view:
        raise ValueError("The mask excludes every detector pixel; nothing is left to compare.")

    if getattr(args, 'simulate_mask_angle_range', False):
        # the range enclosing the unmasked pixels as seen from the sample centre; with
        # --simulate_mask_angle_range_factor auto every neutron gets its own window of directions that can reach it
        # (neutron_angle_windows, in process_particles), with a numeric factor the range scaled by it is simulated
        args.mask_angle_range = list(instrument.get_masked_angle_range(mask))
        factor = getattr(args, 'simulate_mask_angle_range_factor', 'auto')
        if factor == 'auto':
            args.angle_window_region = list(args.mask_angle_range)
            args.angle_range = args.mask_angle_range  # replaced by the union of the windows (_set_simulated_angle_range)
        else:
            args.angle_window_region = None  # (args2 of a joint fit is a copy of args)
            args.angle_range = list(instrument.get_masked_angle_range(mask, factor=factor))
        fmt = lambda r: f"horiz=[{r[0]:.4f}, {r[1]:.4f}], vert=[{r[2]:.4f}, {r[3]:.4f}]"
        print(f"Mask angle range [deg] (the unmasked pixels seen from the sample centre): {fmt(args.mask_angle_range)}"
              + ("" if factor == 'auto' else f"; simulated with factor {factor:g}: {fmt(args.angle_range)}"))

    return hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw

def load_and_precondition_particles(args: Any) -> Tuple[Any, str, Any]:
    """
    Load particle data and precondition it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    tuple
        (particles, particle_type, mcpl_metadata)
    """
    from .tof_filtering import get_tof_filtering_limits
    tof_limits = get_tof_filtering_limits(args)
    particles, particle_type, mcpl_metadata = get_particles(
        args.filename, args.intensity_factor, tof_limits, args.input_weight_limit, use_polarization=args.use_polarization
    )
    particles = precondition(particles, args)
    if len(particles) == 0:
        raise ValueError("No incident particles are left to simulate (none hit the sample). Check --alpha, the sample size "
                         "and --sample_orientation, or use --allow_sample_miss.")
    print(f"Loaded and preconditioned {len(particles)} particles.")
    return particles, particle_type, mcpl_metadata

def run_simulation_evaluation(
    grid_point: Dict[str, Any],
    args: Any,
    particles: Any,
    particle_type: str,
    hist_nxs: np.ndarray,
    hist_nxs_error: np.ndarray,
    y_edges_nxs: np.ndarray,
    z_edges_nxs: np.ndarray,
    mask: np.ndarray,
    label_prefix: str = "sim",
    save_simulation_output: bool = False,
    mcpl_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    """
    Run a single simulation point evaluation and return fitness metrics.

    Parameters
    ----------
    grid_point : dict
        Parameters to evaluate.
    args : argparse.Namespace
        Command-line arguments.
    particles : Any
        Preconditioned simulation particles.
    particle_type : str
        Particle type identifier.
    hist_nxs : np.ndarray
        NeXus data histogram.
    hist_nxs_error : np.ndarray
        NeXus data errors.
    y_edges_nxs : np.ndarray
        NeXus y edges.
    z_edges_nxs : np.ndarray
        NeXus z edges.
    mask : np.ndarray
        Mask array.
    label_prefix : str, optional
        Prefix for saved outputs.
    save_simulation_output : bool, optional
        Whether to save scipp results.
    mcpl_metadata : dict, optional
        MCPL source metadata (from `get_particles`) to embed when saving scipp results.

    Returns
    -------
    tuple
        (metrics, record, sim_data); metrics as returned by calculate_fitness.
    """
    sample_args_dict = {}
    if args.sample_arguments:
        for pair in args.sample_arguments.split(';'):
            if '=' in pair:
                k, v = pair.split('=', 1)
                sample_args_dict[k.strip()] = convert_val(v.strip())

    for k, v in grid_point.items():
        sample_args_dict[k] = v

    args.sample_arguments = ';'.join(f"{k}={v}" for k, v in sample_args_dict.items())
    params = pack_parameters(args, particle_type)

    if args.no_parallel:
        result = process_particles(particles, params)
    else:
        process_number = args.parallel_processes if args.parallel_processes else max(1, get_available_cores() - 1)
        result = process_particles_parallelly(particles, params, process_number)

    instrument = params['instrument']

    raw_hist = result['pixelHist']
    raw_err = np.sqrt(result['pixelHistWeightsSquared'])

    hist_sim = instrument.detector.coords.rotate_detector_image(raw_hist)
    hist_sim_error = instrument.detector.coords.rotate_detector_image(raw_err)

    y_edges, z_edges = instrument.get_q_pixel_limits(wavelength=instrument.wavelength_selected)
    edges = [None, y_edges, z_edges] # match the expected edges list format

    param_str = '_'.join(f"{k}_{v}" for k, v in grid_point.items())
    if save_simulation_output:
        savename = os.path.join(args.output_dir, f"{label_prefix}_{param_str}")
        from .input_output import save_simulation_results_as_scipp
        temp_read_chunk_size = getattr(args, 'temp_read_chunk_size', 1000000) #this will be used for TOF instruments
        save_simulation_results_as_scipp(savename, params, result, args, mcpl_metadata, temp_read_chunk_size)

    if hist_nxs.shape != hist_sim.shape:
        if hist_nxs.shape == hist_sim.T.shape:
            hist_sim = hist_sim.T
            hist_sim_error = hist_sim_error.T
        else:
            raise ValueError(f"Incompatible shapes: NeXus={hist_nxs.shape}, Sim={hist_sim.shape}")

    # expected counts over the measurement time (deterministic: no Poisson sampling, which
    # would make the objective function random) and their Monte Carlo uncertainty
    # flat background: fixed (--background, default 0) or fitted to this pattern (--fit_background)
    fit_background = getattr(args, 'fit_background', False)
    background = 0.0 if (fit_background or args.background is None) else args.background
    hist_sim, hist_sim_mc_error = upscale_simple(
        hist_sim, hist_sim_error, args.experiment_time, background, poisson_sampling=False
    )
    if fit_background:
        background, _ = fit_flat_background(hist_nxs, hist_sim, hist_sim_mc_error, args.loss_function)
        hist_sim = hist_sim + background

    # hist_nxs is NaN outside the mask (prepare_experimental_data)
    metrics = calculate_fitness(hist_nxs, hist_sim, hist_sim_mc_error)
    metrics['background'] = background
    _warn_if_mc_uncertainty_large(metrics, args)

    # for display: expected spread of a measurement = Poisson + Monte Carlo
    hist_sim_error = np.sqrt(np.maximum(hist_sim, 0.0) + hist_sim_mc_error**2)

    # hist_sim_masked/_error_masked (NaN-filled) are only needed for the
    # comparison plot's display, not for the fitness calculation above.
    hist_sim_masked = apply_mask(hist_sim, mask, np.nan)
    hist_sim_error_masked = apply_mask(hist_sim_error, mask, 0.0)

    record = copy.deepcopy(grid_point)
    if fit_background:
        record['background'] = background
    for key in LOSS_FUNCTIONS:
        record[key] = metrics[key]

    if args.png:
        plot_path = os.path.join(args.output_dir, f"{label_prefix}_{param_str}.png")
        y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges_nxs[0], y_edges_nxs[-1]]
        z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges_nxs[0], z_edges_nxs[-1]]
        # Determine default intensity_min if not provided
        if args.intensity_min is not None:
            intensity_min = float(args.intensity_min)
        else:
            intensity_min = 1.0 if args.experiment_time else 1e-9

        save_comparison_plot(
            hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs,
            hist_sim_masked, hist_sim_error_masked, edges[1], edges[2],
            args.q_min, args.q_max, y_plot_range, z_plot_range,
            plot_path, f"Sim ({param_str})", intensity_min, split_view=getattr(args, 'split_view', False)
        )

    sim_data = {
        'hist_sim_masked': hist_sim_masked,
        'hist_sim_error_masked': hist_sim_error_masked,
        'edges': edges
    }

    return metrics, record, sim_data


_MC_WARNING_ISSUED = [False]


def _warn_if_mc_uncertainty_large(metrics: Dict[str, float], args: Any) -> None:
    """Warn once per run if the simulation's Monte Carlo variance is not small compared to the counting variance."""
    ratio = metrics.get('mc_to_poisson_variance', np.nan)
    if not _MC_WARNING_ISSUED[0] and np.isfinite(ratio) and ratio > MC_TO_POISSON_WARNING_RATIO:
        _MC_WARNING_ISSUED[0] = True
        print(f"WARNING: in more than 5% of the unmasked pixels the Monte Carlo variance of the simulation exceeds "
              f"the expected counting (Poisson) variance (95th percentile of their ratio: {ratio:.2f}). The loss "
              f"accounts for it, but the comparison is limited by the simulation statistics there; consider more "
              f"simulated statistics (more incident neutrons, or more --outgoing_directions).")

def save_summary_csv(records: List[Dict[str, Any]], output_dir: str, filename: str) -> None:
    """
    Save evaluation records to a CSV file.

    Parameters
    ----------
    records : list of dict
        List of evaluation records.
    output_dir : str
        Output directory.
    filename : str
        Output CSV file name.
    """
    if not records:
        return
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, filename)
    fieldnames = list(records[0].keys())
    with open(summary_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

def save_and_print_summary(
    records: List[Dict[str, Any]],
    output_dir: str,
    filename: str,
    title_header: str,
    extra_summary_text: Optional[str] = None,
    sort_key: str = 'poisson_deviance'
) -> None:
    """
    Save evaluation records to CSV and print a nicely formatted summary to standard output.

    Parameters
    ----------
    records : list of dict
        Evaluation records.
    output_dir : str
        Output directory path.
    filename : str
        CSV filename.
    title_header : str
        Header text for the summary section.
    extra_summary_text : str, optional
        Additional text to append to the summary output.
    """
    if records and sort_key in records[0]:
        records.sort(key=lambda r: (np.isnan(r[sort_key]), r[sort_key]))

    save_summary_csv(records, output_dir, filename)
    summary_path = os.path.join(output_dir, filename)

    print(f"\n{title_header} complete! Summary saved to: {summary_path}")

    summary_lines = []
    summary_lines.append(f"--- {title_header} Results (Sorted by {sort_key}) ---")
    if records:
        headers = list(records[0].keys())
        col_widths = {h: max(len(h), 12) for h in headers}

        for r in records:
            for h in headers:
                val_str = f"{r[h]:.4e}" if isinstance(r[h], float) else str(r[h])
                col_widths[h] = max(col_widths[h], len(val_str))

        header_row = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
        separator = "-+-".join("-" * col_widths[h] for h in headers)
        summary_lines.append(header_row)
        summary_lines.append(separator)
        for r in records:
            row_str = " | ".join(
                (f"{r[h]:<{col_widths[h]}.4e}" if isinstance(r[h], float)
                 else f"{str(r[h]):<{col_widths[h]}}")
                for h in headers
            )
            summary_lines.append(row_str)
    else:
        summary_lines.append("No records to display.")

    if extra_summary_text:
        summary_lines.append("\n--- Fit Results ---")
        summary_lines.append(extra_summary_text)

    summary_text_block = "\n".join(summary_lines)
    print("\n" + summary_text_block)

    if summary_path and os.path.exists(summary_path):
        with open(summary_path, mode='a') as f:
            f.write("\n\n" + summary_text_block + "\n")

def create_fit_evolution_gif(
    output_dir: str,
    gif_name: str = "fit_evolution.gif",
    duration: int = 500,
    is_joint: bool = False
) -> None:
    """
    Find all fit_eval_*.png files in output_dir, sort them by evaluation index,
    and compile them into an animated GIF.

    Parameters
    ----------
    output_dir : str
        Directory containing the PNG files.
    gif_name : str, optional
        Filename of the output GIF. Default is 'fit_evolution.gif'.
    duration : int, optional
        Duration of each frame in milliseconds. Default is 500.
    is_joint : bool, optional
        Whether the fitting process was joint (dual-sample).
    """
    import glob
    import re
    from PIL import Image

    if is_joint:
        pattern = os.path.join(output_dir, "fit_eval_joint_*.png")
        png_files = glob.glob(pattern)
        if not png_files:
            pattern = os.path.join(output_dir, "fit_eval_*.png")
            png_files = [f for f in glob.glob(pattern) if "_s1_" not in os.path.basename(f) and "_s2_" not in os.path.basename(f)]
    else:
        pattern = os.path.join(output_dir, "fit_eval_*.png")
        png_files = [f for f in glob.glob(pattern) if "_joint_" not in os.path.basename(f)]

    if not png_files:
        print("No fit evaluation PNG figures found to create GIF.")
        return

    def get_eval_index(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'(\d+)', basename)
        if match:
            return int(match.group(1))
        return 0

    png_files.sort(key=get_eval_index)

    images = [Image.open(f) for f in png_files]
    gif_path = os.path.join(output_dir, gif_name)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"Created animated fit evolution GIF: {gif_path}")

def run_automated_fit(
    args: Any,
    particles: Any,
    particle_type: str,
    hist_nxs: np.ndarray,
    hist_nxs_error: np.ndarray,
    y_edges_nxs: np.ndarray,
    z_edges_nxs: np.ndarray,
    mask: np.ndarray,
    mcpl_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Execute an automated parameter fit using scipy.optimize.

    Supports both single-sample fitting and joint dual-sample fitting.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    particles : Any
        Preconditioned simulation particles.
    particle_type : str
        Particle type identifier.
    hist_nxs : np.ndarray
        NeXus data histogram.
    hist_nxs_error : np.ndarray
        NeXus data errors.
    y_edges_nxs : np.ndarray
        NeXus y edges.
    z_edges_nxs : np.ndarray
        NeXus z edges.
    mask : np.ndarray
        Mask array.
    """
    import scipy.optimize
    if args.gif:
        args.png = True

    is_joint_fit = bool(getattr(args, 'nxs2', None) or getattr(args, 'fit2', None) or getattr(args, 'fit_common', None))

    if is_joint_fit:
        print("\n========== JOINT / DUAL-SAMPLE FITTING MODE ==========")
        args2 = make_secondary_args(args)
        hist_nxs2, hist_nxs_error2, y_edges_nxs2, z_edges_nxs2, mask2, _, _ = prepare_experimental_data(args2)

        # sample 2 needs its own particles if anything that get_particles/precondition use differs
        if (getattr(args, 'filename2', None)
                or (getattr(args, 'alpha2', None) is not None and args.alpha2 != args.alpha)
                or (getattr(args, 'intensity_factor2', None) is not None and args.intensity_factor2 != args.intensity_factor)):
            particles2, particle_type2, mcpl_metadata2 = load_and_precondition_particles(args2)
        else:
            particles2, particle_type2, mcpl_metadata2 = particles, particle_type, mcpl_metadata
        if getattr(args2, 'simulate_mask_angle_range', False):
            _set_simulated_angle_range(args2, particles2, particle_type2)
        set_outgoing_directions_from_sampling(args2, particles2, particle_type2)  # sample 2: its own particles and range

        param_names, x0, bounds, s1_map, s2_map = parse_joint_fit_arguments(args)
    else:
        param_names, x0, bounds = parse_fit_arguments(args.fit)
        s1_map = {name: idx for idx, name in enumerate(param_names)}
        s2_map = {}

    fit_integers = set()
    if getattr(args, 'fit_integer', None):
        for item in args.fit_integer:
            for name in item:
                fit_integers.add(name)

    print(f"\nStarting automated optimization using {args.optimizer.upper()} optimizer...")
    print(f"Parameters to fit: {param_names}")
    if fit_integers:
        print(f"Integer parameters: {sorted(list(fit_integers))}")
    print(f"Initial guess x0: {x0}")
    print(f"Bounds: {bounds}")
    print(f"Max evaluations: {args.max_evals}")
    print(f"Loss metric: {args.loss_function}")
    print(f"Convergence tolerances: xatol={args.xatol}, fatol={args.fatol}")

    eval_counter = [0]
    records = []
    start_total_time = time.time()
    fit_background = getattr(args, 'fit_background', False)
    if fit_background:
        print("Flat background: fitted for each evaluation (--fit_background)")

    def is_integer(name: str) -> bool:
        base_name = name[3:] if name.startswith(('s1_', 's2_')) else name
        return name in fit_integers or base_name in fit_integers

    def print_progress(message: str, eval_start_time: float) -> None:
        eval_duration = time.time() - eval_start_time
        avg_iter_time = (time.time() - start_total_time) / eval_counter[0]
        eta = avg_iter_time * max(0, args.max_evals - eval_counter[0])
        print(f"Fit Eval #{eval_counter[0]}/{args.max_evals}: {message} | Iter: {eval_duration:.2f}s | Avg: {avg_iter_time:.2f}s | ETA: {format_time(eta)}")

    def objective_function(x):
        eval_start_time = time.time()
        eval_counter[0] += 1

        # Map continuous variables to rounded integers where configured
        values = [int(np.round(val)) if is_integer(name) else float(val) for name, val in zip(param_names, x)]
        display_point = dict(zip(param_names, values))
        grid_point_s1 = {name: values[idx] for name, idx in s1_map.items()}

        # Bounds penalty (the optimizers are given the bounds; this guards rounding and DE edge cases)
        for name, val, (low, high) in zip(param_names, values, bounds):
            if (low is not None and val < low) or (high is not None and val > high):
                print_progress(f"Bound constraint violated ({name}: {val} outside [{low}, {high}]). Penalty applied", eval_start_time)
                return 1e9

        if not is_joint_fit:
            metrics, _, _ = run_simulation_evaluation(
                grid_point_s1, args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask,
                label_prefix=f"fit_eval_{eval_counter[0]}", mcpl_metadata=mcpl_metadata
            )
            rec = copy.deepcopy(display_point)
            rec['eval_index'] = eval_counter[0]
            if fit_background:
                rec['background'] = metrics['background']
            for key in LOSS_FUNCTIONS:
                rec[key] = metrics[key]
            loss = metrics[args.loss_function]
        else:
            grid_point_s2 = {name: values[idx] for name, idx in s2_map.items()}

            png_backup = getattr(args, 'png', False)
            args.png = False
            metrics1, _, sim_data1 = run_simulation_evaluation(
                grid_point_s1, args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask,
                label_prefix=f"fit_eval_s1_{eval_counter[0]}", mcpl_metadata=mcpl_metadata
            )
            metrics2, _, sim_data2 = run_simulation_evaluation(
                grid_point_s2, args2, particles2, particle_type2, hist_nxs2, hist_nxs_error2, y_edges_nxs2, z_edges_nxs2, mask2,
                label_prefix=f"fit_eval_s2_{eval_counter[0]}", mcpl_metadata=mcpl_metadata2
            )
            args.png = png_backup

            # joint loss: sum of the per-sample (per-pixel normalised) metrics
            rec = copy.deepcopy(display_point)
            rec['eval_index'] = eval_counter[0]
            if fit_background:
                rec['background_sample1'] = metrics1['background']
                rec['background_sample2'] = metrics2['background']
            for key in LOSS_FUNCTIONS:
                rec[f"{key}_sample1"] = metrics1[key]
                rec[f"{key}_sample2"] = metrics2[key]
                rec[key] = metrics1[key] + metrics2[key]
            loss = rec[args.loss_function]

            if args.png:
                plot_path = os.path.join(args.output_dir, f"fit_eval_joint_{eval_counter[0]:03d}.png")
                y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges_nxs[0], y_edges_nxs[-1]]
                z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges_nxs[0], z_edges_nxs[-1]]
                save_joint_comparison_plot(
                    hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs,
                    sim_data1['hist_sim_masked'], sim_data1['hist_sim_error_masked'], sim_data1['edges'][1], sim_data1['edges'][2],
                    hist_nxs2, hist_nxs_error2, y_edges_nxs2, z_edges_nxs2,
                    sim_data2['hist_sim_masked'], sim_data2['hist_sim_error_masked'], sim_data2['edges'][1], sim_data2['edges'][2],
                    args.q_min, args.q_max, y_plot_range, z_plot_range,
                    plot_path, split_view=getattr(args, 'split_view', False)
                )

        records.append(rec)
        save_summary_csv(records, args.output_dir, "fit_summary.csv")

        if np.isnan(loss):
            loss = 1e9

        param_str = ', '.join(f"{k}={v}" if isinstance(v, int) else f"{k}={format_fit_value(v)}" for k, v in display_point.items())
        background_str = ''
        if fit_background:
            background_str = (f" | background = {rec['background']:.3f}" if 'background' in rec else
                              f" | background = {rec['background_sample1']:.3f}, {rec['background_sample2']:.3f}")
        print_progress(f"{param_str} --> {args.loss_function} = {loss:.4f}{background_str}", eval_start_time)
        return loss

    if args.optimizer.lower() == 'differential-evolution':
        integrality = [is_integer(name) for name in param_names]
        # DE evaluates the initial population (popsize * N) and then popsize * N per generation.
        popsize = args.popsize
        population = max(5, popsize * len(param_names))  # scipy uses at least 5 population members
        if args.max_evals < population:
            print(f"WARNING: --max_evals {args.max_evals} is smaller than the initial DE population ({population}); "
                  f"the initial population alone will be evaluated.")
        de_maxiter = max(0, args.max_evals // population - 1)
        # DE stops when the spread (standard deviation) of the losses of its population is below
        # --fatol (absolute; tol=0 disables scipy's relative criterion). It has no parameter-space
        # criterion, so --xatol does not apply to DE.
        print(f"Differential Evolution: population {population}, at most {de_maxiter + 1} generations "
              f"({(de_maxiter + 1) * population} evaluations); stops when the population's loss spread < {args.fatol}")
        # A Generator, because scipy seeds the legacy RandomState from an int seed, which must be
        # below 2**32, while the default --seed is a random 63-bit number
        seed = getattr(args, 'seed', None)
        de_rng = np.random.default_rng(seed) if seed is not None else None
        opt_res = scipy.optimize.differential_evolution(
            objective_function,
            bounds,
            x0=x0,
            maxiter=de_maxiter,
            popsize=popsize,
            tol=0.0,
            atol=args.fatol,
            integrality=integrality,
            polish=False,
            seed=de_rng
        )
        best_x = np.asarray(opt_res.x, dtype=float)
    else:
        # Nelder-Mead / Powell work in scaled coordinates u = (x - x0) / scale, with scale the bound
        # range (or |x0|, or 1): all parameters are O(1), so --xatol is relative and tiny-valued
        # parameters (e.g. SLDs ~1e-6) are handled like any other.
        x0_arr = np.asarray(x0, dtype=float)
        scale = np.array([
            (high - low) if (low is not None and high is not None) else (abs(v) if v != 0 else 1.0)
            for v, (low, high) in zip(x0_arr, bounds)
        ])
        to_x = lambda u: x0_arr + scale * np.asarray(u)
        u_bounds = [((low - v) / sc_ if low is not None else None, (high - v) / sc_ if high is not None else None)
                    for v, sc_, (low, high) in zip(x0_arr, scale, bounds)]
        has_bounds = any(b is not None for pair in u_bounds for b in pair)
        opt_method = 'nelder-mead' if args.optimizer.lower() == 'nelder-mead' else 'powell'
        opt_options = {'maxiter': args.max_evals, 'maxfev': args.max_evals}
        if opt_method == 'nelder-mead':
            opt_options['xatol'] = args.xatol
            opt_options['fatol'] = args.fatol
            # initial simplex: 10% of the scale per parameter, at least one unit for integer parameters
            steps = [max(0.1, 1.0 / sc_) if is_integer(name) else 0.1 for name, sc_ in zip(param_names, scale)]
            simplex = [np.zeros(len(param_names))]
            for i, step in enumerate(steps):
                vertex = np.zeros(len(param_names))
                lo, hi = u_bounds[i]
                vertex[i] = step if (hi is None or step <= hi) else -step
                simplex.append(vertex)
            opt_options['initial_simplex'] = np.array(simplex)
        else:
            opt_options['xtol'] = args.xatol
            opt_options['ftol'] = args.fatol

        opt_res = scipy.optimize.minimize(
            lambda u: objective_function(to_x(u)), np.zeros(len(param_names)), method=opt_method,
            options=opt_options, bounds=u_bounds if has_bounds else None
        )
        best_x = to_x(opt_res.x)

    total_runtime = time.time() - start_total_time
    total_evals = max(1, eval_counter[0])
    avg_iter_runtime = total_runtime / total_evals

    fit_results_lines = [
        f"Optimizer Success: {opt_res.success}",
        f"Optimizer Message: {opt_res.message}",
        f"Best Loss ({args.loss_function}): {opt_res.fun:.4f}",
        "Optimal Parameters:"
    ]
    best_params = dict(zip(param_names, best_x))
    for k, v in best_params.items():
        val_str = str(int(np.round(v))) if is_integer(k) else format_fit_value(v)
        fit_results_lines.append(f"  {k} = {val_str}")
    if fit_background and records:
        best_rec = min(records, key=lambda r: (np.isnan(r[args.loss_function]), r[args.loss_function]))
        if 'background' in best_rec:
            fit_results_lines.append(f"  background = {best_rec['background']:.4f} [counts per pixel] "
                                     f"(fitted flat background of the best evaluation, #{best_rec['eval_index']})")
        else:
            fit_results_lines.append(f"  background = {best_rec['background_sample1']:.4f} (sample 1), "
                                     f"{best_rec['background_sample2']:.4f} (sample 2) [counts per pixel] "
                                     f"(fitted flat backgrounds of the best evaluation, #{best_rec['eval_index']})")

    fit_results_lines.extend([
        "\n--- Runtime Statistics ---",
        f"Total Runtime: {format_time(total_runtime)} ({total_runtime:.2f}s)",
        f"Average Iteration Runtime: {avg_iter_runtime:.2f}s",
        f"Total Evaluations Completed: {total_evals}"
    ])

    extra_summary_text = "\n".join(fit_results_lines)

    save_and_print_summary(records, args.output_dir, "fit_summary.csv", "Optimization", extra_summary_text=extra_summary_text, sort_key=args.loss_function)

    if args.gif:
        create_fit_evolution_gif(args.output_dir, is_joint=is_joint_fit)

def run_parameter_scan(
    args: Any,
    particles: Any,
    particle_type: str,
    hist_nxs: np.ndarray,
    hist_nxs_error: np.ndarray,
    y_edges_nxs: np.ndarray,
    z_edges_nxs: np.ndarray,
    mask: np.ndarray,
    mcpl_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Execute a parameter scan (grid sweep) over specified variables.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    particles : Any
        Preconditioned simulation particles.
    particle_type : str
        Particle type identifier.
    hist_nxs : np.ndarray
        NeXus data histogram.
    hist_nxs_error : np.ndarray
        NeXus data errors.
    y_edges_nxs : np.ndarray
        NeXus y edges.
    z_edges_nxs : np.ndarray
        NeXus z edges.
    mask : np.ndarray
        Mask array.
    mcpl_metadata : dict, optional
        MCPL source metadata (from `get_particles`) to embed in saved scipp results.
    """
    scanned_params = parse_scan_arguments(args.scan)
    keys = list(scanned_params.keys())
    value_lists = [scanned_params[k] for k in keys]

    grid = []
    for combo in itertools.product(*value_lists):
        grid.append(dict(zip(keys, combo)))

    total_evals = len(grid)
    print(f"Starting parameter scan with {total_evals} configurations...")

    records = []
    start_total_time = time.time()

    for idx, grid_point in enumerate(grid):
        iter_start_time = time.time()
        current_count = idx + 1
        print(f"\n[{current_count}/{total_evals}] Running simulation with: {grid_point}")
        metrics, record, _ = run_simulation_evaluation(
            grid_point, args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask,
            label_prefix="sim", save_simulation_output=True, mcpl_metadata=mcpl_metadata
        )

        iter_duration = time.time() - iter_start_time
        total_elapsed = time.time() - start_total_time
        avg_iter_time = total_elapsed / current_count
        remaining = total_evals - current_count
        eta = avg_iter_time * max(0, remaining)

        background_str = f", background={metrics['background']:.3f}" if getattr(args, 'fit_background', False) else ''
        print(f"Fit results: poisson_deviance={metrics['poisson_deviance']:.4f}, reduced_chi2={metrics['reduced_chi2']:.4f}, log_residual={metrics['log_residual']:.4e}{background_str} | Iter: {iter_duration:.2f}s | Avg: {avg_iter_time:.2f}s | ETA: {format_time(eta)}")
        records.append(record)
        save_summary_csv(records, args.output_dir, "scan_summary.csv")

    total_runtime = time.time() - start_total_time
    avg_iter_runtime = total_runtime / max(1, total_evals)

    runtime_summary_lines = [
        "--- Runtime Statistics ---",
        f"Total Runtime: {format_time(total_runtime)} ({total_runtime:.2f}s)",
        f"Average Iteration Runtime: {avg_iter_runtime:.2f}s",
        f"Total Configurations Scanned: {total_evals}"
    ]
    extra_summary_text = "\n".join(runtime_summary_lines)

    save_and_print_summary(records, args.output_dir, "scan_summary.csv", "Scan", extra_summary_text=extra_summary_text, sort_key=args.loss_function)

def _set_simulated_angle_range(args: Any, particles: np.ndarray, particle_type: str) -> None:
    """
    --simulate_mask_angle_range, once the particles are known: the per-neutron outgoing-angle windows of the
    neutrons hitting the sample (neutron_angle_windows) are calculated and their size and union printed. With
    --simulate_mask_angle_range_factor auto they are simulated (args.angle_range is set to their union, for
    information); with a numeric factor its range (set by prepare_experimental_data) is kept, and a warning is printed
    where it does not contain the union, i.e. where rays reaching the unmasked region are not simulated.
    """
    instrument = build_instrument(args, particle_type)  # the outgoing directions may not be chosen yet (--sampling)
    region = getattr(args, 'mask_angle_range', None) or args.angle_range
    factor = getattr(args, 'simulate_mask_angle_range_factor', 'auto')
    union, (size_h, size_v) = angle_window_extent(args, particles, instrument, region)
    print(f"Outgoing-angle window of each neutron (the directions that can reach the mask angle range, "
          f"+-{angle_window_margin(instrument):.4f} for the resolution): {size_h:.4f} x {size_v:.4f} deg; "
          f"their union: horiz=[{union[0]:.4f}, {union[1]:.4f}], vert=[{union[2]:.4f}, {union[3]:.4f}] "
          f"({union[1] - union[0]:.4f} x {union[3] - union[2]:.4f} deg)")
    if factor == 'auto':
        args.angle_range = union
        return
    names = ('horizontal minimum', 'horizontal maximum', 'vertical minimum', 'vertical maximum')
    short = [f"{name} {used:.4f} (union {needed:.4f})" for i, (name, used, needed) in enumerate(zip(names, args.angle_range, union))
             if (used > needed + 1e-9 if i % 2 == 0 else used < needed - 1e-9)]
    if short:
        print(f"WARNING: the simulated angle range of --simulate_mask_angle_range_factor {factor} does not contain the "
              f"union of the per-neutron windows: {'; '.join(short)} [deg]. Rays reaching the edge of the unmasked "
              f"region are missing ('auto' simulates the window of each neutron).")


def main() -> None:
    """
    Main entry point for fit.py. Parses arguments and delegates to run_automated_fit or run_parameter_scan.
    """
    parser = create_fit_parser()
    args = parse_run_args(parser)
    os.makedirs(args.output_dir, exist_ok=True)

    validate_fit_args(args, parser)

    hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw = prepare_experimental_data(args)

    if args.mask_view:
        plot_path = os.path.join(args.output_dir, "masked_view.png")
        y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges_nxs[0], y_edges_nxs[-1]]
        z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges_nxs[0], z_edges_nxs[-1]]
        # Determine default intensity_min if not provided
        if args.intensity_min is not None:
            intensity_min = float(args.intensity_min)
        else:
            intensity_min = 1.0 if args.experiment_time else 1e-9

        save_view_masks_plot(
            hist_nxs_raw, hist_nxs_error_raw,
            hist_nxs, hist_nxs_error,
            y_edges_nxs, z_edges_nxs,
            args.q_min, args.q_max, y_plot_range, z_plot_range,
            plot_path, intensity_min
        )
        return

    particles, particle_type, mcpl_metadata = load_and_precondition_particles(args)
    if getattr(args, 'simulate_mask_angle_range', False):
        _set_simulated_angle_range(args, particles, particle_type)
    set_outgoing_directions_from_sampling(args, particles, particle_type)  # once, for the final angle range

    if args.fit or args.fit2 or args.fit_common:
        run_automated_fit(args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, mcpl_metadata)
    else:
        run_parameter_scan(args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, mcpl_metadata)

if __name__ == '__main__':
    main()
