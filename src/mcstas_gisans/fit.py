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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .run_cli import parse_args as parse_run_args
from .input_output import get_particles
from .preconditioning import precondition
from .parameters import pack_parameters, set_outgoing_directions_from_sampling, build_instrument, angle_window_extent, angle_window_margin
from .run import process_particles, process_particles_parallelly
from .hardware import get_available_cores
from .nexus_reader import read_nexus_data, warn_if_duration_mismatch
from .experiment_time import upscale_simple
from .masking import get_mask, apply_mask, save_view_masks_plot, exit_angle_band_pixels

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
from . import fit_optimize, fit_restart
from .fit_continue import known_population_objective, load_de_continuation, replay_de_population  # noqa: F401 (re-exported)

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
MC_DISCOUNT_WARNING_FRACTION = 0.1  # warn if the MC variance lowers the fitted loss by more than 10%
MC_DIAGNOSTICS = ('mc_to_poisson_variance', 'mc_dominated_fraction',
                  'poisson_deviance_without_mc', 'reduced_chi2_without_mc',
                  'mc_discount_poisson_deviance', 'mc_discount_reduced_chi2')


def poisson_deviance_with_mc(counts: np.ndarray, expected: np.ndarray, mc_variance: np.ndarray) -> np.ndarray:
    """
    Per-pixel deviance 2 [ln P(N | N) - ln P(N | m, sigma^2)] of measured counts N against a
    simulated expectation m with Monte Carlo variance sigma^2.

    The finite simulation statistics are modelled by a gamma-distributed expectation (mean m,
    variance sigma^2), which makes N negative-binomially distributed with mean m and variance
    m + sigma^2 (effective likelihood for weighted Monte Carlo, cf. Arguelles, Schneider & Yuan,
    JHEP 06 (2019) 030). Limits: sigma -> 0 gives the Poisson deviance 2 [m - N + N ln(N/m)]
    (unbiased also at low counts); at high counts it approaches (N - m)^2 / (m + sigma^2).
    P(N | N) is the saturated Poisson term, so a perfect model gives ~1 per pixel. Without Monte
    Carlo variance this is the Poisson deviance of generalised linear models, also known as the
    Cash statistic (Cash, ApJ 228 (1979) 939) or the Baker-Cousins likelihood chi-square (Baker &
    Cousins, NIM 221 (1984) 437). See the fitting guide of the documentation.
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
        - ``mc_dominated_fraction``: fraction of the pixels where sigma_MC^2 > m.
        - ``poisson_deviance_without_mc``, ``reduced_chi2_without_mc``: the same losses with
          sigma_MC = 0.
        - ``mc_discount_poisson_deviance``, ``mc_discount_reduced_chi2``: 1 - loss / loss without
          MC, i.e. how much the Monte Carlo variance lowers the loss. A large value means that the
          comparison is limited by the simulation statistics, not by the counting statistics.
    """
    keep = np.isfinite(hist_nxs) & np.isfinite(hist_sim)
    n_valid = int(np.sum(keep))
    if n_valid == 0:
        return {key: np.nan for key in LOSS_FUNCTIONS + MC_DIAGNOSTICS}

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

    # the same losses without the Monte Carlo variance: how much does it lower them?
    poisson_deviance_without_mc = float(np.sum(poisson_deviance_with_mc(counts, expected, np.zeros_like(expected))) / n_valid)
    variance = np.where(expected > 0, expected, 1.0)
    reduced_chi2_without_mc = float(np.sum((counts - expected) ** 2 / variance) / n_valid)
    discount = lambda loss, loss_without_mc: 1.0 - loss / loss_without_mc if loss_without_mc > 0 else 0.0

    return {
        'poisson_deviance': poisson_deviance,
        'reduced_chi2': reduced_chi2,
        'log_residual': log_residual,
        'mc_to_poisson_variance': mc_to_poisson_variance,
        'mc_dominated_fraction': float(np.mean(mc_variance > expected)),
        'poisson_deviance_without_mc': poisson_deviance_without_mc,
        'reduced_chi2_without_mc': reduced_chi2_without_mc,
        'mc_discount_poisson_deviance': discount(poisson_deviance, poisson_deviance_without_mc),
        'mc_discount_reduced_chi2': discount(reduced_chi2, reduced_chi2_without_mc),
    }

def fit_flat_background(
    hist_nxs: np.ndarray,
    hist_signal: np.ndarray,
    hist_signal_mc_error: np.ndarray,
    loss_function: str,
    pixels_per_bin: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Flat background that minimises a loss between measured and simulated counts (``--fit_background``).

    The background b is searched by a bounded 1D minimisation between 0 and the mean measured
    counts per pixel of the used pixels (a flat level above the mean cannot fit), so it costs no
    extra simulation. Minimising over b for each parameter set gives the same optimum as fitting b
    as a parameter of the fit.

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
    pixels_per_bin : np.ndarray, optional
        For binned (summed) counts, e.g. the Qy profile of qz_band_profile: the number of pixels
        summed in each bin, which then gets pixels_per_bin * b of background. Default: one pixel each.

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
    if pixels_per_bin is None:
        loss = lambda b: calculate_fitness(counts, signal + b, error)[loss_function]
        b_max = max(float(np.mean(counts)), 1e-6)
    else:
        pixels = np.asarray(pixels_per_bin, dtype=float)[keep]
        loss = lambda b: calculate_fitness(counts, signal + b * pixels, error)[loss_function]
        b_max = max(float(np.sum(counts) / np.sum(pixels)), 1e-6)
    res = minimize_scalar(loss, bounds=(0.0, b_max), method='bounded', options={'xatol': 1e-3 * b_max})
    best_b, best_loss = float(res.x), float(res.fun)
    loss_at_zero = loss(0.0)  # the bounded method does not evaluate the end points themselves
    if loss_at_zero <= best_loss:
        best_b, best_loss = 0.0, loss_at_zero
    return best_b, best_loss


def has_qz_band(args: Any) -> bool:
    """Whether the band of the 1D Qy profile is given (a Qz band --q_min < --q_max, or --band_exit_angle), i.e. whether
    the 1D losses are defined."""
    return args.q_max > args.q_min or getattr(args, 'band_exit_angle', None) is not None


def band_description(args: Any) -> str:
    """The band of the 1D Qy profile, for messages."""
    if getattr(args, 'band_exit_angle', None) is not None:
        low, high = args.band_exit_angle
        return f"the exit-angle band [{low:g}, {high:g}] deg around the specular (--band_exit_angle)"
    return f"the Qz band [{args.q_min}, {args.q_max}] 1/nm"


def loss_keys(metrics: Dict[str, float]) -> List[str]:
    """The losses reported per evaluation: LOSS_FUNCTIONS of the detector image, then those of the Qy
    profile of the Qz band (suffix _1d) if it is given; with several wavelength slices (TOF) the losses are summed over
    the slices and those of every slice follow (suffix _slice<n>, combine_slice_metrics)."""
    keys = list(LOSS_FUNCTIONS) + [f"{key}_1d" for key in LOSS_FUNCTIONS if f"{key}_1d" in metrics]
    n_slices = sum(1 for key in metrics if key.startswith('background_slice'))
    return keys + [f"{key}_slice{n}" for n in range(1, n_slices + 1) for key in keys if f"{key}_slice{n}" in metrics]


def objective_loss_key(args: Any) -> str:
    """The metric that is minimised and by which the summaries are sorted: --loss_function of the
    detector image (2D) or, with --fit_objective 1d, of the Qy profile of the Qz band (key suffix _1d)."""
    return args.loss_function + ('_1d' if getattr(args, 'fit_objective', '2d') == '1d' else '')


def qz_band_indices(q_min: float, q_max: float, z_edges: np.ndarray) -> Tuple[int, int]:
    """
    First and last Qz bin (inclusive) of the band [q_min, q_max]: from the bin containing q_min to the
    bin containing q_max, clipped to the detector and ordered, as in the 1D slice of the comparison
    plots (np.digitize - 1 in save_comparison_plot, then plotting_utils.extract_range_to_1d).
    """
    last = len(z_edges) - 2
    i0, i1 = (min(max(0, int(np.digitize(q, z_edges)) - 1), last) for q in (q_min, q_max))
    return min(i0, i1), max(i0, i1)


def qz_band_profile(
    hist_nxs: np.ndarray,
    hist_sim: np.ndarray,
    hist_sim_mc_error: np.ndarray,
    z_edges: np.ndarray,
    q_min: float,
    q_max: float,
    band: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Qy profile of the Qz band [q_min, q_max] (the 1D slice of the comparison plots): the measured and
    the simulated counts summed over the pixels of each Qy bin (axis 0) within the band (axis 1,
    qz_band_indices) that are used in the 2D comparison (both finite, i.e. unmasked). With band (a
    boolean image of the shape of hist_nxs, e.g. the pixels of --band_exit_angle), the pixels of band
    are summed instead of the Qz band (q_min and q_max are not used).

    A sum of Poisson counts is Poisson distributed, so calculate_fitness applies to the profile as
    it is; the Monte Carlo variances of the summed pixels add.

    Parameters
    ----------
    hist_nxs : np.ndarray
        Measured counts (NaN outside the mask), shape (Qy bins, Qz bins).
    hist_sim : np.ndarray
        Simulated expected counts (including the background of each pixel, if any).
    hist_sim_mc_error : np.ndarray
        Monte Carlo uncertainty of hist_sim.
    z_edges : np.ndarray
        Qz bin edges (axis 1).
    q_min, q_max : float
        The Qz band [1/nm].
    band : np.ndarray, optional
        The pixels of the band (True), instead of the Qz band.

    Returns
    -------
    tuple
        (counts, expected, mc_error, n_pixels) per Qy bin; counts is NaN for the Qy bins without a
        used pixel in the band (so that calculate_fitness leaves them out).
    """
    if hist_nxs.shape[1] != len(z_edges) - 1:
        raise ValueError(f"The Qz edges ({len(z_edges)}) do not match axis 1 of the detector image {hist_nxs.shape}.")
    if band is None:
        i0, i1 = qz_band_indices(q_min, q_max, z_edges)
        columns = slice(i0, i1 + 1)
        nxs, sim, mc_error = hist_nxs[:, columns], hist_sim[:, columns], hist_sim_mc_error[:, columns]
        used = np.isfinite(nxs) & np.isfinite(sim)
    else:
        if band.shape != hist_nxs.shape:
            raise ValueError(f"The band ({band.shape}) does not match the detector image {hist_nxs.shape}.")
        nxs, sim, mc_error = hist_nxs, hist_sim, hist_sim_mc_error
        used = np.isfinite(nxs) & np.isfinite(sim) & band
    n_pixels = np.sum(used, axis=1)
    counts = np.where(n_pixels > 0, np.sum(np.where(used, nxs, 0.0), axis=1), np.nan)
    expected = np.sum(np.where(used, sim, 0.0), axis=1)
    mc_error_1d = np.sqrt(np.sum(np.where(used, mc_error ** 2, 0.0), axis=1))
    return counts, expected, mc_error_1d, n_pixels


def profile_plot_data(counts: np.ndarray, expected: np.ndarray, mc_error: np.ndarray, args: Any) -> Dict[str, Any]:
    """The 1D panel of save_comparison_plot from the profiles of qz_band_profile (--band_exit_angle): the measured
    counts with their Poisson error and the expected counts with the expected spread of a measurement (Poisson + Monte
    Carlo), in the Qy bins with used pixels."""
    used = np.isfinite(counts)
    return {
        'counts': counts,
        'counts_error': np.sqrt(np.where(used, counts, 0.0)),
        'expected': np.where(used, expected, np.nan),
        'expected_error': np.where(used, np.sqrt(np.maximum(expected, 0.0) + mc_error ** 2), 0.0),
        'title': f"Qy profile of {band_description(args)}",
    }


def format_parameter_label(grid_point: Dict[str, Any]) -> str:
    """Short, readable parameter list for plot titles: 4 significant digits (e.g. 'radius = 77.22, microgel_sld = 4.404e-06')."""
    def value(v):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            return str(v)
        return str(int(v)) if float(v).is_integer() and abs(v) < 1e6 else f"{float(v):.4g}"
    return ", ".join(f"{k} = {value(v)}" for k, v in grid_point.items())


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
    split_view: bool = False,
    label_nxs: str = "D22 measurement",
    profile: Optional[Dict[str, Any]] = None
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
        Description of the simulation (e.g. its parameters), shown wrapped as the title of the simulated map;
        the curves are labelled 'D22 measurement' and 'Simulation'.
    intensity_min : float, optional
        Minimum intensity for color scaling.
    split_view : bool, optional
        One split 2D map instead of two separate maps.
    label_nxs : str, optional
        Label for the measured data.
    profile : dict, optional
        The 1D panel when it is not a Qz range of the maps (--band_exit_angle, profile_plot_data): the Qy
        profiles 'counts', 'counts_error', 'expected', 'expected_error' (per Qy bin of y_edges_sim) and the
        'title'; q_min and q_max are then not used.
    """
    import textwrap
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
                      label_left=label_nxs, label_right="Simulation", title_text=textwrap.fill(label_sim, 90), ax=ax_map,
                      intensity_min=intensity_min, intensity_max=intensity_max,
                      y_range=y_plot_range, z_range=z_plot_range, match_horizontal_axes=True)
        map_axes = [ax_map]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 2D maps (no grid is added to these)
        log_plot_2d(hist_nxs, y_edges_nxs, z_edges_nxs, label_nxs, ax=axes[0, 0],
                    intensity_min=intensity_min, intensity_max=intensity_max,
                    y_range=y_plot_range, z_range=z_plot_range, output='none')

        log_plot_2d(hist_sim, y_edges_sim, z_edges_sim, textwrap.fill(label_sim, 70), ax=axes[0, 1],
                    intensity_min=intensity_min, intensity_max=intensity_max,
                    y_range=y_plot_range, z_range=z_plot_range, output='none')
        map_axes = [axes[0, 0], axes[0, 1]]

        gs = axes[1, 0].get_gridspec()
        axes[1, 0].remove()
        axes[1, 1].remove()
        ax_bottom = fig.add_subplot(gs[1:, :])
    # zooming one map zooms all of them, and the Qy axis of the 1D slice
    link_axes(map_axes, ax_bottom)

    if profile is not None:
        y_bins = 0.5 * (y_edges_sim[1:] + y_edges_sim[:-1])
        values_nxs, errors_nxs, y_bins_nxs = profile['counts'], profile['counts_error'], y_bins
        values_sim, errors_sim, y_bins_sim = profile['expected'], profile['expected_error'], y_bins
        title, z_limits = profile['title'], []
    else:
        qz_min_index = np.digitize(q_min, z_edges_sim) - 1
        qz_max_index = np.digitize(q_max, z_edges_sim) - 1

        # For 1D extraction, replace NaN with 0 so np.sum works properly
        hist_nxs_1d = np.nan_to_num(hist_nxs, nan=0.0)
        hist_sim_1d = np.nan_to_num(hist_sim, nan=0.0)

        values_nxs, errors_nxs, y_bins_nxs, z_limits = extract_range_to_1d(
            hist_nxs_1d, hist_nxs_error, y_edges_nxs, z_edges_nxs, [qz_min_index, qz_max_index]
        )
        values_sim, errors_sim, y_bins_sim, _ = extract_range_to_1d(
            hist_sim_1d, hist_sim_error, y_edges_sim, z_edges_sim, [qz_min_index, qz_max_index]
        )
        title = f"Qz=[{z_limits[0]:.4f} 1/nm, {z_limits[1]:.4f} 1/nm]"
    plot_q_1d(values_nxs, errors_nxs, y_bins_nxs, 'Qy [1/nm]', color='blue',
              title_text='', label=label_nxs, ax=ax_bottom, limits=y_plot_range, output='none')
    plot_q_1d(values_sim, errors_sim, y_bins_sim, 'Qy [1/nm]', color='green',
              label='Simulation', ax=ax_bottom, limits=y_plot_range, output='none')

    for ax in map_axes:
        for z_limit in z_limits:  # the summed Qz range
            ax.axhline(z_limit, color='magenta', linestyle='--')

    # Format 1D overlay plot (grid only on the major ticks of this plot)
    ax_bottom.set_title(title)
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
    from .plotting_utils import joint_comparison_figure

    samples = [
        {'measured': (hist_nxs1, hist_nxs_error1, y_edges_nxs1, z_edges_nxs1),
         'simulated': (hist_sim_masked1, hist_sim_error_masked1, edges_sim1_0, edges_sim1_1),
         'title': None, 'map_labels': ("Sample 1 Measurement", label_sim1),
         'legend_labels': ("Sample 1 Measurement", label_sim1)},
        {'measured': (hist_nxs2, hist_nxs_error2, y_edges_nxs2, z_edges_nxs2),
         'simulated': (hist_sim_masked2, hist_sim_error_masked2, edges_sim2_0, edges_sim2_1),
         'title': None, 'map_labels': ("Sample 2 Measurement", label_sim2),
         'legend_labels': ("Sample 2 Measurement", label_sim2)},
    ]
    fig = joint_comparison_figure(samples, q_min, q_max, y_plot_range, z_plot_range, split_view=split_view)
    fig.savefig(savename, dpi=300)
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
        if getattr(args, 'fit_objective', '2d') == '1d' and not has_qz_band(args):
            parser.error("--fit_objective 1d compares the Qy profiles of the Qz band [--q_min, --q_max]: "
                         f"give the band with --q_min < --q_max [1/nm] (got --q_min {args.q_min}, --q_max {args.q_max}) "
                         "or with --band_exit_angle.")
    for option in ('mask_exit_angle_band', 'band_exit_angle'):
        band = getattr(args, option, None)
        if band is not None and not band[0] < band[1]:
            parser.error(f"--{option} {band[0]:g} {band[1]:g}: expected LOW < HIGH [deg].")
    if getattr(args, 'band_exit_angle', None) is not None:
        if args.q_max > args.q_min:
            parser.error("--band_exit_angle defines the band of the 1D profile instead of --q_min/--q_max: give one of them.")
        if args.nxs2 or args.fit2 or args.fit_common or args.sample_arguments2:
            parser.error("--band_exit_angle is not implemented for joint (two-sample) fits; use --q_min/--q_max.")
    if not args.nxs:
        parser.error("the following arguments are required: --nxs")
    is_tof = getattr(args, 'instrument_params', {}).get('tof_instrument', False)
    wavelength_slices = getattr(args, 'wavelength_slice', None)
    if wavelength_slices:
        if not is_tof:
            parser.error("--wavelength_slice is only used for TOF instruments (the wavelength of a monochromatic instrument is --wavelength_selected).")
        for lambda_min, lambda_max in wavelength_slices:
            if not 0 < lambda_min < lambda_max:
                parser.error(f"--wavelength_slice {lambda_min:g} {lambda_max:g}: expected 0 < MIN < MAX [Angstrom].")
    if getattr(args, 'intensity_factor_spectrum', None) and not wavelength_slices:
        parser.error("--intensity_factor_spectrum calibrates the simulated TOF histograms per wavelength: it requires a TOF "
                     "instrument and --wavelength_slice (for monochromatic data, --intensity_factor is the calibration).")
    if is_tof:
        if not wavelength_slices:
            parser.error("TOF instruments are compared in wavelength slices: give --wavelength_slice MIN MAX [Angstrom] (one or more).")
        if args.nxs2 or args.fit2 or args.fit_common or args.sample_arguments2:
            parser.error("Joint (two-sample) fits are not implemented for TOF instruments yet.")
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

    if getattr(args, 'polish', None) and args.optimizer.lower() != 'differential-evolution':
        parser.error("--polish refines the best point of a Differential Evolution fit: it requires --optimizer differential-evolution "
                     "(Nelder-Mead and Powell are local optimisers themselves).")
    if getattr(args, 'de_continue', None) and args.optimizer.lower() != 'differential-evolution':
        parser.error("--de_continue requires --optimizer differential-evolution.")

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
    if getattr(args, 'wavelength_slice', None):
        return prepare_tof_experimental_data(args)

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
    if getattr(args, 'mask_exit_angle_band', None) is not None:
        band_pixels = _exit_angle_band_image(instrument, wavelength_val, args.mask_exit_angle_band, hist_nxs_raw.shape)
        print(f"--mask_exit_angle_band {args.mask_exit_angle_band[0]:g} {args.mask_exit_angle_band[1]:g}: "
              f"{int(np.sum(band_pixels))} pixels have their exit angle in the band at {wavelength_val:g} AA")
        mask &= band_pixels
    # the pixels of the 1D band of the exit angle (qz_band_profile)
    args.band_pixels = None if getattr(args, 'band_exit_angle', None) is None else \
        _exit_angle_band_image(instrument, wavelength_val, args.band_exit_angle, hist_nxs_raw.shape)

    hist_nxs = apply_mask(hist_nxs_raw, mask, np.nan)
    hist_nxs_error = apply_mask(hist_nxs_error_raw, mask, 0.0)
    n_unmasked = int(np.sum(np.isfinite(hist_nxs)))
    print(f"Unmasked detector pixels used for the comparison: {n_unmasked}")
    if n_unmasked == 0 and not args.mask_view:
        raise ValueError("The mask excludes every detector pixel; nothing is left to compare.")
    if has_qz_band(args):
        # the pixels of the 1D losses (qz_band_profile): the unmasked ones within the band
        _print_qz_band(args, hist_nxs, z_edges_nxs, band=args.band_pixels)

    if getattr(args, 'simulate_mask_angle_range', False):
        _set_mask_angle_range(args, instrument, mask)  # the Q bins are the detector pixels
    else:
        args.direction_keep_pixels = None

    return hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw


def _set_mask_angle_range(args: Any, instrument: Any, pixel_mask: np.ndarray) -> None:
    """
    --simulate_mask_angle_range: the range enclosing the unmasked (True) detector pixels of pixel_mask (rotated detector
    image [iy_BA, iz_BA]) as seen from the sample centre; with --simulate_mask_angle_range_factor auto every neutron gets
    its own window of directions that can reach it (neutron_angle_windows, in process_particles), with a numeric factor
    the range scaled by it is simulated.
    """
    args.mask_angle_range = list(instrument.get_masked_angle_range(pixel_mask))
    factor = getattr(args, 'simulate_mask_angle_range_factor', 'auto')
    if factor == 'auto':
        args.angle_window_region = list(args.mask_angle_range)
        args.angle_range = args.mask_angle_range  # replaced by the union of the windows (_set_simulated_angle_range)
    else:
        args.angle_window_region = None  # (args2 of a joint fit is a copy of args)
        args.angle_range = list(instrument.get_masked_angle_range(pixel_mask, factor=factor))
    fmt = lambda r: f"horiz=[{r[0]:.4f}, {r[1]:.4f}], vert=[{r[2]:.4f}, {r[3]:.4f}]"
    print(f"Mask angle range [deg] (the unmasked pixels seen from the sample centre): {fmt(args.mask_angle_range)}"
          + ("" if factor == 'auto' else f"; simulated with factor {factor:g}: {fmt(args.angle_range)}"))
    # the unmasked pixels in the raw detector frame of the ray tracing (run.directions_reaching_region): the
    # outgoing directions aimed only at masked pixels are not simulated (unless --keep_masked_directions)
    args.direction_keep_pixels = None if getattr(args, 'keep_masked_directions', False) else raw_pixel_mask(instrument, pixel_mask)


def raw_pixel_mask(instrument: Any, mask: np.ndarray) -> np.ndarray:
    """
    The mask of the compared image (the rotated detector image of run_simulation_evaluation, transposed there if its
    shape requires) in the raw NeXus pixel frame [ix, iy] of the ray tracing.
    """
    det = instrument.detector
    raw_index = np.arange(det.pixels_x_nexus * det.pixels_y_nexus).reshape(det.pixels_x_nexus, det.pixels_y_nexus)
    rotated = det.coords.rotate_detector_image(raw_index)
    if rotated.shape != mask.shape and rotated.T.shape == mask.shape:
        rotated = rotated.T
    if rotated.shape != mask.shape:
        raise ValueError(f"The mask {mask.shape} does not match the detector image {rotated.shape}.")
    keep = np.zeros(raw_index.size, dtype=bool)
    keep[rotated[np.asarray(mask, dtype=bool)]] = True
    return keep.reshape(raw_index.shape)


def _exit_angle_band_image(instrument: Any, wavelength: float, band: Sequence[float], shape: Tuple[int, int]) -> np.ndarray:
    """The pixels of an exit-angle band (masking.exit_angle_band_pixels) as the rotated detector image of the measurement."""
    image = instrument.detector.coords.rotate_detector_image(exit_angle_band_pixels(instrument, wavelength, band))
    if image.shape != tuple(shape):
        raise ValueError(f"The detector image of the exit-angle band {image.shape} does not match the measured image {shape}.")
    return image


def _print_qz_band(args: Any, hist_nxs: np.ndarray, z_edges_nxs: np.ndarray, what: str = "pixels",
                   band: Optional[np.ndarray] = None) -> None:
    """Print the unmasked bins of the band of the 1D losses (qz_band_profile: the Qz band, or the pixels of band); stop
    if --fit_objective 1d has none."""
    if band is None:
        i0, i1 = qz_band_indices(args.q_min, args.q_max, z_edges_nxs)
        in_band = np.isfinite(hist_nxs[:, i0:i1 + 1])
        name = f"Qz band of the 1D losses: [{z_edges_nxs[i0]:.4f}, {z_edges_nxs[i1 + 1]:.4f}] 1/nm"
        counts = np.nansum(hist_nxs[:, i0:i1 + 1])
    else:
        in_band = np.isfinite(hist_nxs) & band
        name = f"Band of the 1D losses ({band_description(args)})"
        counts = np.sum(hist_nxs[in_band])
    n_band = int(np.sum(in_band))
    print(f"{name}: {n_band} unmasked {what} in {int(np.sum(np.any(in_band, axis=1)))} Qy bins, {counts:.0f} counts")
    if n_band == 0 and not args.mask_view:
        if getattr(args, 'fit_objective', '2d') == '1d':
            raise ValueError(f"The band of the 1D profile ({band_description(args)}) has no unmasked pixel: nothing is "
                             "left to compare with --fit_objective 1d.")
        print(f"WARNING: the band of the 1D profile ({band_description(args)}) has no unmasked pixel; the 1D losses are "
              "not defined (NaN).")


def prepare_tof_experimental_data(args: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    TOF instruments (--wavelength_slice): load the measured TOF histograms of the --nxs file(s) (summed), build the
    wavelength slices (tof_slices.build_wavelength_slices) and their measured Q histograms with the masks, and store
    them in args.tof_slices (and the TOF bin edges [s] of the simulated histograms in args.tof_bin_edges).

    The mask of a slice keeps the Q bins that are unmasked by the --mask_* options and receive counts from at least one
    (pixel, TOF bin) of the slice (the coverage of the detector at the wavelengths of the slice). The --mask_* options
    apply to every slice; --mask_exit_angle_band selects the pixels of each slice at its centre wavelength (the
    pixel_selection of the slice), and --band_exit_angle the pixels of its 1D profile (profile_pixels, profile_hist_nxs,
    profile_unit). With --intensity_factor_spectrum, args.tof_intensity_factors holds the factor of every simulated
    (pixel, TOF bin).

    Returns
    -------
    tuple
        (hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw) of the first slice,
        as prepare_experimental_data.
    """
    from .instrument import Instrument
    from .instrument_defaults import get_instrument_parameters
    from .nexus_reader import get_nxs_format, read_sans2d_raw
    from .tof_slices import (build_wavelength_slices, intensity_factors_of_tof_bins, read_intensity_factor_spectrum,
                             tof_bin_wavelengths)

    instr_params = get_instrument_parameters(args)
    instrument = Instrument(instr_params, args.alpha, None, args.sample_orientation, args.wfm, args.no_gravity)
    detector_shape = (instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus)

    nxs_paths = args.nxs if isinstance(args.nxs, list) else [args.nxs]
    print(f"Loading experimental TOF data from {len(nxs_paths)} file(s):")
    counts, tof_edges = None, None
    for path in nxs_paths:
        nxs_format = get_nxs_format(path, getattr(args, 'nxs_format', 'auto'))
        if nxs_format != 'isis_sans2d':
            raise ValueError(f"{path}: --wavelength_slice compares measured TOF histograms (--nxs_format isis_sans2d), "
                             f"but the file is read as '{nxs_format}'.")
        data = read_sans2d_raw(path)
        if data['counts'].shape[:2] != detector_shape:
            raise ValueError(f"The detector of {path} has {data['counts'].shape[:2]} pixels, but the instrument detector "
                             f"has {detector_shape} (see --instrument_detector_pixels).")
        if tof_edges is None:
            tof_edges, counts = data['tof_edges'], data['counts'].astype(np.int64)
        elif len(data['tof_edges']) != len(tof_edges) or not np.allclose(data['tof_edges'], tof_edges):
            raise ValueError(f"The TOF bins of {path} differ from those of {nxs_paths[0]}: the files cannot be summed.")
        else:
            counts += data['counts']
        distance = data['rear_detector_distance']
        print(f"  {path}: '{data['title']}', {data['counts'].sum():.0f} counts on the detector, "
              f"{len(tof_edges) - 1} TOF bins {tof_edges[0] * 1e3:.4g}-{tof_edges[-1] * 1e3:.4g} ms"
              + (f", duration {data['duration']:.0f} s" if data['duration'] is not None else '')
              + (f", sample-detector distance {distance:.4f} m" if distance is not None else ''))
        if distance is not None and abs(distance - instrument.sample_detector_distance) > 0.002:
            print(f"WARNING: the sample-detector distance of {path} ({distance:.4f} m) differs from that of the "
                  f"instrument ({instrument.sample_detector_distance:.4f} m, --instrument_sample_detector_distance).")
    warn_if_duration_mismatch(nxs_paths, args.experiment_time, label=f"{len(nxs_paths)} --nxs file(s)")

    source_sample_distance = instrument.nominal_source_sample_distance
    mask_band = getattr(args, 'mask_exit_angle_band', None)
    slices, first_bin, last_bin = build_wavelength_slices(instrument, tof_edges, args.wavelength_slice, source_sample_distance,
                                                          exit_angle_band=mask_band)
    args.tof_bin_edges = tof_edges[first_bin:last_bin + 1]
    measured = counts.reshape(-1, counts.shape[-1])[:, first_bin:last_bin].astype(np.float64)
    print(f"Wavelength of the TOF bins: lambda = h t / (m (L1 + L2)), L1 = {source_sample_distance:g} m, L2 = sample-pixel "
          f"distance; TOF bins {first_bin}-{last_bin - 1} ({args.tof_bin_edges[0] * 1e3:.4g}-{args.tof_bin_edges[-1] * 1e3:.4g} ms) "
          f"are simulated")
    args.tof_intensity_factors = None
    if getattr(args, 'intensity_factor_spectrum', None):
        spectrum = read_intensity_factor_spectrum(args.intensity_factor_spectrum)
        args.tof_intensity_factors = intensity_factors_of_tof_bins(
            spectrum, tof_bin_wavelengths(instrument, args.tof_bin_edges, source_sample_distance), args.wavelength_slice)
        print(f"Intensity factor spectrum {args.intensity_factor_spectrum}: {len(spectrum[0])} points, "
              f"{spectrum[0][0]:g}-{spectrum[0][-1]:g} AA; the factor at the centre of the slices: "
              + ", ".join(f"{np.interp(0.5 * (a + b), *spectrum):.4g}" for a, b in args.wavelength_slice))
    profile_band = getattr(args, 'band_exit_angle', None)
    all_pixels = np.ones(instrument.detector.pixels_x_nexus * instrument.detector.pixels_y_nexus, dtype=bool)

    used_pixels = np.zeros(detector_shape, dtype=bool)
    for wavelength_slice in slices:
        centre = 0.5 * (wavelength_slice.lambda_min + wavelength_slice.lambda_max)
        if mask_band is not None:
            print(f"Wavelength slice {wavelength_slice.label}: {int(np.sum(wavelength_slice.pixel_selection))} pixels have "
                  f"their exit angle in --mask_exit_angle_band {mask_band[0]:g} {mask_band[1]:g} at {centre:g} AA")
        hist_raw = wavelength_slice.to_q(measured, pixels=all_pixels)  # every pixel, for the mask view
        hist_error_raw = np.sqrt(hist_raw)
        mask = get_mask(
            wavelength_slice.y_edges, wavelength_slice.z_edges,
            mask_qy_range=args.mask_qy_range,
            qy_min_cut=args.mask_qy_min_cut, qy_max_cut=args.mask_qy_max_cut,
            qz_min_cut=args.mask_qz_min_cut, qz_max_cut=args.mask_qz_max_cut,
            exclude_q_box=args.mask_exclude_q_box,
            include_q_box=args.mask_include_q_box,
            shape=hist_raw.shape
        ) & wavelength_slice.coverage
        wavelength_slice.hist_nxs_raw, wavelength_slice.hist_nxs_error_raw = hist_raw, hist_error_raw
        # the compared counts: those of the pixel selection (--mask_exit_angle_band) only, which can share a Q bin with
        # other pixels at other wavelengths of the slice
        hist = hist_raw if wavelength_slice.pixel_selection is None else wavelength_slice.to_q(measured)
        wavelength_slice.hist_nxs = apply_mask(hist, mask, np.nan)
        wavelength_slice.hist_nxs_error = apply_mask(np.sqrt(hist), mask, 0.0)
        wavelength_slice.mask = mask
        n_unmasked = int(np.sum(mask))
        print(f"Wavelength slice {wavelength_slice.label}: {wavelength_slice.n_tof_bins} TOF bins, {np.sum(hist_raw):.0f} counts in "
              f"its Q grid ({np.sum(wavelength_slice.coverage)} Q bins with counts of the detector, {n_unmasked} unmasked, "
              f"{np.nansum(wavelength_slice.hist_nxs):.0f} counts in them)")
        if n_unmasked == 0 and not args.mask_view:
            raise ValueError(f"The mask excludes every Q bin of the wavelength slice {wavelength_slice.label}; nothing is left to compare.")
        if profile_band is not None:
            # the 1D profile: the (pixel, TOF bin) of the pixels of the band (and of the pixel selection) in unmasked Q bins
            pixels = exit_angle_band_pixels(instrument, centre, profile_band).ravel()
            if wavelength_slice.pixel_selection is not None:
                pixels = pixels & wavelength_slice.pixel_selection
            wavelength_slice.profile_pixels = pixels
            wavelength_slice.profile_unit = wavelength_slice.background_unit_of(pixels)
            wavelength_slice.profile_hist_nxs = apply_mask(wavelength_slice.to_q(measured, pixels=pixels),
                                                           mask & (wavelength_slice.profile_unit > 0), np.nan)
            _print_qz_band(args, wavelength_slice.profile_hist_nxs, wavelength_slice.z_edges, what="Q bins",
                           band=np.ones(wavelength_slice.q_shape, dtype=bool))
        elif has_qz_band(args):
            _print_qz_band(args, wavelength_slice.hist_nxs, wavelength_slice.z_edges, what="Q bins")
        used_pixels |= wavelength_slice.used_pixels(mask)
    args.tof_slices = slices

    if getattr(args, 'simulate_mask_angle_range', False):
        # the detector pixels with a TOF bin of a slice in an unmasked Q bin
        _set_mask_angle_range(args, instrument, instrument.detector.coords.rotate_detector_image(used_pixels))
    else:
        args.direction_keep_pixels = None

    first = slices[0]
    return first.hist_nxs, first.hist_nxs_error, first.y_edges, first.z_edges, first.mask, first.hist_nxs_raw, first.hist_nxs_error_raw

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
    slices = getattr(args, 'tof_slices', None)
    if slices and len(particles):
        # TOF: only the neutrons that can be detected in a TOF bin of a wavelength slice
        from .tof_slices import particles_reaching_slices
        instrument = build_instrument(args, particle_type)
        reaching = particles_reaching_slices(particles, instrument, slices, instrument.nominal_source_sample_distance)
        print(f"{int(np.sum(reaching))} of the {len(particles)} preconditioned particles can be detected in the wavelength slice(s) "
              f"and are simulated.")
        particles = particles[reaching]
    if len(particles) == 0:
        raise ValueError("No incident particles are left to simulate (none hit the sample, or none can reach the wavelength slices). "
                         "Check --alpha, the sample size and --sample_orientation, or use --allow_sample_miss.")
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
    tof_slices = getattr(args, 'tof_slices', None)
    if tof_slices:
        params['tof_bin_edges'] = args.tof_bin_edges  # (pixel, TOF bin) histograms instead of events

    if args.no_parallel:
        result = process_particles(particles, params)
    else:
        process_number = args.parallel_processes if args.parallel_processes else max(1, get_available_cores() - 1)
        result = process_particles_parallelly(particles, params, process_number)

    if result.get('directions') and params.get('direction_keep_pixels') is not None and not getattr(args, '_skipped_directions_reported', False):
        args._skipped_directions_reported = True
        print(f"Directions aimed only at masked pixels, not simulated: {result['directions_skipped']} of {result['directions']} "
              f"({100 * result['directions_skipped'] / result['directions']:.1f}%; --keep_masked_directions simulates them)")

    if tof_slices:
        return evaluate_wavelength_slices(grid_point, args, result, label_prefix, save_simulation_output)

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
    def band_profile(sim):  # the Qy profile of the Qz band, from the same pixels (1D losses)
        return qz_band_profile(hist_nxs, sim, hist_sim_mc_error, z_edges_nxs, args.q_min, args.q_max,
                               band=getattr(args, 'band_pixels', None))

    if fit_background:
        # fitted for the minimised loss (--fit_objective): over the pixels, or over the profile,
        # whose bins get (number of summed pixels) x background
        if getattr(args, 'fit_objective', '2d') == '1d':
            counts_1d, signal_1d, mc_error_1d, n_pixels = band_profile(hist_sim)
            background, _ = fit_flat_background(counts_1d, signal_1d, mc_error_1d, args.loss_function, pixels_per_bin=n_pixels)
        else:
            background, _ = fit_flat_background(hist_nxs, hist_sim, hist_sim_mc_error, args.loss_function)
        hist_sim = hist_sim + background

    # hist_nxs is NaN outside the mask (prepare_experimental_data)
    metrics = calculate_fitness(hist_nxs, hist_sim, hist_sim_mc_error)
    if has_qz_band(args):
        metrics.update({f"{key}_1d": value for key, value in calculate_fitness(*band_profile(hist_sim)[:3]).items()})
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
    for key in loss_keys(metrics):
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

        profile = profile_plot_data(*band_profile(hist_sim)[:3], args) if getattr(args, 'band_pixels', None) is not None else None
        save_comparison_plot(
            hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs,
            hist_sim_masked, hist_sim_error_masked, edges[1], edges[2],
            args.q_min, args.q_max, y_plot_range, z_plot_range,
            plot_path, f"Simulation: {format_parameter_label(grid_point)}", intensity_min,
            split_view=getattr(args, 'split_view', False), profile=profile
        )

    sim_data = {
        'hist_sim_masked': hist_sim_masked,
        'hist_sim_error_masked': hist_sim_error_masked,
        'edges': edges
    }

    return metrics, record, sim_data


def combine_slice_metrics(slice_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    The metrics of an evaluation with several wavelength slices (TOF): the losses (and those without the Monte Carlo
    variance) are summed over the slices, mc_to_poisson_variance and mc_dominated_fraction are their maxima and the
    discounts are those of the summed losses. The losses and backgrounds of every slice are added with the suffix
    _slice<n> (n = 1, 2, ...); 'background' is that of the first slice. A single slice is returned as it is.
    """
    if len(slice_metrics) == 1:
        return dict(slice_metrics[0])
    combined = {}
    for suffix in ('', '_1d'):
        if f"poisson_deviance{suffix}" not in slice_metrics[0]:
            continue
        total = lambda key: float(np.sum([m[key + suffix] for m in slice_metrics]))
        largest = lambda key: float(np.max([m[key + suffix] for m in slice_metrics]))
        for key in LOSS_FUNCTIONS + ('poisson_deviance_without_mc', 'reduced_chi2_without_mc'):
            combined[key + suffix] = total(key)
        for key in ('mc_to_poisson_variance', 'mc_dominated_fraction'):
            combined[key + suffix] = largest(key)
        for loss in ('poisson_deviance', 'reduced_chi2'):
            without_mc = combined[f"{loss}_without_mc{suffix}"]
            combined[f"mc_discount_{loss}{suffix}"] = 1.0 - combined[loss + suffix] / without_mc if without_mc > 0 else 0.0
    combined['background'] = slice_metrics[0]['background']
    for index, metrics in enumerate(slice_metrics):
        for key in loss_keys(metrics):
            combined[f"{key}_slice{index + 1}"] = metrics[key]
        combined[f"background_slice{index + 1}"] = metrics['background']
    return combined


def background_entries(metrics: Dict[str, float]) -> Dict[str, float]:
    """The flat background(s) of an evaluation for its record: 'background', or with several wavelength slices (TOF)
    'background_slice<n>' of every slice."""
    slice_keys = [key for key in metrics if key.startswith('background_slice')]
    return {key: metrics[key] for key in slice_keys} if slice_keys else {'background': metrics['background']}


def evaluate_wavelength_slices(
    grid_point: Dict[str, Any],
    args: Any,
    result: Dict[str, Any],
    label_prefix: str = "sim",
    save_simulation_output: bool = False
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    """
    The comparison of a simulation of a TOF instrument with the measurement in the wavelength slices args.tof_slices
    (prepare_tof_experimental_data): the simulated (pixel, TOF bin) histograms (result['tofHist'] and the squared
    weights) are converted to the Q histogram of every slice exactly as the measurement, scaled to expected counts over
    --experiment_time, and compared as in run_simulation_evaluation. The flat background b [counts per pixel over the
    slice] adds b * background_unit (the Q histogram of one count per pixel) to the simulated counts; with
    --fit_background it is fitted for every slice. The losses of the slices are summed (combine_slice_metrics). With
    --intensity_factor_spectrum the simulated (pixel, TOF bin) histograms are multiplied by args.tof_intensity_factors
    (the squared weights by their squares) first. With --band_exit_angle the 1D profile of a slice is that of the
    (pixel, TOF bin) of its profile_pixels (prepare_tof_experimental_data). The measured, simulated and background
    counts in the unmasked Q bins of every slice are printed.

    With save_simulation_output, the simulated Q histogram of every slice (rate [1/s] and its Monte Carlo uncertainty,
    without background, with the intensity factor spectrum) is saved as .npz (hist, error, yEdges, zEdges; readable by mg_plot); with --png a comparison
    plot is saved for every slice (the files of the slices after the first have the suffix _slice<n>).

    Returns
    -------
    tuple
        (metrics, record, sim_data) as run_simulation_evaluation; sim_data of the first slice.
    """
    slices = args.tof_slices
    fit_background = getattr(args, 'fit_background', False)
    fixed_background = 0.0 if (fit_background or args.background is None) else args.background
    param_str = '_'.join(f"{k}_{v}" for k, v in grid_point.items())

    tof_hist, tof_hist_weights_squared = result['tofHist'], result['tofHistWeightsSquared']
    factors = getattr(args, 'tof_intensity_factors', None)
    if factors is not None:  # --intensity_factor_spectrum
        tof_hist = tof_hist * factors
        tof_hist_weights_squared = tof_hist_weights_squared * factors ** 2

    slice_metrics = []
    sim_data = None
    for index, wavelength_slice in enumerate(slices):
        hist_nxs, z_edges = wavelength_slice.hist_nxs, wavelength_slice.z_edges
        rate = wavelength_slice.to_q(tof_hist)
        rate_error = np.sqrt(wavelength_slice.to_q(tof_hist_weights_squared))
        # expected counts over the measurement time (deterministic) and their Monte Carlo uncertainty
        hist_signal = rate * args.experiment_time
        hist_sim_mc_error = rate_error * args.experiment_time
        unit = wavelength_slice.background_unit

        # the 1D losses: the Qy profile of the Qz band, or of the (pixel, TOF bin) of the exit-angle band
        if wavelength_slice.profile_pixels is not None:
            pixels = wavelength_slice.profile_pixels
            profile_nxs, profile_unit = wavelength_slice.profile_hist_nxs, wavelength_slice.profile_unit
            profile_signal = wavelength_slice.to_q(tof_hist, pixels=pixels) * args.experiment_time
            profile_mc_error = np.sqrt(wavelength_slice.to_q(tof_hist_weights_squared, pixels=pixels)) * args.experiment_time
            whole = np.ones(wavelength_slice.q_shape, dtype=bool)
        else:
            profile_nxs, profile_unit, profile_signal, profile_mc_error = hist_nxs, unit, hist_signal, hist_sim_mc_error
            whole = None

        def band_profile(background, values=None, mc_error=None):
            """(counts, expected incl. the background, mc_error, n) per Qy bin; of values instead of the signal if given."""
            values = profile_signal + background * profile_unit if values is None else values
            return qz_band_profile(profile_nxs, values, profile_mc_error if mc_error is None else mc_error, z_edges,
                                   args.q_min, args.q_max, band=whole)

        background = fixed_background
        if fit_background:
            if getattr(args, 'fit_objective', '2d') == '1d':
                counts_1d, signal_1d, mc_error_1d, _ = band_profile(0.0)
                unit_1d = band_profile(0.0, profile_unit, np.zeros_like(profile_unit))[1]  # the background of the summed Q bins
                background, _ = fit_flat_background(counts_1d, signal_1d, mc_error_1d, args.loss_function, pixels_per_bin=unit_1d)
            else:
                background, _ = fit_flat_background(hist_nxs, hist_signal, hist_sim_mc_error, args.loss_function, pixels_per_bin=unit)
        hist_sim = hist_signal + background * unit

        metrics = calculate_fitness(hist_nxs, hist_sim, hist_sim_mc_error)
        used = np.isfinite(hist_nxs)
        counts_line = (f"measured {np.sum(hist_nxs[used]):.0f}, simulated {np.sum(hist_signal[used]):.1f} + background "
                       f"{background * np.sum(unit[used]):.1f} counts in the {int(np.sum(used))} unmasked Q bins")
        if has_qz_band(args):
            profile_1d = band_profile(background)
            metrics.update({f"{key}_1d": value for key, value in calculate_fitness(*profile_1d[:3]).items()})
            used_1d = np.isfinite(profile_1d[0])
            counts_line += (f"; 1D profile: measured {np.sum(profile_1d[0][used_1d]):.0f}, simulated incl. background "
                            f"{np.sum(profile_1d[1][used_1d]):.1f} in {int(np.sum(used_1d))} Qy bins")
        print(f"  Wavelength slice {wavelength_slice.label}: {counts_line}")
        metrics['background'] = background
        slice_metrics.append(metrics)

        # for display: expected spread of a measurement = Poisson + Monte Carlo
        hist_sim_error = np.sqrt(np.maximum(hist_sim, 0.0) + hist_sim_mc_error**2)
        hist_sim_masked = apply_mask(hist_sim, wavelength_slice.mask, np.nan)
        hist_sim_error_masked = apply_mask(hist_sim_error, wavelength_slice.mask, 0.0)
        if sim_data is None:
            sim_data = {
                'hist_sim_masked': hist_sim_masked,
                'hist_sim_error_masked': hist_sim_error_masked,
                'edges': [None, wavelength_slice.y_edges, z_edges]
            }

        suffix = '' if index == 0 else f"_slice{index + 1}"
        if save_simulation_output:
            npz_path = os.path.join(args.output_dir, f"{label_prefix}_{param_str}{suffix}.npz")
            np.savez(npz_path, hist=rate, error=rate_error, yEdges=wavelength_slice.y_edges, zEdges=z_edges,
                     wavelength_slice=[wavelength_slice.lambda_min, wavelength_slice.lambda_max],
                     background_unit=unit)
            print(f"Created {npz_path} (simulated Q histogram of the wavelength slice {wavelength_slice.label} [1/s])")
        if args.png:
            plot_path = os.path.join(args.output_dir, f"{label_prefix}_{param_str}{suffix}.png")
            y_plot_range = args.y_plot_range if args.y_plot_range else [wavelength_slice.y_edges[0], wavelength_slice.y_edges[-1]]
            z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges[0], z_edges[-1]]
            intensity_min = float(args.intensity_min) if args.intensity_min is not None else 1.0
            profile = profile_plot_data(*band_profile(background)[:3], args) if whole is not None else None
            save_comparison_plot(
                hist_nxs, wavelength_slice.hist_nxs_error, wavelength_slice.y_edges, z_edges,
                hist_sim_masked, hist_sim_error_masked, wavelength_slice.y_edges, z_edges,
                args.q_min, args.q_max, y_plot_range, z_plot_range,
                plot_path, f"Sim ({param_str}), {wavelength_slice.label}", intensity_min,
                split_view=getattr(args, 'split_view', False), label_nxs=f"Measurement, {wavelength_slice.label}",
                profile=profile
            )

    metrics = combine_slice_metrics(slice_metrics)
    _warn_if_mc_uncertainty_large(metrics, args)

    record = copy.deepcopy(grid_point)
    if fit_background:
        record.update(background_entries(metrics))
    for key in loss_keys(metrics):
        record[key] = metrics[key]
    return metrics, record, sim_data


_MC_WARNING_ISSUED = [False]


def _warn_if_mc_uncertainty_large(metrics: Dict[str, float], args: Any) -> None:
    """
    Warn once per run if the simulation's Monte Carlo variance is not small compared to the
    counting variance: in more than 5% of the pixels (95th percentile of sigma_MC^2 / m above
    MC_TO_POISSON_WARNING_RATIO), or in few pixels where it lowers the loss of the fit
    (poisson_deviance, reduced_chi2) by more than MC_DISCOUNT_WARNING_FRACTION, which hides misfit
    there (e.g. where the pattern is steep compared to the outgoing-direction grid). With
    --fit_objective 1d, the loss of the fit is that of the Qy profile of the Qz band (suffix _1d).
    """
    if _MC_WARNING_ISSUED[0]:
        return
    ratio = metrics.get('mc_to_poisson_variance', np.nan)
    loss_function = getattr(args, 'loss_function', 'poisson_deviance')
    suffix = '_1d' if getattr(args, 'fit_objective', '2d') == '1d' else ''
    discount = metrics.get(f'mc_discount_{loss_function}{suffix}', np.nan)
    many_pixels = np.isfinite(ratio) and ratio > MC_TO_POISSON_WARNING_RATIO
    large_discount = np.isfinite(discount) and discount > MC_DISCOUNT_WARNING_FRACTION
    if not (many_pixels or large_discount):
        return
    _MC_WARNING_ISSUED[0] = True
    fraction = metrics.get('mc_dominated_fraction', np.nan)
    loss_str = ''
    if np.isfinite(discount):
        loss_str = (f" Including it lowers the {loss_function}{suffix} from {metrics[f'{loss_function}_without_mc{suffix}']:.4g} "
                    f"to {metrics[loss_function + suffix]:.4g} ({discount:.0%} lower).")
    print(f"WARNING: the Monte Carlo variance of the simulation exceeds the expected counting (Poisson) variance "
          f"in {fraction:.1%} of the unmasked pixels (95th percentile of their ratio: {ratio:.2f}).{loss_str} "
          f"The loss accounts for it, but the comparison is limited by the simulation statistics there and a "
          f"misfit of the model is penalised less; consider more simulated statistics (more --outgoing_directions, "
          f"especially along the direction in which the pattern is steep, or more incident neutrons). Loss values "
          f"from different outgoing-direction grids or MCPL files are not directly comparable.")

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
    fieldnames = list(dict.fromkeys(key for r in records for key in r))  # all columns (continued fits: earlier records first)
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
        # (TOF: the plots of the first wavelength slice; those of the others have the suffix _slice<n>)
        png_files = [f for f in glob.glob(pattern) if "_joint_" not in os.path.basename(f) and "_slice" not in os.path.basename(f)]

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
    mcpl_metadata: Optional[Dict[str, Any]] = None,
    run: Optional[fit_restart.FitRun] = None
) -> None:
    """
    Execute an automated parameter fit using scipy.optimize (see fit_optimize), single-sample or joint.

    The fit lives in --output_dir (fit_restart): an unfinished fit there with the same settings is continued (its
    evaluations are retraced from fit_evaluations.jsonl without simulation), a finished one or one with other
    settings is not touched, and --overwrite starts from scratch. Every evaluation is written to fit_summary.csv
    and fit_evaluations.jsonl as soon as it is done.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    particles : Any
        Preconditioned simulation particles.
    particle_type : str
        Particle type identifier.
    hist_nxs, hist_nxs_error : np.ndarray
        NeXus data histogram and its errors.
    y_edges_nxs, z_edges_nxs : np.ndarray
        NeXus Q edges.
    mask : np.ndarray
        Mask array.
    run : fit_restart.FitRun, optional
        The state of the output directory (prepared by main(); made here if not given).
    """
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

    def is_integer(name: str) -> bool:
        base_name = name[3:] if name.startswith(('s1_', 's2_')) else name
        return name in fit_integers or base_name in fit_integers
    integer_mask = [is_integer(name) for name in param_names]

    optimizer = args.optimizer.lower()
    polish_method = getattr(args, 'polish', None)
    if polish_method and optimizer != 'differential-evolution':
        raise ValueError("--polish requires --optimizer differential-evolution")

    print(f"\nStarting automated optimization using {args.optimizer.upper()} optimizer"
          + (f", then a {polish_method} polish of its best point" if polish_method else '') + "...")
    print(f"Parameters to fit: {param_names}")
    if fit_integers:
        print(f"Integer parameters: {sorted(list(fit_integers))}" + (" (fixed at the DE best during the polish)" if polish_method else ''))
    print(f"Initial guess x0: {x0}")
    print(f"Bounds: {bounds}")
    print(f"Max evaluations: {args.max_evals}" + (f" (+ at most {args.polish_max_evals} for the polish)" if polish_method else ''))
    objective_key = objective_loss_key(args)
    print(f"Loss metric: {objective_key}" + (f" (of the Qy profile of {band_description(args)}, --fit_objective 1d)"
                                             if objective_key != args.loss_function else ''))
    print(f"Convergence tolerances: xatol={args.xatol}, fatol={args.fatol}"
          + (f"; polish: xatol={args.polish_xatol}, fatol={args.polish_fatol}" if polish_method else ''))

    continuation = None
    if getattr(args, 'de_continue', None):
        if optimizer != 'differential-evolution':
            raise ValueError("--de_continue requires --optimizer differential-evolution")
        continuation = load_de_continuation(args.de_continue, param_names, objective_key,
                                            fit_optimize.de_population_size(len(param_names), args.popsize))
        print(f"Starting from the population of the DE fit {continuation['summary_path']}: {len(continuation['records'])} evaluations "
              f"(last #{continuation['last_eval_index']}), recovered population of {len(continuation['energies'])}, "
              f"losses {np.min(continuation['energies']):.4f} .. {np.max(continuation['energies']):.4f} "
              f"(spread {np.std(continuation['energies']):.4f}); best so far #"
              f"{min(continuation['records'], key=lambda r: r[objective_key] if not isinstance(r[objective_key], str) else np.inf)['eval_index']}")

    if run is None:
        run = fit_restart.prepare_output_dir(args, getattr(args, '_settings', None) or fit_restart.settings_fingerprint(args))
    run.install_signal_handlers()

    eval_offset = continuation['last_eval_index'] if continuation else 0  # evaluation numbers continue
    eval_counter = [eval_offset]
    records = list(continuation['records']) if continuation else []
    phase = ['de' if optimizer == 'differential-evolution' else optimizer]
    budget = {'de': args.max_evals, 'polish': getattr(args, 'polish_max_evals', None)}
    start_total_time = time.time()
    fit_background = getattr(args, 'fit_background', False)
    if fit_background:
        print("Flat background: fitted for each evaluation (--fit_background)")

    def point_string(display_point: Dict[str, Any]) -> str:
        return ', '.join(f"{k}={v}" if isinstance(v, int) else f"{k}={format_fit_value(v)}" for k, v in display_point.items())

    def print_progress(message: str, eval_start_time: float) -> None:
        eval_duration = time.time() - eval_start_time
        new_evals = max(1, run.new_evaluations)
        avg_iter_time = (time.time() - start_total_time) / new_evals
        total_budget = eval_offset + (budget['de'] or 0) + ((budget['polish'] or 0) if polish_method else 0)
        eta = avg_iter_time * max(0, total_budget - eval_counter[0])
        print(f"Fit Eval #{eval_counter[0]}/{total_budget}: {message} | Iter: {eval_duration:.2f}s | Avg: {avg_iter_time:.2f}s | "
              f"ETA: {format_time(eta)}" + (f" | phase {phase[0]}" if polish_method else ''))

    def objective_function(x):
        eval_start_time = time.time()
        eval_counter[0] += 1

        # Map continuous variables to rounded integers where configured
        values = [int(np.round(val)) if is_integer(name) else float(val) for name, val in zip(param_names, x)]
        display_point = dict(zip(param_names, values))
        grid_point_s1 = {name: values[idx] for name, idx in s1_map.items()}

        # an interrupted fit retraces its evaluations from the cache (fit_restart)
        cached = run.cache.next_cached(phase[0], x)
        if cached is not None:
            if cached['eval_index'] != eval_counter[0]:
                print(f"WARNING: cached evaluation #{cached['eval_index']} answers request #{eval_counter[0]}")
            if cached.get('record') is not None:
                records.append(cached['record'])
                save_summary_csv(records, args.output_dir, "fit_summary.csv")
            run.record(cached, cached=True)
            print(f"Fit Eval #{eval_counter[0]} (from the cache, phase {phase[0]}): {point_string(display_point)} --> {objective_key} = {cached['loss']:.4f}")
            return cached['loss']
        run.print_retrace_done()
        entry = {'eval_index': eval_counter[0], 'phase': phase[0], 'x': [float(v) for v in x]}

        # Bounds penalty (the optimizers are given the bounds; this guards rounding and DE edge cases)
        for name, val, (low, high) in zip(param_names, values, bounds):
            if (low is not None and val < low) or (high is not None and val > high):
                run.record(dict(entry, loss=1e9, record=None), cached=False)
                print_progress(f"Bound constraint violated ({name}: {val} outside [{low}, {high}]). Penalty applied", eval_start_time)
                return 1e9

        if not is_joint_fit:
            metrics, _, _ = run_simulation_evaluation(
                grid_point_s1, args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask,
                label_prefix=f"fit_eval_{eval_counter[0]}", mcpl_metadata=mcpl_metadata
            )
            rec = copy.deepcopy(display_point)
            rec['eval_index'] = eval_counter[0]
            rec['phase'] = phase[0]
            if fit_background:
                rec.update(background_entries(metrics))
            for key in loss_keys(metrics):
                rec[key] = metrics[key]
            loss = metrics[objective_key]
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
            rec['phase'] = phase[0]
            if fit_background:
                rec['background_sample1'] = metrics1['background']
                rec['background_sample2'] = metrics2['background']
            for key in loss_keys(metrics1):
                rec[f"{key}_sample1"] = metrics1[key]
                rec[f"{key}_sample2"] = metrics2[key]
                rec[key] = metrics1[key] + metrics2[key]
            loss = rec[objective_key]

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
        run.record(dict(entry, loss=float(loss), record=rec), cached=False)

        background_str = ''
        if fit_background:
            background_keys = ['background'] if 'background' in rec else [key for key in rec if key.startswith('background_s')]
            background_str = " | background = " + ", ".join(f"{rec[key]:.3f}" for key in background_keys)
        # the same loss of the other comparison (image or Qy profile of the Qz band), for information
        other_key = args.loss_function if objective_key != args.loss_function else f"{args.loss_function}_1d"
        other_str = f" ({other_key} = {rec[other_key]:.4f})" if other_key in rec else ''
        print_progress(f"{point_string(display_point)} --> {objective_key} = {loss:.4f}{other_str}{background_str}", eval_start_time)
        return loss

    phases = []  # (description, scipy result)
    try:
        if optimizer == 'differential-evolution':
            de_objective = objective_function
            if continuation:
                de_objective = known_population_objective(objective_function, continuation['population'], continuation['energies'], bounds)
            opt_res = fit_optimize.run_differential_evolution(
                de_objective, bounds, x0, integer_mask, args.popsize, args.max_evals, args.fatol, getattr(args, 'seed', None),
                init=continuation['population'] if continuation else None, stream=eval_offset)
            phases.append(("differential evolution", opt_res))
            if polish_method:
                print(f"\n========== POLISH: {polish_method} from the DE best (loss {opt_res.fun:.4f}) ==========")
                phase[0] = 'polish'
                polish_res, _ = fit_optimize.polish(objective_function, opt_res.x, bounds, integer_mask, polish_method,
                                                    args.polish_max_evals, args.polish_xatol, args.polish_fatol)
                phases.append((f"polish ({polish_method})", polish_res))
        else:
            opt_res, _ = fit_optimize.run_local(objective_function, x0, bounds, optimizer, args.max_evals, args.xatol,
                                                args.fatol, integer_mask)
            phases.append((optimizer, opt_res))
    except BaseException as err:
        if not run._terminated:
            run.finish('failed', f"{type(err).__name__}: {err}")
        raise

    total_runtime = time.time() - start_total_time
    new_evals = max(1, run.new_evaluations)
    valid = [r for r in records if not np.isnan(r[objective_key])]
    best_rec = min(valid, key=lambda r: r[objective_key]) if valid else None

    main_res = phases[0][1]
    fit_results_lines = [
        f"Optimizer Success: {main_res.success}",
        f"Optimizer Message: {main_res.message}",
    ]
    for name, res in phases[1:]:
        fit_results_lines += [f"Polish Success: {res.success}", f"Polish Message: {res.message}"]
    for name, res in phases:
        fit_results_lines.append(f"Phase {name}: best loss {res.fun:.4f}, {res.nfev} evaluations")
    if best_rec is not None:
        fit_results_lines.append(f"Best Loss ({objective_key}): {best_rec[objective_key]:.4f}")
        fit_results_lines.append(f"Best Evaluation: #{best_rec['eval_index']}" + (f" (phase {best_rec['phase']})" if 'phase' in best_rec else ''))
        fit_results_lines.append("Optimal Parameters:")
        for k in param_names:
            v = best_rec[k]
            fit_results_lines.append(f"  {k} = {int(v) if is_integer(k) else format_fit_value(float(v))}")
    if fit_background and best_rec is not None:
        if 'background' in best_rec:
            fit_results_lines.append(f"  background = {best_rec['background']:.4f} [counts per pixel] "
                                     f"(fitted flat background of the best evaluation, #{best_rec['eval_index']})")
        elif 'background_slice1' in best_rec:
            slice_backgrounds = ", ".join(f"{best_rec[key]:.4f}" for key in best_rec if key.startswith('background_slice'))
            fit_results_lines.append(f"  background = {slice_backgrounds} (wavelength slices) [counts per pixel] "
                                     f"(fitted flat backgrounds of the best evaluation, #{best_rec['eval_index']})")
        else:
            fit_results_lines.append(f"  background = {best_rec['background_sample1']:.4f} (sample 1), "
                                     f"{best_rec['background_sample2']:.4f} (sample 2) [counts per pixel] "
                                     f"(fitted flat backgrounds of the best evaluation, #{best_rec['eval_index']})")

    fit_results_lines.extend([
        "\n--- Runtime Statistics ---",
        f"Total Runtime (this run): {format_time(total_runtime)} ({total_runtime:.2f}s)",
        f"Average Iteration Runtime (new evaluations): {total_runtime / new_evals:.2f}s",
        f"Evaluations: {eval_counter[0]} ({run.new_evaluations} simulated in this run, {run.cached_used} retraced from the cache)"
    ])

    extra_summary_text = "\n".join(fit_results_lines)

    save_and_print_summary(records, args.output_dir, "fit_summary.csv", "Optimization", extra_summary_text=extra_summary_text, sort_key=objective_key)

    result = {'message': str(phases[-1][1].message), 'objective': objective_key,
              'phases': [{'name': name, 'success': bool(res.success), 'message': str(res.message), 'best_loss': float(res.fun),
                          'evaluations': int(res.nfev), 'budget_exhausted': 'maximum number' in str(res.message).lower()}
                         for name, res in phases]}
    if best_rec is not None:
        result.update(best_loss=float(best_rec[objective_key]), best_eval_index=int(best_rec['eval_index']),
                      best_parameters={k: best_rec[k] for k in param_names})
    run.finish('finished', result['message'], result)

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

        background_str = ''.join(f", {key}={value:.3f}" for key, value in background_entries(metrics).items()) \
            if getattr(args, 'fit_background', False) else ''
        loss_str = ', '.join(f"{key}={metrics[key]:.4e}" if key.startswith('log_residual') else f"{key}={metrics[key]:.4f}"
                             for key in loss_keys(metrics))
        print(f"Fit results: {loss_str}{background_str} | Iter: {iter_duration:.2f}s | Avg: {avg_iter_time:.2f}s | ETA: {format_time(eta)}")
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

    save_and_print_summary(records, args.output_dir, "scan_summary.csv", "Scan", extra_summary_text=extra_summary_text, sort_key=objective_loss_key(args))

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
    args._settings = fit_restart.settings_fingerprint(args)  # before mg_fit adds derived attributes
    os.makedirs(args.output_dir, exist_ok=True)

    validate_fit_args(args, parser)

    # a fit: decide what to do with --output_dir (new fit, continue an unfinished one, or stop) before loading anything
    is_fit = bool(args.fit or args.fit2 or args.fit_common) and not args.mask_view
    run = fit_restart.prepare_output_dir(args, args._settings) if is_fit else None

    hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw = prepare_experimental_data(args)

    if args.mask_view:
        # Determine default intensity_min if not provided
        if args.intensity_min is not None:
            intensity_min = float(args.intensity_min)
        else:
            intensity_min = 1.0 if args.experiment_time else 1e-9
        def band_counts(hist, z_edges, band):  # the measured Qy profile of --band_exit_angle
            if band is None:
                return None
            zeros = np.zeros_like(hist)
            return {'counts': qz_band_profile(hist, zeros, zeros, z_edges, 0.0, 0.0, band=band)[0],
                    'title': f"Qy profile of {band_description(args)}"}
        # TOF: one plot per wavelength slice (suffix _slice<n> after the first)
        views = [(s.hist_nxs_raw, s.hist_nxs_error_raw, s.hist_nxs, s.hist_nxs_error, s.y_edges, s.z_edges,
                  None if s.profile_hist_nxs is None else band_counts(s.profile_hist_nxs, s.z_edges, np.ones(s.q_shape, dtype=bool)))
                 for s in getattr(args, 'tof_slices', None) or []] or \
                [(hist_nxs_raw, hist_nxs_error_raw, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs,
                  band_counts(hist_nxs, z_edges_nxs, getattr(args, 'band_pixels', None)))]
        for index, (raw, raw_error, masked, masked_error, y_edges, z_edges, profile) in enumerate(views):
            plot_path = os.path.join(args.output_dir, "masked_view.png" if index == 0 else f"masked_view_slice{index + 1}.png")
            y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges[0], y_edges[-1]]
            z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges[0], z_edges[-1]]
            save_view_masks_plot(
                raw, raw_error, masked, masked_error, y_edges, z_edges,
                args.q_min, args.q_max, y_plot_range, z_plot_range,
                plot_path, intensity_min, profile=profile,
                masked_style=getattr(args, 'mask_view_style', 'grey')
            )
        return

    particles, particle_type, mcpl_metadata = load_and_precondition_particles(args)
    if getattr(args, 'simulate_mask_angle_range', False):
        _set_simulated_angle_range(args, particles, particle_type)
    set_outgoing_directions_from_sampling(args, particles, particle_type)  # once, for the final angle range

    if args.fit or args.fit2 or args.fit_common:
        run_automated_fit(args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, mcpl_metadata, run=run)
    else:
        run_parameter_scan(args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, mcpl_metadata)

if __name__ == '__main__':
    main()
