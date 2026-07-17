#!/usr/bin/env python3

"""
Executes a parameter scan (grid sweep) of BornAgain simulations, evaluates the 
match against experimental NeXus data, and outputs a summary of fit metrics.
"""

import os
import copy
import csv
import itertools
import numpy as np
from multiprocessing import cpu_count

from .run_cli import create_argparser as create_run_parser, parse_args as parse_run_args
from .input_output import get_particles, save_q_histogram_file
from .preconditioning import precondition
from .parameters import pack_parameters
from .run import process_particles, process_particles_parallelly
from .read_d22 import read_nexus_data
from .experiment_time import upscale_simple

def create_scan_parser():
  parser = create_run_parser()
  
  # Make filename optional for the run parser, since we may only want to view masks without running a simulation
  for action in parser._actions:
    if action.dest == 'filename':
      action.nargs = '?'
      action.default = ''
      
  scan_group = parser.add_argument_group('Parameter scan options')
  scan_group.add_argument('--scan', action='append', nargs='+', required=False,
                          help='Parameter name followed by values to scan, e.g., --scan radius 10 12 15')
  scan_group.add_argument('--nxs', type=str, required=True, help='Path to experimental NeXus file to match.')
  scan_group.add_argument('--experiment_time', type=float, default=None, help='Virtual experiment time in seconds for upscaling the simulation.')
  scan_group.add_argument('--background', type=float, default=0.0, help='Flat background level added during upscaling.')
  scan_group.add_argument('--output_dir', type=str, default='scan_results', help='Directory to save scan results.')
  
  scan_plot_group = parser.add_argument_group('Plotting options for scan results')
  scan_plot_group.add_argument('--png', action='store_true', help='Generate comparison PNG plot for each simulation configuration.')
  scan_plot_group.add_argument('--y_plot_range', nargs=2, type=float, help='Plot y range.')
  scan_plot_group.add_argument('--z_plot_range', nargs=2, type=float, help='Plot z range.')
  scan_plot_group.add_argument('--q_min', type=float, default=0.0, help='Minimum Qz value for 1D slice comparison [1/nm].')
  scan_plot_group.add_argument('--q_max', type=float, default=0.0, help='Maximum Qz value for 1D slice comparison [1/nm].')
  
  
  scan_mask_group = parser.add_argument_group('Masking options to exclude data ranges for the fitness calculation')
  scan_mask_group.add_argument('--view_masks', action='store_true', help='Only view the applied masks on the experimental NeXus data, then exit.')
  scan_mask_group.add_argument('--mask_qy_range', nargs=2, type=float, default=None, help='Qy range to mask out from the fitness calculation (e.g., -0.05 0.05) [1/nm].')
  scan_mask_group.add_argument('--qy_min_cut', type=float, default=None, help='Lower Qy cut option. Any data below this Qy value is disregarded [1/nm].')
  scan_mask_group.add_argument('--qy_max_cut', type=float, default=None, help='Upper Qy cut option. Any data above this Qy value is disregarded [1/nm].')
  scan_mask_group.add_argument('--qz_min_cut', type=float, default=None, help='Lower Qz cut option. Any data below this Qz value is disregarded [1/nm].')
  scan_mask_group.add_argument('--qz_max_cut', type=float, default=None, help='Upper Qz cut option. Any data above this Qz value is disregarded [1/nm].')
  
  return parser

def parse_scan_arguments(scan_args):
  scanned_params = {}
  for item in scan_args:
    if len(item) < 2:
      raise ValueError(f"Scan parameter must have at least one value: {item}")
    name = item[0]
    values = []
    for val_str in item[1:]:
      try:
        val = int(val_str)
      except ValueError:
        try:
          val = float(val_str)
        except ValueError:
          val = val_str
      values.append(val)
    scanned_params[name] = values
  return scanned_params

def convert_val(value_str):
  try:
    return int(value_str)
  except ValueError:
    try:
      return float(value_str)
    except ValueError:
      return value_str

def get_mask(y_edges, z_edges, mask_qy_range=None,
             qy_min_cut=None, qy_max_cut=None, qz_min_cut=None, qz_max_cut=None,
             shape=None):
  """Calculates a boolean mask of shape matching the detector data.
  True = keep, False = mask out."""
  if shape is None:
    shape = (len(y_edges) - 1, len(z_edges) - 1)
  
  keep_mask = np.ones(shape, dtype=bool)
  
  y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
  z_centres = (z_edges[:-1] + z_edges[1:]) / 2.0
  
  # Center Qy mask
  if mask_qy_range is not None:
    qy_mask = (y_centres >= mask_qy_range[0]) & (y_centres <= mask_qy_range[1])
    if shape[0] == len(y_centres):
      keep_mask[qy_mask, :] = False
    else:
      keep_mask[:, qy_mask] = False
      
  # Qy min cut
  if qy_min_cut is not None:
    qy_mask = y_centres < qy_min_cut
    if shape[0] == len(y_centres):
      keep_mask[qy_mask, :] = False
    else:
      keep_mask[:, qy_mask] = False
      
  # Qy max cut
  if qy_max_cut is not None:
    qy_mask = y_centres > qy_max_cut
    if shape[0] == len(y_centres):
      keep_mask[qy_mask, :] = False
    else:
      keep_mask[:, qy_mask] = False
      
  # Qz min cut
  if qz_min_cut is not None:
    qz_mask = z_centres < qz_min_cut
    if shape[0] == len(z_centres):
      keep_mask[qz_mask, :] = False
    else:
      keep_mask[:, qz_mask] = False
      
  # Qz max cut
  if qz_max_cut is not None:
    qz_mask = z_centres > qz_max_cut
    if shape[0] == len(z_centres):
      keep_mask[qz_mask, :] = False
    else:
      keep_mask[:, qz_mask] = False
      
  return keep_mask

def apply_mask(data, mask, fill_value):
  """Applies a precalculated boolean mask to the data, replacing False values with fill_value."""
  res = data.astype(np.float64) if isinstance(fill_value, float) and np.isnan(fill_value) else data.copy()
  res[~mask] = fill_value
  return res

def calculate_fitness(hist_nxs, hist_nxs_error, hist_sim, hist_sim_error):
  """Evaluates fitness directly on the pre-masked histograms.
  Masked regions are represented by NaN in the histograms."""
  valid_mask = np.isfinite(hist_nxs) & np.isfinite(hist_sim)
  
  I_exp = hist_nxs[valid_mask]
  I_sim = hist_sim[valid_mask]
  sigma_exp = hist_nxs_error[valid_mask]
  sigma_sim = hist_sim_error[valid_mask]
  
  sigma_exp = np.where(sigma_exp > 0, sigma_exp, 1.0)
  
  # Chi-square: weighted square deviations directly comparing absolute counts
  total_error_sq = sigma_exp**2 + sigma_sim**2
  total_error_sq = np.where(total_error_sq > 0, total_error_sq, 1.0)
  chi2 = np.sum(((I_exp - I_sim) ** 2) / total_error_sq)
  reduced_chi2 = chi2 / len(I_exp) if len(I_exp) > 0 else np.nan
  
  # Logarithmic residual:
  pos_mask = (I_exp > 0) & (I_sim > 0)
  if np.any(pos_mask):
    log_I_exp = np.log10(I_exp[pos_mask])
    log_I_sim = np.log10(I_sim[pos_mask])
    log_residual = np.mean((log_I_exp - log_I_sim) ** 2)
  else:
    log_residual = np.nan
    
  return reduced_chi2, log_residual

def save_comparison_plot(hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs,
                         hist_sim, hist_sim_error, y_edges_sim, z_edges_sim,
                         q_min, q_max, y_plot_range, z_plot_range,
                         savename, label_sim):
  import matplotlib.pyplot as plt
  from .plotting_utils import plot_q_1d, log_plot_2d, extract_range_to_1d
  
  intensity_min = 1.0
  
  fig, axes = plt.subplots(2, 2, figsize=(16, 12))
  
  # Plot 2D maps (no grid is added to these)
  log_plot_2d(hist_nxs, y_edges_nxs, z_edges_nxs, "D22 measurement", ax=axes[0, 0],
              intensity_min=intensity_min, intensity_max=hist_nxs[~np.isnan(hist_nxs)].max(),
              y_range=y_plot_range, z_range=z_plot_range, output='none')
              
  log_plot_2d(hist_sim, y_edges_sim, z_edges_sim, label_sim, ax=axes[0, 1],
              intensity_min=intensity_min, intensity_max=hist_nxs[~np.isnan(hist_nxs)].max(),
              y_range=y_plot_range, z_range=z_plot_range, output='none')
              
  gs = axes[1, 0].get_gridspec()
  axes[1, 0].remove()
  axes[1, 1].remove()
  ax_bottom = fig.add_subplot(gs[1:, :])
  
  qz_min_index = np.digitize(q_min, z_edges_sim) - 1
  qz_max_index = np.digitize(q_max, z_edges_sim)
  
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
            
  axes[0, 0].axhline(z_edges_sim[qz_min_index], color='magenta', linestyle='--')
  axes[0, 0].axhline(z_edges_sim[qz_max_index], color='magenta', linestyle='--')
  axes[0, 1].axhline(z_edges_sim[qz_min_index], color='magenta', linestyle='--')
  axes[0, 1].axhline(z_edges_sim[qz_max_index], color='magenta', linestyle='--')
  
  # Format 1D overlay plot (grid only on the major ticks of this plot)
  ax_bottom.set_title(f"Qz=[{z_limits[0]:.4f} 1/nm, {z_limits[1]:.4f} 1/nm]")
  ax_bottom.grid(True, which='major')
  ax_bottom.legend(loc='upper left')
  
  plt.tight_layout()
  plt.savefig(savename, dpi=300)
  plt.close(fig)
  print(f"Created comparison plot: {savename}")
 
def save_view_masks_plot(hist_raw, hist_raw_error, hist_masked, hist_masked_error,
                         y_edges_nxs, z_edges_nxs, q_min, q_max, y_plot_range, z_plot_range,
                         savename):
  import matplotlib.pyplot as plt
  from .plotting_utils import plot_q_1d, log_plot_2d, extract_range_to_1d
  
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

def main():
  parser = create_scan_parser()
  args = parse_run_args(parser)
  
  # Setup output directory
  os.makedirs(args.output_dir, exist_ok=True)
  
  # Manual validation for required options
  if not args.view_masks:
    if not args.filename:
      parser.error("the following arguments are required: filename")
    if not args.scan:
      parser.error("the following arguments are required: --scan")
  if not args.nxs:
    parser.error("the following arguments are required: --nxs")
      
  # Load experimental data
  wavelength_val = args.wavelength_selected if args.wavelength_selected else (args.wavelength if args.wavelength else 6.0)
  print(f"Loading experimental NeXus data from: {args.nxs}")
  hist_nxs_raw, hist_nxs_error_raw, y_edges_nxs, z_edges_nxs = read_nexus_data(
      args.nxs, args.alpha, wavelength_val, args.sample_orientation
  )
  print(f"Loaded NeXus dataset of shape {hist_nxs_raw.shape}")
  
  # Calculate boolean mask once
  mask = get_mask(
      y_edges_nxs, z_edges_nxs,
      mask_qy_range=args.mask_qy_range,
      qy_min_cut=args.qy_min_cut, qy_max_cut=args.qy_max_cut,
      qz_min_cut=args.qz_min_cut, qz_max_cut=args.qz_max_cut,
      shape=hist_nxs_raw.shape
  )
  
  # Pre-mask the nexus data
  hist_nxs = apply_mask(hist_nxs_raw, mask, np.nan)
  hist_nxs_error = apply_mask(hist_nxs_error_raw, mask, 0.0)

  # Short-circuit if only viewing masks
  if args.view_masks:
    plot_path = os.path.join(args.output_dir, "masked_view.png")
    y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges_nxs[0], y_edges_nxs[-1]]
    z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges_nxs[0], z_edges_nxs[-1]]
    save_view_masks_plot(
        hist_nxs_raw, hist_nxs_error_raw,
        hist_nxs, hist_nxs_error,
        y_edges_nxs, z_edges_nxs,
        args.q_min, args.q_max, y_plot_range, z_plot_range,
        plot_path
    )
    return
  
  # Parse parameter grid
  scanned_params = parse_scan_arguments(args.scan)
  keys = list(scanned_params.keys())
  value_lists = [scanned_params[k] for k in keys]
  
  grid = []
  for combo in itertools.product(*value_lists):
    grid.append(dict(zip(keys, combo)))
    
  print(f"Starting parameter scan with {len(grid)} configurations...")
  
  # Load MCPL particles once
  from .tof_filtering import get_tof_filtering_limits
  tof_limits = get_tof_filtering_limits(args)
  particles, particle_type = get_particles(args.filename, args.intensity_factor, tof_limits, args.input_weight_limit, use_polarization=args.use_polarization)
  
  # Precondition particles once
  particles = precondition(particles, args)
  print(f"Loaded and preconditioned {len(particles)} particles.")

  records = []
  
  for idx, grid_point in enumerate(grid):
    print(f"\n[{idx+1}/{len(grid)}] Running simulation with: {grid_point}")
    
    # Parse existing sample arguments
    sample_args_dict = {}
    if args.sample_arguments:
      for pair in args.sample_arguments.split(';'):
        if '=' in pair:
          k, v = pair.split('=')
          sample_args_dict[k.strip()] = convert_val(v.strip())
          
    # Apply scan parameters
    for k, v in grid_point.items():
      sample_args_dict[k] = v
      
    # Update args.sample_arguments
    args.sample_arguments = ';'.join(f"{k}={v}" for k, v in sample_args_dict.items())
    
    # Pack parameters
    params = pack_parameters(args, particle_type)
    
    # Execute simulation
    if args.no_parallel:
      result = process_particles(particles, params)
    else:
      process_number = args.parallel_processes if args.parallel_processes else (cpu_count() - 2)
      result = process_particles_parallelly(particles, params, process_number)
      
    # Extract histogram
    q_hist = result['qHist']
    q_hist_weights_squared = result['qHistWeightsSquared']
    q_hist_error = np.sqrt(q_hist_weights_squared)
    edges = [np.array(np.histogram_bin_edges(None, bins=b, range=r), dtype=np.float64)
             for b, r in zip(params['bins'], params['hist_ranges'])]
             
    # Save simulation output file
    param_str = '_'.join(f"{k}_{v}" for k, v in grid_point.items())
    savename = os.path.join(args.output_dir, f"sim_{param_str}")
    save_q_histogram_file(savename, q_hist, q_hist_error, edges)
    
    # Evaluate fitness
    record = copy.deepcopy(grid_point)
    if hist_nxs is not None:
      hist_sim = np.sum(q_hist, axis=2)
      hist_sim_error = np.sqrt(np.sum(q_hist_weights_squared, axis=2))
      
      # Handle potential orientation/shape transpositions
      if hist_nxs.shape != hist_sim.shape:
        if hist_nxs.shape == hist_sim.T.shape:
          hist_sim = hist_sim.T
          hist_sim_error = hist_sim_error.T
        else:
          raise ValueError(f"Incompatible shapes: NeXus={hist_nxs.shape}, Sim={hist_sim.shape}")
      
      
      if args.experiment_time:
        hist_sim, hist_sim_error = upscale_simple(hist_sim, hist_sim_error, args.experiment_time, args.background)
        
      # Mask the simulated data using the precalculated mask
      hist_sim_masked = apply_mask(hist_sim, mask, np.nan)
      hist_sim_error_masked = apply_mask(hist_sim_error, mask, 0.0)
          
      reduced_chi2, log_residual = calculate_fitness(
          hist_nxs, hist_nxs_error, hist_sim_masked, hist_sim_error_masked
      )
      record['reduced_chi2'] = reduced_chi2
      record['log_residual'] = log_residual
      print(f"Fit results: reduced_chi2={reduced_chi2:.4f}, log_residual={log_residual:.4e}")
      
      if args.png:
        plot_path = os.path.join(args.output_dir, f"sim_{param_str}.png")
        y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges_nxs[0], y_edges_nxs[-1]]
        z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges_nxs[0], z_edges_nxs[-1]]
        save_comparison_plot(
            hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs,
            hist_sim_masked, hist_sim_error_masked, edges[0], edges[1],
            args.q_min, args.q_max, y_plot_range, z_plot_range,
            plot_path, f"Sim ({param_str})"
        )
      
    records.append(record)
    
  # Write summary table using Python built-in csv module
  summary_path = os.path.join(args.output_dir, "scan_summary.csv")
  if records:
    fieldnames = list(records[0].keys())
    with open(summary_path, mode='w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for r in records:
        writer.writerow(r)
        
  print(f"\nScan complete! Summary saved to: {summary_path}")
  print("\n--- Scan Results ---")
  if records:
    # Build a simple text-based table
    headers = list(records[0].keys())
    col_widths = {h: max(len(h), 12) for h in headers}
    
    # Calculate max width based on data
    for r in records:
      for h in headers:
        val_str = f"{r[h]:.4e}" if isinstance(r[h], float) else str(r[h])
        col_widths[h] = max(col_widths[h], len(val_str))
        
    header_row = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_row)
    print(separator)
    for r in records:
      row_str = " | ".join(
          (f"{r[h]:<{col_widths[h]}.4e}" if isinstance(r[h], float)
           else f"{str(r[h]):<{col_widths[h]}}")
          for h in headers
      )
      print(row_str)
  else:
    print("No records to display.")

if __name__ == '__main__':
  main()
