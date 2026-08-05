#!/usr/bin/env python3

"""
Executes a parameter scan (grid sweep) of BornAgain simulations, evaluates the 
match against experimental NeXus data, and outputs a summary of fit metrics.
"""

import os
import copy
import csv
import itertools
import time
import numpy as np
from multiprocessing import cpu_count

from .run_cli import create_argparser as create_run_parser, parse_args as parse_run_args
from .input_output import get_particles, save_q_histogram_file
from .preconditioning import precondition
from .parameters import pack_parameters
from .run import process_particles, process_particles_parallelly
from .read_d22 import read_nexus_data
from .experiment_time import upscale_simple
from .masking import get_mask, apply_mask, save_view_masks_plot

def format_time(seconds):
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
  scan_group.add_argument('--poisson_sampling', action='store_true', help='Enable random Poisson noise sampling on the simulated data. (Off by default during scans/fits to ensure deterministic, smooth objective function evaluation for optimizer convergence.)')
  scan_group.add_argument('--output_dir', type=str, default='scan_results', help='Directory to save scan results.')
  
  scan_plot_group = parser.add_argument_group('Plotting options for scan results')
  scan_plot_group.add_argument('--png', action='store_true', help='Generate comparison PNG plot for each simulation configuration.')
  scan_plot_group.add_argument('--y_plot_range', nargs=2, type=float, help='Plot y range.')
  scan_plot_group.add_argument('--z_plot_range', nargs=2, type=float, help='Plot z range.')
  scan_plot_group.add_argument('--q_min', type=float, default=0.0, help='Minimum Qz value for 1D slice comparison [1/nm].')
  scan_plot_group.add_argument('--q_max', type=float, default=0.0, help='Maximum Qz value for 1D slice comparison [1/nm].')
  
  
  scan_mask_group = parser.add_argument_group('Masking options to exclude data ranges for the fitness calculation')
  scan_mask_group.add_argument('--mask_view', action='store_true', help='Only view the applied masks on the experimental NeXus data, then exit.')
  scan_mask_group.add_argument('--mask_qy_range', nargs=2, type=float, default=None, help='Qy range to mask out from the fitness calculation (e.g., -0.05 0.05) [1/nm].')
  scan_mask_group.add_argument('--mask_qy_min_cut', type=float, default=None, help='Lower Qy cut option. Any data below this Qy value is disregarded [1/nm].')
  scan_mask_group.add_argument('--mask_qy_max_cut', type=float, default=None, help='Upper Qy cut option. Any data above this Qy value is disregarded [1/nm].')
  scan_mask_group.add_argument('--mask_qz_min_cut', type=float, default=None, help='Lower Qz cut option. Any data below this Qz value is disregarded [1/nm].')
  scan_mask_group.add_argument('--mask_qz_max_cut', type=float, default=None, help='Upper Qz cut option. Any data above this Qz value is disregarded [1/nm].')
  scan_mask_group.add_argument('--mask_exclude_q_box', action='append', nargs=4, type=float, default=None,
                               help='Exclude rectangular Q-region defined by 4 numbers: qy_min qy_max qz_min qz_max [1/nm]. (Can be specified multiple times).')
  scan_mask_group.add_argument('--mask_include_q_box', action='append', nargs=4, type=float, default=None,
                               help='Include rectangular Q-region defined by 4 numbers: qy_min qy_max qz_min qz_max [1/nm]. Applied after exclusions. (Can be specified multiple times).')
  scan_mask_group.add_argument('--simulate_mask_angle_range', action='store_true',
                               help='Calculate minimum simulation angle range enclosing the unmasked detector pixels to optimize performance.')
  scan_mask_group.add_argument('--simulate_mask_angle_range_factor', type=float, default=1.0,
                               help='Expansion factor for --simulate_mask_angle_range (default: 1.0). Use e.g. 1.05 for a 5%% safety margin around the ROI.')
  fit_group = parser.add_argument_group('Automated optimization / fitting options')
  fit_group.add_argument('--fit', action='append', nargs='+', required=False,
                         help='Parameter to fit with initial guess and optional min/max bounds, e.g., --fit radius 51 40 60')
  fit_group.add_argument('--fit_integer', action='append', nargs='+', required=False, default=None,
                         help='Specify parameter names to fit as integers (e.g. --fit_integer layerNumber). These parameters will be constrained to integer values during optimization (rounded for Nelder-Mead and Powell, and natively handled for Differential Evolution).')
  fit_group.add_argument('--optimizer', type=str, default='nelder-mead', choices=['nelder-mead', 'powell', 'differential-evolution'],
                         help='Optimization algorithm to use (default: nelder-mead).')
  fit_group.add_argument('--popsize', type=int, default=15,
                         help='Population size multiplier for Differential Evolution (default: 15). The total population is popsize * number_of_parameters. A smaller value reduces evaluations per generation but reduces search diversity.')
  fit_group.add_argument('--max_evals', type=int, default=10,
                         help='Maximum number of objective function evaluations for the optimizer (default: 10).')
  fit_group.add_argument('--loss_function', type=str, default='reduced_chi2', choices=['reduced_chi2', 'log_residual'],
                         help='Metric to minimize during optimization (default: reduced_chi2).')
  fit_group.add_argument('--xatol', type=float, default=0.01,
                         help='Absolute parameter convergence tolerance. (SciPy default: 1e-4. Suggested for Monte Carlo simulations: 0.01).')
  fit_group.add_argument('--fatol', type=float, default=0.05,
                         help='Absolute loss function convergence tolerance. (SciPy default: 1e-4. Suggested for Monte Carlo simulations: 0.05 matching MC Poisson noise floor).')
  fit_group.add_argument('--gif', action='store_true',
                         help='Generate animated GIF showing the evolution of the fitting process.')

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

def parse_fit_arguments(fit_args):
  """
  Parses --fit arguments.
  Supported formats per parameter:
    --fit name x0
    --fit name min max
    --fit name x0 min max
  Returns:
    param_names: list of parameter names
    x0_list: list of initial values (float)
    bounds_list: list of (min_val, max_val) tuples or None
  """
  param_names = []
  x0_list = []
  bounds_list = []
  
  for item in fit_args:
    if len(item) < 2:
      raise ValueError(f"Fit parameter must specify at least name and initial value: {item}")
    name = item[0]
    param_names.append(name)
    
    if len(item) == 2:
      x0 = float(item[1])
      bounds = (None, None)
    elif len(item) == 3:
      b_min = float(item[1])
      b_max = float(item[2])
      x0 = (b_min + b_max) / 2.0
      bounds = (b_min, b_max)
    else:
      x0 = float(item[1])
      b_min = float(item[2])
      b_max = float(item[3])
      bounds = (b_min, b_max)
      
    x0_list.append(x0)
    bounds_list.append(bounds)
    
  return param_names, x0_list, bounds_list

def convert_val(value_str):
  try:
    return int(value_str)
  except ValueError:
    try:
      return float(value_str)
    except ValueError:
      return value_str

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

def validate_scan_args(args, parser):
  if not args.mask_view:
    if not args.filename:
      parser.error("the following arguments are required: filename")
    if not args.scan and not args.fit:
      parser.error("Either --scan or --fit must be specified.")
  if not args.nxs:
    parser.error("the following arguments are required: --nxs")
  if args.fit and args.optimizer.lower() == 'differential-evolution':
    param_names = [item[0] for item in args.fit]
    _, _, bounds = parse_fit_arguments(args.fit)
    for name, (low, high) in zip(param_names, bounds):
      if low is None or high is None:
        parser.error(f"Differential Evolution optimizer requires finite bounds for all fitted parameters. Please specify bounds in --fit for '{name}' (e.g. --fit {name} <initial_guess> <min_bound> <max_bound>).")

def prepare_experimental_data(args):
  wavelength_val = args.wavelength_selected if args.wavelength_selected else (args.wavelength if args.wavelength else 6.0)
  print(f"Loading experimental NeXus data from: {args.nxs}")
  hist_nxs_raw, hist_nxs_error_raw, y_edges_nxs, z_edges_nxs = read_nexus_data(
      args.nxs, args.alpha, wavelength_val, args.sample_orientation
  )
  print(f"Loaded NeXus dataset of shape {hist_nxs_raw.shape}")
  
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
  
  if getattr(args, 'simulate_mask_angle_range', False):
    from .instrument import Instrument
    from .instrument_defaults import instrument_defaults
    instr_params = instrument_defaults[args.instrument] #note: the instrument defaults are already updated by parse_run_args
    instrument = Instrument(instr_params, args.alpha, wavelength_val, args.sample_orientation, args.wfm, args.no_gravity)
    
    len_y_centres = len(y_edges_nxs) - 1
    factor = getattr(args, 'simulate_mask_angle_range_factor', 1.0)
    mask_angle_range = list(instrument.get_masked_angle_range(mask, len_y_centres, factor=factor))
    args.angle_range = mask_angle_range
    print(f"Mask angle range [deg] (factor={factor:.2f}): horiz=[{mask_angle_range[0]:.4f}, {mask_angle_range[1]:.4f}], vert=[{mask_angle_range[2]:.4f}, {mask_angle_range[3]:.4f}]")

  return hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw

def load_and_precondition_particles(args):
  from .tof_filtering import get_tof_filtering_limits
  tof_limits = get_tof_filtering_limits(args)
  particles, particle_type = get_particles(
      args.filename, args.intensity_factor, tof_limits, args.input_weight_limit, use_polarization=args.use_polarization
  )
  particles = precondition(particles, args)
  print(f"Loaded and preconditioned {len(particles)} particles.")
  return particles, particle_type

def run_simulation_evaluation(grid_point, args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, save_npz=True, label_prefix="sim"):
  sample_args_dict = {}
  if args.sample_arguments:
    for pair in args.sample_arguments.split(';'):
      if '=' in pair:
        k, v = pair.split('=')
        sample_args_dict[k.strip()] = convert_val(v.strip())
        
  for k, v in grid_point.items():
    sample_args_dict[k] = v
    
  args.sample_arguments = ';'.join(f"{k}={v}" for k, v in sample_args_dict.items())
  params = pack_parameters(args, particle_type)
  
  if args.no_parallel:
    result = process_particles(particles, params)
  else:
    process_number = args.parallel_processes if args.parallel_processes else (cpu_count() - 2)
    result = process_particles_parallelly(particles, params, process_number)
    
  q_hist = result['qHist']
  q_hist_weights_squared = result['qHistWeightsSquared']
  q_hist_error = np.sqrt(q_hist_weights_squared)
  edges = [np.array(np.histogram_bin_edges(None, bins=b, range=r), dtype=np.float64)
           for b, r in zip(params['bins'], params['hist_ranges'])]
           
  param_str = '_'.join(f"{k}_{v}" for k, v in grid_point.items())
  if save_npz:
    savename = os.path.join(args.output_dir, f"{label_prefix}_{param_str}")
    save_q_histogram_file(savename, q_hist, q_hist_error, edges)
    
  hist_sim = np.sum(q_hist, axis=2)
  hist_sim_error = np.sqrt(np.sum(q_hist_weights_squared, axis=2))
  
  if hist_nxs.shape != hist_sim.shape:
    if hist_nxs.shape == hist_sim.T.shape:
      hist_sim = hist_sim.T
      hist_sim_error = hist_sim_error.T
    else:
      raise ValueError(f"Incompatible shapes: NeXus={hist_nxs.shape}, Sim={hist_sim.shape}")
      
  if args.experiment_time:
    hist_sim, hist_sim_error = upscale_simple(
        hist_sim, hist_sim_error, args.experiment_time, args.background,
        poisson_sampling=args.poisson_sampling
    )
    
  hist_sim_masked = apply_mask(hist_sim, mask, np.nan)
  hist_sim_error_masked = apply_mask(hist_sim_error, mask, 0.0)
  
  reduced_chi2, log_residual = calculate_fitness(
      hist_nxs, hist_nxs_error, hist_sim_masked, hist_sim_error_masked
  )
  
  record = copy.deepcopy(grid_point)
  record['reduced_chi2'] = reduced_chi2
  record['log_residual'] = log_residual
  
  if args.png:
    plot_path = os.path.join(args.output_dir, f"{label_prefix}_{param_str}.png")
    y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges_nxs[0], y_edges_nxs[-1]]
    z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges_nxs[0], z_edges_nxs[-1]]
    save_comparison_plot(
        hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs,
        hist_sim_masked, hist_sim_error_masked, edges[0], edges[1],
        args.q_min, args.q_max, y_plot_range, z_plot_range,
        plot_path, f"Sim ({param_str})"
    )
    
  return reduced_chi2, log_residual, record

def save_summary_csv(records, output_dir, filename):
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

def save_and_print_summary(records, output_dir, filename, title_header, extra_summary_text=None):
  if records and 'reduced_chi2' in records[0]:
    records.sort(key=lambda r: (np.isnan(r['reduced_chi2']), r['reduced_chi2']))

  save_summary_csv(records, output_dir, filename)
  summary_path = os.path.join(output_dir, filename)
        
  print(f"\n{title_header} complete! Summary saved to: {summary_path}")
  
  summary_lines = []
  summary_lines.append(f"--- {title_header} Results (Sorted by reduced_chi2) ---")
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

def create_fit_evolution_gif(output_dir, gif_name="fit_evolution.gif", duration=500):
  """
  Finds all fit_eval_*.png files in output_dir, sorts them by evaluation index,
  and compiles them into an animated GIF.
  """
  import glob
  from PIL import Image
  
  pattern = os.path.join(output_dir, "fit_eval_*.png")
  png_files = glob.glob(pattern)
  
  if not png_files:
    print("No fit evaluation PNG figures found to create GIF.")
    return
    
  def get_eval_index(filepath):
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    if len(parts) >= 3 and parts[2].isdigit():
      return int(parts[2])
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

def run_automated_fit(args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask):
  import scipy.optimize
  if args.gif:
    args.png = True

  param_names, x0, bounds = parse_fit_arguments(args.fit)
  
  fit_integers = set()
  if args.fit_integer:
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
  
  def objective_function(x):
    eval_start_time = time.time()
    eval_counter[0] += 1
    
    # Map continuous variables to rounded integers where configured
    grid_point = {}
    for name, val in zip(param_names, x):
      if name in fit_integers:
        grid_point[name] = int(np.round(val))
      else:
        grid_point[name] = float(val)
      
    # Bounds penalty
    for name, val, (low, high) in zip(param_names, x, bounds):
      eval_val = int(np.round(val)) if name in fit_integers else val
      if low is not None and eval_val < low:
        eval_duration = time.time() - eval_start_time
        total_elapsed = time.time() - start_total_time
        avg_iter_time = total_elapsed / eval_counter[0]
        remaining = args.max_evals - eval_counter[0]
        eta = avg_iter_time * max(0, remaining)
        print(f"Fit Eval #{eval_counter[0]}/{args.max_evals}: Bound constraint violated ({eval_val} < {low}). Penalty applied | Iter: {eval_duration:.2f}s | Avg: {avg_iter_time:.2f}s | ETA: {format_time(eta)}")
        return 1e9
      if high is not None and eval_val > high:
        eval_duration = time.time() - eval_start_time
        total_elapsed = time.time() - start_total_time
        avg_iter_time = total_elapsed / eval_counter[0]
        remaining = args.max_evals - eval_counter[0]
        eta = avg_iter_time * max(0, remaining)
        print(f"Fit Eval #{eval_counter[0]}/{args.max_evals}: Bound constraint violated ({eval_val} > {high}). Penalty applied | Iter: {eval_duration:.2f}s | Avg: {avg_iter_time:.2f}s | ETA: {format_time(eta)}")
        return 1e9
        
    reduced_chi2, log_residual, record = run_simulation_evaluation(
        grid_point, args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask,
        save_npz=False, label_prefix=f"fit_eval_{eval_counter[0]}"
    )
    
    rec = copy.deepcopy(grid_point)
    rec['eval_index'] = eval_counter[0]
    rec['reduced_chi2'] = reduced_chi2
    rec['log_residual'] = log_residual
    records.append(rec)
    save_summary_csv(records, args.output_dir, "fit_summary.csv")
    
    loss = log_residual if args.loss_function == 'log_residual' else reduced_chi2
    if np.isnan(loss):
      loss = 1e9
      
    eval_duration = time.time() - eval_start_time
    total_elapsed = time.time() - start_total_time
    avg_iter_time = total_elapsed / eval_counter[0]
    remaining = args.max_evals - eval_counter[0]
    eta = avg_iter_time * max(0, remaining)
    
    param_str = ', '.join(f"{k}={v}" if isinstance(v, int) else f"{k}={v:.4f}" for k, v in grid_point.items())
    print(f"Fit Eval #{eval_counter[0]}/{args.max_evals}: {param_str} --> {args.loss_function} = {loss:.4f} | Iter: {eval_duration:.2f}s | Avg: {avg_iter_time:.2f}s | ETA: {format_time(eta)}")
    return loss

  if args.optimizer.lower() == 'differential-evolution':
    integrality = [name in fit_integers for name in param_names]
    # Each generation in DE evaluates popsize * len(param_names) times (default popsize is 15).
    # We scale maxiter so that the total evaluations respect args.max_evals.
    popsize = args.popsize
    de_maxiter = max(1, args.max_evals // (popsize * len(param_names)))
    opt_res = scipy.optimize.differential_evolution(
        objective_function,
        bounds,
        x0=x0,
        maxiter=de_maxiter,
        popsize=popsize,
        integrality=integrality,
        polish=False
    )
  else:
    opt_method = 'nelder-mead' if args.optimizer.lower() == 'nelder-mead' else 'powell'
    opt_options = {'maxiter': args.max_evals}
    if opt_method == 'nelder-mead':
      opt_options['maxfev'] = args.max_evals
      opt_options['xatol'] = args.xatol
      opt_options['fatol'] = args.fatol
    elif opt_method == 'powell':
      opt_options['xtol'] = args.xatol
      opt_options['ftol'] = args.fatol
      
    opt_res = scipy.optimize.minimize(
        objective_function, x0, method=opt_method, options=opt_options
    )
  
  total_runtime = time.time() - start_total_time
  total_evals = max(1, eval_counter[0])
  avg_iter_runtime = total_runtime / total_evals
  
  fit_results_lines = [
      f"Optimizer Success: {opt_res.success}",
      f"Optimizer Message: {opt_res.message}",
      f"Best Loss ({args.loss_function}): {opt_res.fun:.4f}",
      "Optimal Parameters:"
  ]
  best_params = dict(zip(param_names, opt_res.x))
  for k, v in best_params.items():
    if k in fit_integers:
      val_str = str(int(np.round(v)))
    else:
      val_str = f"{v:.4f}"
    fit_results_lines.append(f"  {k} = {val_str}")
    
  fit_results_lines.extend([
      "\n--- Runtime Statistics ---",
      f"Total Runtime: {format_time(total_runtime)} ({total_runtime:.2f}s)",
      f"Average Iteration Runtime: {avg_iter_runtime:.2f}s",
      f"Total Evaluations Completed: {total_evals}"
  ])
  
  extra_summary_text = "\n".join(fit_results_lines)
  
  save_and_print_summary(records, args.output_dir, "fit_summary.csv", "Optimization", extra_summary_text=extra_summary_text)

  if args.gif:
    create_fit_evolution_gif(args.output_dir)

def run_parameter_scan(args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask):
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
    reduced_chi2, log_residual, record = run_simulation_evaluation(
        grid_point, args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask,
        save_npz=True, label_prefix="sim"
    )
    
    iter_duration = time.time() - iter_start_time
    total_elapsed = time.time() - start_total_time
    avg_iter_time = total_elapsed / current_count
    remaining = total_evals - current_count
    eta = avg_iter_time * max(0, remaining)
    
    print(f"Fit results: reduced_chi2={reduced_chi2:.4f}, log_residual={log_residual:.4e} | Iter: {iter_duration:.2f}s | Avg: {avg_iter_time:.2f}s | ETA: {format_time(eta)}")
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
  
  save_and_print_summary(records, args.output_dir, "scan_summary.csv", "Scan", extra_summary_text=extra_summary_text)

def main():
  parser = create_scan_parser()
  args = parse_run_args(parser)
  os.makedirs(args.output_dir, exist_ok=True)
  
  validate_scan_args(args, parser)
  
  hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw = prepare_experimental_data(args)
  
  if args.mask_view:
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
    
  particles, particle_type = load_and_precondition_particles(args)
  
  if args.fit:
    run_automated_fit(args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask)
  else:
    run_parameter_scan(args, particles, particle_type, hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask)

if __name__ == '__main__':
  main()
