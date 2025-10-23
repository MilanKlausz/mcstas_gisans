#!/usr/bin/env python3

"""
Main plotting script to create 2D/1D Q plots from simulation results
"""

import numpy as np
import matplotlib.pyplot as plt

from .plotting_utils import plot_q_1d, log_plot_2d, create_2d_histogram, extract_range_to_1d#, extract_range_to_1d_vertical
from .experiment_time import handle_experiment_time
from .input_output import unpack_q_histogram_file, unpack_raw_q_list_file

def get_plot_ranges(datasets, y_plot_range, z_plot_range):
  """Get plot ranges. Return ranges if provided, otherwise find the minimum and
  maximum from the datasets."""
  if not y_plot_range:
    y_edge_min = min([y_edges[0] for _,_,y_edges,_,_ in datasets])
    y_edge_max = max([y_edges[-1] for _,_,y_edges,_,_ in datasets])
    y_plot_range = [y_edge_min, y_edge_max]
  if not z_plot_range:
    z_edge_min = min([z_edges[0] for _,_,_,z_edges,_ in datasets])
    z_edge_max = max([z_edges[-1] for _,_,_,z_edges,_ in datasets])
    z_plot_range = [z_edge_min, z_edge_max]
  min_value = min([hist.min().min() for hist,_,_,_,_ in datasets])
  max_value = max([hist.max().max() for hist,_,_,_,_ in datasets])
  return y_plot_range, z_plot_range, min_value, max_value

def get_overlay_plot_axes(column=2):
  """Get axes for special subplot layout for dataset comparison. Create a
  grid of subplots, replacing the bottom row with a single larger subplot"""
  fig, axes = plt.subplots(2, column, figsize=(16, 12))

  #Replace the bottom row of the grid with a single new subplot
  gs = axes[1, 0].get_gridspec() #Get GridSpec from the bottom left subplot
  for i in range(column): #Remove all bottom subplots
    axes[1, i].remove()
  axes_bottom = fig.add_subplot(gs[1:, :]) #cover the row with a new subplot

  axes = axes.flatten()
  axes_top = [axes[i] for i in range(column)]
  return axes_top, axes_bottom

def get_datasets(args):
  """Prepare the datasets to be plotted. Process input files, and scale to
  experiment time if required"""
  datasets = []
  y_data_range = args.y_range
  z_data_range = args.z_range

  if args.nxs:
    from .read_d22 import read_nexus_data
    hist, hist_error, y_edges, z_edges = read_nexus_data(args.nxs)
    label = 'D22 measurement'
    nxs_sum = np.sum(hist)
    if args.verbose:
      print(f"NXS sum: {nxs_sum}")
    datasets.append((hist, hist_error, y_edges, z_edges, label))
    y_data_range = [y_edges[0], y_edges[-1]]
    z_data_range = [z_edges[0], z_edges[-1]]

  if args.filename:
    labels = args.label if args.label else args.filename #default to filenames
    for filename, label in zip(args.filename, labels):
      with np.load(filename) as npFile:
        if 'hist' in npFile.files: #new file with histograms
          hist, hist_error, _, y_edges, z_edges = unpack_q_histogram_file(npFile)
          hist = np.sum(hist, axis=2)
          hist_error = np.sum(hist_error, axis=2)
          y_data_range = [y_edges[0], y_edges[-1]]
          z_data_range = [z_edges[0], z_edges[-1]]
        else: #old 'raw data' file with a list of unhistogrammed qEvents #FIXME still uses McStas axis labels
          x, y, _, weights = unpack_raw_q_list_file(npFile)
          bins_hor = args.bins[0] if not args.nxs else len(y_edges)-1 #override bin number to match stored data for better comparison
          bins_vert = args.bins[1] if not args.nxs else len(z_edges)-1
          hist, hist_error, y_edges, z_edges = create_2d_histogram(x, y, weights, y_bins=bins_hor, z_bins=bins_vert, y_range=y_data_range, z_range=z_data_range)

      qz_index = np.digitize(args.q_min, z_edges) - 1

      hist, hist_error = handle_experiment_time(hist, hist_error, qz_index, args.experiment_time, args.find_experiment_time, args.minimum_count_number, args.minimum_count_fraction, args.iterate, args.maximum_iteration_number, args.verbose, args.background)

      hist_sum = np.sum(hist)
      if args.verbose:
        print(f"{filename} sum: {hist_sum}")
      if args.normalise_to_nxs:
        hist *= nxs_sum / hist_sum #normalise total intensity of the sim to the nxs data
        hist_error *= nxs_sum / hist_sum
      datasets.append((hist, hist_error, y_edges, z_edges, label))

      if args.csv:
        csv_filename = f"{filename.rsplit('.', 1)[0]}.csv"
        np.savetxt(csv_filename, hist, delimiter=',')
        print(f"Created {csv_filename}")

  return datasets

def main():
  from .plot_cli import create_argparser, parse_args
  parser = create_argparser()
  args = parse_args(parser)

  datasets = get_datasets(args)

  if args.dual_plot:
    _, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))
    plot_output = 'none'
    match_horisontal_axes = True
  else:
    match_horisontal_axes = False
    if args.overlay:
      axes_top, axes_bottom = get_overlay_plot_axes(len(datasets))
    else:
      ax1, ax2 = None, None
    if args.pdf:
      plot_output = ".pdf"
    elif args.png:
      plot_output = ".png"
    else:
      plot_output = 'show'

  if args.intensity_min is not None:
    intensity_min = float(args.intensity_min)
  else:
    is_upscaled = args.experiment_time or args.find_experiment_time
    intensity_min = 1e-9 if not is_upscaled else 1

  if args.overlay:
    y_plot_range, z_plot_range, _ , max_value = get_plot_ranges(datasets, args.y_plot_range, args.z_plot_range)
    line_colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'brown']
    for dataset_index, dataset in enumerate(datasets):
      plot_2d_axes = axes_top[dataset_index]
      line_color = line_colors[dataset_index]
      hist, hist_error, y_edges, z_edges, label = dataset
      common_maximum = max_value if args.individual_colorbars is False else None
      log_plot_2d(hist, y_edges, z_edges, label, ax=plot_2d_axes, intensity_min=intensity_min, intensity_max=common_maximum, y_range=y_plot_range, z_range=z_plot_range, savename=args.savename, output='none')

      ### TODO in dev temp OFF ###
      qz_min_index = np.digitize(args.q_min, z_edges) - 1
      qz_max_index = np.digitize(args.q_max, z_edges)
      values, errors, y_bins, z_limits = extract_range_to_1d(hist, hist_error, y_edges, z_edges, [qz_min_index, qz_max_index])
      title_text = f" Qz=[{z_limits[0]:.4f}1/nm, {z_limits[1]:.4f}1/nm]"
      horisontal_axis_label = 'Qy [1/nm]'
      plot_q_1d(values, errors, y_bins, horisontal_axis_label, color=line_color, title_text=title_text, label=label, ax=axes_bottom, limits=y_plot_range, savename=args.savename, output='none')
      plot_2d_axes.axhline(z_edges[qz_min_index], color='magenta', linestyle='--', label='q_z = 0') #TODO the label seems to be unfinished but unused
      plot_2d_axes.axhline(z_edges[qz_max_index], color='magenta', linestyle='--', label='q_z = 0') #TODO the label seems to be unfinished but unused
      ### TODO in dev temp OFF ###
      # ### TODO in dev ###
      # qy_min_index = np.digitize(args.q_min, y_edges) - 1
      # qy_max_index = np.digitize(args.q_max, y_edges)
      # values, errors, z_bins, y_limits = extract_range_to_1d_vertical(hist, hist_error, y_edges, z_edges, [qy_min_index, qy_max_index])
      # title_text = f"Qy=[{y_limits[0]:.4f}1/nm, {y_limits[1]:.4f}1/nm]"
      # horisontal_axis_label = 'Qz [1/nm]'
      # plotQ1D_vert(values, errors, z_bins, horisontal_axis_label, color=lineColor, title_text='', label=label, ax=axes_bottom, limits=z_plot_range, savename=args.savename, output='none')
      # plot_2d_axes.axvline(y_edges[qy_min_index], color='magenta', linestyle='--', label='q_y = 0') #TODO the label seems to be unfinished but unused
      # plot_2d_axes.axvline(y_edges[qy_max_index], color='magenta', linestyle='--', label='q_y = 0') #TODO the label seems to be unfinished but unused
      # ### TODO in dev ###
  
      # ### TEMP manual work
      # y_first_peak_min = 0.04 #TODO
      # y_first_peak_max = 0.085 #TODO
      # q_first_peak_min_index = np.digitize(y_first_peak_min, y_bins) - 1
      # q_first_peak_max_index = np.digitize(y_first_peak_max, y_bins)
      # axes_bottom.axvline(y_bins[q_first_peak_min_index], color='magenta', linestyle='--')
      # axes_bottom.axvline(y_bins[q_first_peak_max_index], color='magenta', linestyle='--')

      # first_peak_sum_intensity = sum(values[q_first_peak_min_index:q_first_peak_max_index])
      # print(f"{label} - {q_first_peak_min_index=}, {q_first_peak_max_index=}")
      # print(f"{label} - first peak sum intensity: {first_peak_sum_intensity}")
      # ### TEMP manual work

    axes_bottom.set_ylim(bottom=intensity_min)
    axes_bottom.grid()
    axes_bottom.legend()
    plt.tight_layout()
    if not args.pdf and not args.png:
      plt.show()
    else:
      if(args.pdf):
        filename = f"{args.savename}.pdf"
      elif(args.png):
        filename = f"{args.savename}.png"
      plt.savefig(filename, dpi=300)
      print(f"Created {filename}")

  if not args.overlay:
    for hist, hist_error, y_edges, z_edges, label in datasets:
      y_plot_range = args.y_plot_range if args.y_plot_range else [y_edges[0], y_edges[-1]]
      z_plot_range = args.z_plot_range if args.z_plot_range else [z_edges[0], z_edges[-1]]
      log_plot_2d(hist, y_edges, z_edges, '', ax=ax1, intensity_min=intensity_min, y_range=y_plot_range, z_range=z_plot_range, savename=args.savename, match_horisontal_axes=match_horisontal_axes, output=plot_output)

      qz_min_index_exp = np.digitize(args.q_min, z_edges) - 1
      qz_max_index_exp = np.digitize(args.q_max, z_edges)
      values, errors, y_bins, z_limits = extract_range_to_1d(hist, hist_error, y_edges, z_edges, [qz_min_index_exp, qz_max_index_exp])
      title_text = f" Qz=[{z_limits[0]:.4f}1/nm, {z_limits[1]:.4f}1/nm]"
      horisontal_axis_label = 'Qy [1/nm]'
      plot_q_1d(values, errors, y_bins, horisontal_axis_label, color='blue', title_text=title_text, label=label, ax=ax2, limits=y_plot_range, savename=args.savename, output=plot_output)

  if args.dual_plot:
    if not args.pdf and not args.png:
      plt.show()
    else:
      if(args.pdf):
        filename = f"{args.savename}.pdf"
      elif(args.png):
        filename = f"{args.savename}.png"
      plt.savefig(filename, dpi=300)
      print(f"Created {filename}")

if __name__=='__main__':
  main()
