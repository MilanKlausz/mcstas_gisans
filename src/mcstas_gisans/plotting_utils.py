"""
Collection of plotting functions
"""

import numpy as np
# from neutron_utilities import calculate_wavelength
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def show_or_save(output, filename_base):
  if output == 'show':
    plt.show()
  elif output != 'none':
    filename = filename_base + output
    plt.savefig(filename, dpi=300)
    print(f"Created {filename}")

def log_plot_2d(hist, y_edges, z_edges, title_text=None, ax=None, intensity_min=1e-9, intensity_max=None, y_range=None, z_range=None, savename='plotQ', match_horisontal_axes=False, output='show', add_colorbar=True):
  if y_range is None: y_range = [-0.55, 0.55]
  if z_range is None: z_range = [-0.5, 0.6]
  if ax is None:
    _, ax = plt.subplots()

  cmap = plt.get_cmap('jet')
  cmap.set_bad('k') # Handle empty bins giving error with LogNorm
  intensity_max = intensity_max if intensity_max is not None else hist.max().max()
  quadmesh = ax.pcolormesh(y_edges, z_edges, hist.T, norm=colors.LogNorm(intensity_min, vmax=intensity_max), cmap=cmap)

  ax.set_xlim(y_range)
  ax.set_ylim(z_range)
  ax.set_xlabel('Qy [1/nm]')
  ax.set_ylabel('Qz [1/nm]')
  ax.set_title(title_text)
  fig = ax.figure

  # plt.gca().invert_xaxis() #optionally invert x-axis?

  if add_colorbar:
    if not match_horisontal_axes:
      cbar = fig.colorbar(quadmesh, ax=ax, orientation='vertical')
    else:
      cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
      cbar = fig.colorbar(quadmesh, cax=cax)

  # cbar.set_label('Intensity') # Optionally set the colorbar label

  show_or_save(output, savename+'_2D')
  return quadmesh

def plot_q_1d(values, errors, bin_edges, horisontal_axis_label, color='blue', title_text=None, label='', ax=None, limits=None, savename='plotQ', output='show'):
  if limits is None: limits = [-0.55, 0.55]
  if ax is None:
    _, ax = plt.subplots()

  ax.errorbar(bin_edges, values, yerr=errors, fmt='o-', capsize=5, ecolor='red', color=color, label=label)
  ax.set_xlabel(horisontal_axis_label)
  ax.set_ylabel('Intensity')
  ax.set_title(title_text)
  ax.set_yscale("log")
  ax.set_xlim(limits)

  show_or_save(output, savename+'_qSlice')

def create_2d_histogram(x, y, weights, y_bins=256, z_bins=128, y_range=None, z_range=None):
  if y_range is None: y_range = [-0.55, 0.55]
  if z_range is None: z_range = [-0.5, 0.6]
  """Create 2D histogram of weighted y-z values, controlling the ranges and
  number of bins along the axes. Histograms are transposed """
  hist, y_edges, z_edges = np.histogram2d(x, y, weights=weights, bins=[y_bins, z_bins], range=[y_range, z_range])
  hist_weight2, _, _ = np.histogram2d(x, y, weights=weights**2, bins=[y_bins, z_bins], range=[y_range, z_range])
  hist_error = np.sqrt(hist_weight2)

  return hist, hist_error, y_edges, z_edges

def extract_range_to_1d(hist, hist_error, y_edges, z_edges, z_index_range):
  """Extract a range of a 2D histogram into a 1D histogram while handling
  the propagation of error of the corresponding histogram of uncertainties"""
  z_idx_0 = min(max(0, z_index_range[0]), len(z_edges) - 2)
  z_idx_1 = min(max(0, z_index_range[1]), len(z_edges) - 2)
  if z_idx_0 > z_idx_1:
      z_idx_0, z_idx_1 = z_idx_1, z_idx_0
  z_limits = [z_edges[z_idx_0], z_edges[z_idx_1+1]]
  values_extracted = hist[:,z_idx_0:z_idx_1+1]
  values = np.sum(values_extracted, axis=1)
  errors_extracted = hist_error[:,z_idx_0:z_idx_1+1]
  errors = np.sqrt(np.sum(errors_extracted**2, axis=1))
  y_bins = (y_edges[:-1] + y_edges[1:]) / 2 # Calculate bin centers from bin edges
  return values, errors, y_bins, z_limits

### TODO in dev ###
def extract_range_to_1d_vertical(hist, hist_error, y_edges, z_edges, y_index_range):
  """Extract a range of a 2D histogram into a 1D histogram while handling
  the propagation of error of the corresponding histogram of uncertainties"""
  y_limits = [y_edges[y_index_range[0]], y_edges[y_index_range[1]+1]]
  values_extracted = hist[y_index_range[0]:y_index_range[1],:]
  values = np.sum(values_extracted, axis=0)
  errors_extracted = hist_error[y_index_range[0]:y_index_range[1],:]
  errors = np.sqrt(np.sum(errors_extracted**2, axis=0))
  z_bins = (z_edges[:-1] + z_edges[1:]) / 2 # Calculate bin centers from bin edges
  return values, errors, z_bins, y_limits