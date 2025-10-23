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

def log_plot_2d(hist, y_edges, z_edges, titleText=None, ax=None, intensity_min=1e-9, intensity_max=None, y_range=[-0.55, 0.55], z_range=[-0.5, 0.6], savename='plotQ', match_horisontal_axes=False, output='show'):
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
  ax.set_title(titleText)
  fig = ax.figure

  # plt.gca().invert_xaxis() #optionally invert x-axis?

  if not match_horisontal_axes:
    cbar = fig.colorbar(quadmesh, ax=ax, orientation='vertical')
  else:
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = fig.colorbar(quadmesh, cax=cax)

  # cbar.set_label('Intensity') # Optionally set the colorbar label

  show_or_save(output, savename+'_2D')

def plot_q_1d(values, errors, bin_edges, horisontal_axis_label, color='blue', title_text=None, label='', ax=None, limits=[-0.55, 0.55], savename='plotQ', output='show'):
  import matplotlib.pyplot as plt
  if ax is None:
    _, ax = plt.subplots()

  ax.errorbar(bin_edges, values, yerr=errors, fmt='o-', capsize=5, ecolor='red', color=color, label=label)
  ax.set_xlabel(horisontal_axis_label)
  ax.set_ylabel('Intensity')
  ax.set_title(title_text)
  ax.set_yscale("log")
  ax.set_xlim(limits)

  show_or_save(output, savename+'_qSlice')

def create_tof_sliced_2d_q_plots(x, y, weights, title_base, bins_hor=300, bins_vert=200):
  # tofLimits = [0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075]
  tofLimits = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075]

  for tofRange in [(tofLimits[i],tofLimits[i+1]) for i in range(len(tofLimits)-1)]:
    tof_filter = (tofRange[0]<time) & (time < tofRange[1])
    xtmp = x[tof_filter]
    ytmp = y[tof_filter]
    wtmp = weights[tof_filter]
    # print(time[tofRange[0]<time])
    if(len(xtmp)>0):
      title_text = f"tofMin={tofRange[0]}_tofMax={tofRange[1]}"
      # title_text = f"lambdaMin={calculate_wavelength(tofRange[0]):.2f}_lambdaMax={calculate_wavelength(tofRange[1]):.2f}" #FIXME pathLength is not known for all instruments at this point
      log_plot_2d(xtmp, ytmp, wtmp, bins_hor, bins_vert, title_base+title_text)
  log_plot_2d(x, y, weights, bins_hor, bins_vert, title_base)
  # log_plot_2d(x, y, weights, bins_hor, bins_vert, titleBase+'Full range')

def create_2d_histogram(x, y, weights, y_bins=256, z_bins=128, y_range=[-0.55, 0.55], z_range=[-0.5, 0.6]):
  """Create 2D histogram of weighted y-z values, controlling the ranges and
  number of bins along the axes. Histograms are transposed """
  hist, y_edges, z_edges = np.histogram2d(x, y, weights=weights, bins=[y_bins, z_bins], range=[y_range, z_range])
  hist_weight2, _, _ = np.histogram2d(x, y, weights=weights**2, bins=[y_bins, z_bins], range=[y_range, z_range])
  hist_error = np.sqrt(hist_weight2)
  hist = hist
  hist_error = hist_error
  return hist, hist_error, y_edges, z_edges

def extract_range_to_1d(hist, hist_error, y_edges, z_edges, z_index_range):
  """Extract a range of a 2D histogram into a 1D histogram while handling
  the propagation of error of the corresponding histogram of uncertainties"""
  z_limits = [z_edges[z_index_range[0]], z_edges[z_index_range[1]+1]]
  values_extracted = hist[:,z_index_range[0]:z_index_range[1]]
  values = np.sum(values_extracted, axis=1)
  errors_extracted = hist_error[:,z_index_range[0]:z_index_range[1]]
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