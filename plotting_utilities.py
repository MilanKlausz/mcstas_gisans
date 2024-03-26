"""
Plotting utilities
"""

import numpy as np
from neutron_utilities import tofToLambda

def logPlot2d(x, z, weights, bins_hor, bins_vert, titleText = None, ax=None, output='show'):
  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  import matplotlib.colorbar as colorbar

  if ax is None:
    _, ax = plt.subplots()

  x_range = [-0.55, 0.55]
  z_range = [-0.5, 0.6]
  # x_range = [x.min(), x.max()]
  # z_range = [z.min(), z.max()]

  hist, xedges, zedges = np.histogram2d(x, z, weights=weights, bins=[bins_hor, bins_vert], range=[x_range, z_range])
  hist = hist.T

  hist_min = hist.min().min()
  # intensity_min = hist_min if hist_min!=0 else 1e-10
  intensity_min = 1e-9
  quadmesh = ax.pcolormesh(xedges, zedges, hist, norm=colors.LogNorm(intensity_min, vmax=hist.max().max()), cmap='gist_ncar')

  ax.set_xlim(-0.55, 0.55)
  ax.set_ylim(-0.5, 0.6)
  ax.set_xlabel('Qx [1/nm]')
  ax.set_ylabel('Qz [1/nm]')
  ax.set_title(titleText)

  # plt.colorbar(quadmesh)
  fig = ax.figure # Get the Figure object from the Axes object
  # cbar = fig.colorbar(quadmesh, ax=ax) # Use the Figure object to create the colorbar
 
  cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
  cbar =  fig.colorbar(quadmesh, cax=cax)
  # cbar.set_label('Intensity') # Optionally set the colorbar label

  if output == 'show':
    plt.show()
  elif output =='.pdf' or output == '.png':
    filename = titleText.replace('.','p')
    plt.savefig(filename+output, dpi=300)

def plotSingleQ(qz, x, z, weights, bins_hor, bins_vert, titleText = None, ax=None, output='show'):
  import matplotlib.pyplot as plt

  if ax is None:
    _, ax = plt.subplots()

  x_range = [-0.55, 0.55]
  z_range = [-0.5, 0.6]

  hist, xedges, zedges = np.histogram2d(x, z, weights=weights, bins=[bins_hor, bins_vert], range=[x_range, z_range])
  hist_weight2, _, _ = np.histogram2d(x, z, weights=weights**2, bins=[bins_hor, bins_vert], range=[x_range, z_range])
  hist_error = np.sqrt(hist_weight2)
  hist = hist.T
  hist_error = hist_error.T

  qz_index = np.digitize(qz, zedges) - 1
  ax.errorbar(xedges[:-1], hist[qz_index,:] , yerr=hist_error[qz_index, :], fmt='o-', capsize=5, ecolor='red', color='blue')

  ax.set_xlabel('Qx [1/nm]') # Set the x-axis title
  ax.set_ylabel('Intensity') # Set the y-axis title
  qLimitText = f" Qz=[{zedges[qz_index]:.4f}nm, {zedges[qz_index+1]:.4f}nm]"
  ax.set_title(titleText+qLimitText)
  ax.set_yscale("log")
  ax.set_xlim(-0.55, 0.55)

  if output == 'show':
    plt.show()
  elif output =='.pdf' or output == '.png':
    filename = titleText.replace('.','p')
    plt.savefig(filename+output, dpi=300)

def createTofSliced2dQPlots(x, z, weights, titleBase, bins_hor=300, bins_vert=200):
  # tofLimits = [0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075]
  tofLimits = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075]

  for tofRange in [(tofLimits[i],tofLimits[i+1]) for i in range(len(tofLimits)-1)]:
    tof_filter = (tofRange[0]<time) & (time < tofRange[1])
    xtmp = x[tof_filter]
    ztmp = z[tof_filter]
    wtmp = weights[tof_filter]
    # print(time[tofRange[0]<time])
    if(len(xtmp)>0):
      # titleText = f"tofMin={tofRange[0]}_tofMax={tofRange[1]}"
      titleText = f"lambdaMin={tofToLambda(tofRange[0]):.2f}_lambdaMax={tofToLambda(tofRange[1]):.2f}"
      logPlot2d(xtmp, ztmp, wtmp, bins_hor, bins_vert, titleBase+titleText)
  logPlot2d(x, z, weights, bins_hor, bins_vert, titleBase+'Full range')