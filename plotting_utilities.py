"""
Plotting utilities
"""

import numpy as np
from neutron_utilities import tofToLambda

def logPlot2d(x, z, weights, plot_bins, titleText = None):
  x_range = [-0.55, 0.55]
  z_range = [-0.5, 0.6]
  # x_range = [x.min(), x.max()]
  # z_range = [z.min(), z.max()]

  hist, xedges, zedges = np.histogram2d(x, z, weights=weights, bins=[plot_bins, plot_bins], range=[x_range, z_range])
  hist = hist.T

  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  _, ax = plt.subplots()# fig return value not used
  hist_min = hist.min().min()
  # intensity_min = hist_min if hist_min!=0 else 1e-10
  intensity_min = 1e-9
  quadmesh = ax.pcolormesh(xedges, zedges, hist, norm=colors.LogNorm(intensity_min, vmax=hist.max().max()), cmap='gist_ncar')

  ax.set_xlim(-0.55, 0.55)
  ax.set_ylim(-0.5, 0.6)
  # ax.set_xlim(xedges.min(), xedges.max())
  # ax.set_ylim(zedges.min(), zedges.max())
  ax.set_xlabel('Qx [1/nm]')
  ax.set_ylabel('Qz [1/nm]')
  plt.title(titleText)

  plt.colorbar(quadmesh)
  filename = titleText.replace('.','p')
  plt.savefig(filename+'.png', dpi=300)
  # plt.show()
    
def createQPlot(q_events, titleBase, plot_bins=400):

  x = q_events[:, 1]
  z = -q_events[:, 3] #NOTE inverting z (to point up instead of down)
  weights = q_events[:, 0]
  time = np.array(q_events[:, 4])

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
      logPlot2d(xtmp, ztmp, wtmp, plot_bins, titleBase+titleText)
  logPlot2d(x, z, weights, plot_bins, titleBase+'Full range')