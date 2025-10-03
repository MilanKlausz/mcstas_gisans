"""
Collection of plotting functions
"""

import numpy as np
import math as m
# from neutron_utilities import calculate_wavelength
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def showOrSave(output, filenameBase):
  if output == 'show':
    plt.show()
  elif output != 'none':
    filename = filenameBase + output
    plt.savefig(filename, dpi=300)
    print(f"Created {filename}")

def logPlot2d(hist, xEdges, yEdges, titleText=None, ax=None, intensityMin=1e-9, intensityMax=None, xRange=[-0.55, 0.55], yRange=[-0.5, 0.6], savename='plotQ', matchXAxes=False, output='show'):
  if ax is None:
    _, ax = plt.subplots()

  cmap = plt.get_cmap('jet')
  cmap.set_bad('k') # Handle empty bins giving error with LogNorm
  intensityMax = intensityMax if intensityMax is not None else hist.max().max()
  quadmesh = ax.pcolormesh(xEdges, yEdges, hist.T, norm=colors.LogNorm(intensityMin, vmax=intensityMax), cmap=cmap)

  ax.set_xlim(xRange)
  ax.set_ylim(yRange)
  ax.set_xlabel('Qx [1/nm]')
  ax.set_ylabel('Qy [1/nm]')
  ax.set_title(titleText)
  fig = ax.figure

  # plt.gca().invert_xaxis() #optionally invert x-axis?

  if not matchXAxes:
    cbar = fig.colorbar(quadmesh, ax=ax, orientation='vertical')
  else:
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = fig.colorbar(quadmesh, cax=cax)

  # cbar.set_label('Intensity') # Optionally set the colorbar label

  showOrSave(output, savename+'_2D')

def plotQ1D(values, errors, xEdges, yLimits, intensityMin=None, color='blue', titleText=None, label='', ax=None, xRange=[-0.55, 0.55], savename='plotQ', output='show'):
  import matplotlib.pyplot as plt
  if ax is None:
    _, ax = plt.subplots()

  ax.errorbar(xEdges, values, yerr=errors, fmt='o-', capsize=5, ecolor='red', color=color, label=label)

  ax.set_xlabel('Qx [1/nm]') # Set the x-axis title
  ax.set_ylabel('Intensity') # Set the y-axis title
  qLimitText = f" Qy=[{yLimits[0]:.4f}1/nm, {yLimits[1]:.4f}1/nm]"
  ax.set_title(titleText+qLimitText)
  # ax.set_ylim(bottom=intensityMin)
  ax.set_yscale("log")
  ax.set_xlim(xRange)

  showOrSave(output, savename+'_qSlice')

def plotQ1D_vert(values, errors, yEdges, xLimits, intensityMin=None, color='blue', titleText=None, label='', ax=None, yRange=[-0.1, 0.3], savename='plotQ', output='show'):
  """TODO in development"""
  import matplotlib.pyplot as plt
  if ax is None:
    _, ax = plt.subplots()

  ax.errorbar(yEdges, values, yerr=errors, fmt='o-', capsize=5, ecolor='red', color=color, label=label)

  ax.set_xlabel('Qy [1/nm]') # Set the x-axis title
  ax.set_ylabel('Intensity') # Set the y-axis title
  qLimitText = f" Qx=[{xLimits[0]:.4f}1/nm, {xLimits[1]:.4f}1/nm]"
  ax.set_title(titleText+qLimitText)
  # ax.set_ylim(bottom=intensityMin)
  ax.set_yscale("log")
  ax.set_xlim(yRange)

  showOrSave(output, savename+'_qSlice')

def createTofSliced2dQPlots(x, y, weights, titleBase, bins_hor=300, bins_vert=200):
  # tofLimits = [0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075]
  tofLimits = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075]

  for tofRange in [(tofLimits[i],tofLimits[i+1]) for i in range(len(tofLimits)-1)]:
    tof_filter = (tofRange[0]<time) & (time < tofRange[1])
    xtmp = x[tof_filter]
    ytmp = y[tof_filter]
    wtmp = weights[tof_filter]
    # print(time[tofRange[0]<time])
    if(len(xtmp)>0):
      titleText = f"tofMin={tofRange[0]}_tofMax={tofRange[1]}"
      # titleText = f"lambdaMin={calculate_wavelength(tofRange[0]):.2f}_lambdaMax={calculate_wavelength(tofRange[1]):.2f}" #FIXME pathLength is not known for all instruments at this point
      logPlot2d(xtmp, ytmp, wtmp, bins_hor, bins_vert, titleBase+titleText)
  logPlot2d(x, y, weights, bins_hor, bins_vert, titleBase)
  # logPlot2d(x, y, weights, bins_hor, bins_vert, titleBase+'Full range')

def create2dHistogram(x, y, weights, xBins=256, yBins=128, xRange=[-0.55, 0.55], yRange=[-0.5, 0.6]):
  """Create 2D histogram of weighted x-y values, controlling the ranges and
  number of bins along the axes. Histograms are transposed """
  hist, xEdges, yEdges = np.histogram2d(x, y, weights=weights, bins=[xBins, yBins], range=[xRange, yRange])
  hist_weight2, _, _ = np.histogram2d(x, y, weights=weights**2, bins=[xBins, yBins], range=[xRange, yRange])
  histError = np.sqrt(hist_weight2)
  hist = hist
  histError = histError
  return hist, histError, xEdges, yEdges

def extractRangeTo1D(hist, histError, xEdges, yEdges, yIndexRange):
  """Extract a range of a 2D histogram into a 1D histogram while handling
  the propagation of error of the corresponding histogram of uncertainties"""
  yLimits = [yEdges[yIndexRange[0]], yEdges[yIndexRange[1]+1]]
  valuesExtracted = hist[:,yIndexRange[0]:yIndexRange[1]]
  values = np.sum(valuesExtracted, axis=1)
  errorsExtracted = histError[:,yIndexRange[0]:yIndexRange[1]]
  errors = np.sqrt(np.sum(errorsExtracted**2, axis=1))
  xBins = (xEdges[:-1] + xEdges[1:]) / 2 # Calculate bin centers from bin edges
  return values, errors, xBins, yLimits

### TODO in dev ###
def extractRangeTo1D_vert(hist, histError, xEdges, yEdges, xIndexRange):
  """Extract a range of a 2D histogram into a 1D histogram while handling
  the propagation of error of the corresponding histogram of uncertainties"""
  xLimits = [xEdges[xIndexRange[0]], xEdges[xIndexRange[1]+1]]
  valuesExtracted = hist[xIndexRange[0]:xIndexRange[1],:]
  values = np.sum(valuesExtracted, axis=0)
  errorsExtracted = histError[xIndexRange[0]:xIndexRange[1],:]
  errors = np.sqrt(np.sum(errorsExtracted**2, axis=0))
  yBins = (yEdges[:-1] + yEdges[1:]) / 2 # Calculate bin centers from bin edges
  return values, errors, yBins, xLimits