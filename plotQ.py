#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from plotting_utilities import plotQ1D, logPlot2d, create2dHistogram  #, createTofSliced2dQPlots
from experiment_time_utilities import handleExperimentTime
from d22data import getStoredData

def unpackQEvents(qEvents):
  weights = qEvents[:, 0]
  x = qEvents[:, 1]
  y = qEvents[:, 2]
  z = qEvents[:, 3]
  x, y, z = transformCoordSystem(x,y,z)
  return x, y, z, weights

def unpackHistogram(npFile):
  hist = npFile['hist']
  histError = npFile['error']
  xEdges = npFile['xEdges']
  yEdges = npFile['yEdges']
  zEdges = npFile['zEdges']
  xEdges, yEdges, zEdges = transformCoordSystem(xEdges, yEdges, zEdges)
  return hist, histError, xEdges, yEdges, zEdges

def transformCoordSystem(x,y,z):
  """inverting x and z so that the coord system will be xyx:right-front-up instead of left-front-down"""
  return -x, y, -z

def getRangeDefaultOrOverride(default, minOverride, maxOverride):
  return [minOverride if minOverride else default[0],
          maxOverride if maxOverride else default[1]]

def main(args):
  xDataRange = args.x_range
  yDataRange = args.z_range
  datasets = []
  experimentTime = args.experiment_time

  if args.plotStoredData:
    hist, histError, xEdges, zEdges = getStoredData(args.nxs)
    xDataRange = [xEdges[0], xEdges[-1]]
    yDataRange = [zEdges[0], zEdges[-1]]
    label = 'D22 measurement'
    datasets.append((hist, histError, xEdges, zEdges, label))
    # if args.overlay:
    #   hist_exp, histError_exp, xEdges_exp, zEdges_exp = hist, histError, xEdges, zEdges

  if args.filename and args.label:
    for filename, label in zip(args.filename, args.label):
      with np.load(filename) as npFile:
        if 'hist' in npFile.files: #new file with histograms
          hist, histError, xEdges, _, zEdges = unpackHistogram(npFile)
          hist = np.sum(hist, axis=1).T #TODO collapse along y-axis for now but it should be optional
          histError = np.sum(histError, axis=1).T #TODO collapse along y-axis for now but it should be optional
          xDataRange = [xEdges[0], xEdges[-1]]
          yDataRange = [zEdges[0], zEdges[-1]]
        else: #old 'raw data' file with a list of unhistogrammed qEvents
          npFileArrayKey = npFile.files[0]
          q_events = npFile[npFileArrayKey]
          x, _, z, weights = unpackQEvents(q_events)
          bins_hor = args.bins[0] if not args.plotStoredData else len(xEdges)-1 #override bin number to match stored data for better comparison
          bins_vert = args.bins[1] if not args.plotStoredData else  len(zEdges)-1
          hist, histError, xEdges, zEdges = create2dHistogram(x, z, weights, xBins=bins_hor, yBins=bins_vert, xRange=xDataRange, yRange=yDataRange)

      qzIndex = np.digitize(args.q_min, zEdges) - 1

      hist, histError = handleExperimentTime(hist, histError, qzIndex, experimentTime, args.find_experiment_time, args.minimum_count_number, args.minimum_count_fraction, args.iterate, args.maximum_iteration_number, args.verbose)
      datasets.append((hist, histError, xEdges, zEdges, label))

      # #TODO experimental
      # detectionEfficiency = 0.7
      # hist = hist * detectionEfficiency
      # histError = histError * detectionEfficiency
      # #TODO experimental

  if args.dual_plot:
    _, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))
    plotOutput = 'none'
    matchXAxes = True
  else:
    matchXAxes = False
    if args.overlay:
      # _, ax2 = plt.subplots(1, figsize=(6, 6))
      fig, axes = plt.subplots(2, 2, figsize=(16, 12))
      # Remove the bottom-right subplot
      axes[1, 1].remove()

      # Get the GridSpec from the bottom left subplot
      gs = axes[1, 0].get_gridspec()
      # Remove the bottom left subplot
      axes[1, 0].remove()
      # Add a new, larger subplot to cover the entire bottom row
      ax2 = fig.add_subplot(gs[1:, :])

      axes = axes.flatten()
      ax1 = axes[0]
      ax3 = axes[1]
      # ax2 = axes[2]
    else:
      ax1, ax2 = None, None
    if args.pdf:
      plotOutput = ".pdf"
    elif args.png:
      plotOutput = ".png"
    else:
      plotOutput = 'show'

  if args.intensity_min is not None:
    intensityMin = float(args.intensity_min)
  else:
    intensityMin = 1e-9 if experimentTime is None else 1


  #TODO fix xEdges, and move the function to plotting_utilities(?)
  def extractRangeTo1D(hist, histError, xEdges, yEdges, yIndexRange):
    yLimits = [yEdges[yIndexRange[0]], yEdges[yIndexRange[1]+1]]
    valuesExtracted = hist[yIndexRange[0]:yIndexRange[1],:]
    values = np.sum(valuesExtracted, axis=0)
    errorsExtracted = histError[yIndexRange[0]:yIndexRange[1],:]
    errors = np.sqrt(np.sum(errorsExtracted**2, axis=0))
    xBins = (xEdges[:-1] + xEdges[1:]) / 2 # Calculate bin centers from bin edges
    return values, errors, xBins, yLimits

  xPlotRange = getRangeDefaultOrOverride(xDataRange, args.x_min, args.x_max)
  yPlotRange = getRangeDefaultOrOverride(yDataRange, args.y_min, args.y_max)

  if args.overlay:
    lineColors = ['blue', 'green', 'orange','purple']
    plot2DAxesList = [ax1, ax3, ax3, ax3]
    plot1DAxes = ax2
    for datasetIndex, dataset in enumerate(datasets):
      plot2DAxes = plot2DAxesList[datasetIndex]
      lineColor = lineColors[datasetIndex]
      hist, histError, xEdges, zEdges, label = dataset
      logPlot2d(hist, xEdges, zEdges, label, ax=plot2DAxes, intensityMin=intensityMin, xRange=xPlotRange, yRange=yPlotRange, savename=args.savename, output='none')

      qzMinIndex = np.digitize(args.q_min, zEdges) - 1
      qzMaxIndex = np.digitize(args.q_max, zEdges)
      values, errors, xBins, zLimits = extractRangeTo1D(hist, histError, xEdges, zEdges, [qzMinIndex, qzMaxIndex])
      plotQ1D(values, errors, xBins, zLimits, intensityMin=intensityMin, color=lineColor, titleText='', label=label, ax=plot1DAxes, xRange=xPlotRange, savename=args.savename, output='none')
      plot2DAxes.axhline(zEdges[qzMinIndex], color='magenta', linestyle='--', label='q_y = 0')
      plot2DAxes.axhline(zEdges[qzMaxIndex], color='magenta', linestyle='--', label='q_y = 0')

      # ### TEMP manual work
      # xFirstPeakMin = 0.04 #TODO
      # xFirstPeakMax = 0.085 #TODO
      # qFirstPeakMinIndex = np.digitize(xFirstPeakMin, xBins) - 1
      # qFirstPeakMaxIndex = np.digitize(xFirstPeakMax, xBins)
      # plot1DAxes.axvline(xBins[qFirstPeakMinIndex], color='magenta', linestyle='--')
      # plot1DAxes.axvline(xBins[qFirstPeakMaxIndex], color='magenta', linestyle='--')

      # firstPeakSumIntensity = sum(values[qFirstPeakMinIndex:qFirstPeakMaxIndex])
      # print(f"{label} - {qFirstPeakMinIndex=}, {qFirstPeakMaxIndex=}")
      # print(f"{label} - first peak sum intensity: {firstPeakSumIntensity}")
      # ### TEMP manual work

    plot1DAxes.set_ylim(bottom=intensityMin)
    plot1DAxes.grid()
    plot1DAxes.legend()
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
    for hist, histError, xEdges, zEdges, label in datasets:
      logPlot2d(hist, xEdges, zEdges, '', ax=ax1, intensityMin=intensityMin, xRange=xPlotRange, yRange=yPlotRange, savename=args.savename, matchXAxes=matchXAxes, output=plotOutput)

      qzMinIndex_exp = np.digitize(args.q_min, zEdges) - 1
      qzMaxIndex_exp = np.digitize(args.q_max, zEdges)
      values, errors, xBins, zLimits = extractRangeTo1D(hist, histError, xEdges, zEdges, [qzMinIndex_exp, qzMaxIndex_exp])
      plotQ1D(values, errors, xBins, zLimits, intensityMin=intensityMin, color='blue', titleText='', label=label, ax=ax2, xRange=xPlotRange, savename=args.savename, output='none')

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

  # createTofSliced2dQPlots(x, z, weights, key)

if __name__=='__main__':

  def zeroToOne(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

  import argparse
  parser = argparse.ArgumentParser(description = 'Create Q plots from an .npz file containing the derived Q values for each outgoing neutron from the BornAgain simulation.')
  parser.add_argument('-f', '--filename', nargs = '*', help = 'Input filename[s].')
  parser.add_argument('-l', '--label', nargs = '*', help = 'Label for input[s].')
  parser.add_argument('-s', '--savename', default='qPlot', required=False, help = 'Output image filename.')
  parser.add_argument('--pdf', action='store_true', help = 'Export figure as pdf.')
  parser.add_argument('--png', action='store_true', help = 'Export figure as png.')
  parser.add_argument('-t', '--experiment_time', default=None, type=int, help = 'Experiment time in seconds to scale the results up to. (e.g. 10800)')
  parser.add_argument('-v', '--verbose', action='store_true', help = 'Verbose output.')

  plotParamGroup = parser.add_argument_group('Control plotting', 'Parameters and options for plotting.')
  plotParamGroup.add_argument('-d', '--dual_plot', default=False, action='store_true', help = 'Create dual plot in a single figure.')
  plotParamGroup.add_argument('-m', '--intensity_min', default=None, help = 'Intensity minimum for the 2D q plot colorbar.')
  plotParamGroup.add_argument('-q', '--q_min', default=0.09, type=float, help = 'Vertical component of the Q values of interest. Used as the minimum of the range is q_max is provided as well.')
  plotParamGroup.add_argument('--q_max', default=0.10, type=float, help = 'Maximum of the vertical component of the Q range of interest.')
  plotParamGroup.add_argument('--x_min', default=None, type=float, help = 'Plot limit: x minimum.')
  plotParamGroup.add_argument('--x_max', default=None, type=float, help = 'Plot limit: x maximum.')
  plotParamGroup.add_argument('--y_min', default=None, type=float, help = 'Plot limit: y minimum')
  plotParamGroup.add_argument('--y_max', default=None, type=float, help = 'Plot limit: y maximum.')

  findTimeParamGroup = parser.add_argument_group('Find experiment time', 'Parameters and options for finding the experiment time to scale up to.')
  findTimeParamGroup.add_argument('--find_experiment_time', action='store_true', help = 'Find the minimum experiment time the results need to be upscaled to in order to get a certain minimum number of counts in the bins.')
  findTimeParamGroup.add_argument('-i', '--iterate', action='store_true', help = 'Iteratively find the experiment time for which the bin count criterion is fulfilled after adding Gaussian noise.')
  findTimeParamGroup.add_argument('--maximum_iteration_number', type=int, default=50, help = 'Maximum number of iterations.')
  findTimeParamGroup.add_argument('--minimum_count_number', default=36, type=float, help = 'Minimum number of counts expected in the bins.')
  findTimeParamGroup.add_argument('--minimum_count_fraction', type=zeroToOne, default=0.8, help = 'The fraction of bins that are required to fulfill the minimum count number criterion. [0,1]')

  rawFormat = parser.add_argument_group('Raw Q events data', 'Use (old) raw data format with Q event list in the file instead of an already histogrammed data.')
  rawFormat.add_argument('--bins', nargs=2, type=int, default=[256, 128], help='Number of histogram bins in x,z directions.')
  rawFormat.add_argument('--x_range', nargs=2, type=float, default=[-0.55, 0.55], help='Qx range of the histogram. (In horizontal plane left to right)')
  rawFormat.add_argument('--z_range', nargs=2, type=float, default=[-0.5, 0.6], help='Qz range of the histogram. (In vertical plane )bottom to top')

  storedDataParamGroup = parser.add_argument_group('Stored data', 'Use stored data files for plotting or comparison.')
  storedDataParamGroup.add_argument('--nxs', help = 'Full path to the D22 Nexus file.')
  storedDataParamGroup.add_argument('--plotStoredData',  action='store_true', help = 'Plot stored data.')
  storedDataParamGroup.add_argument('--overlay',  action='store_true', help = 'Overlay stored data with simulated data.')

  args = parser.parse_args()

  if args.filename is None and not args.plotStoredData:
    parser.error('No input file provided! This is only allowed when the --plotStoredData option is used.')

  main(args)
