#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from plotting_utilities import plotQ1D, logPlot2d, create2dHistogram, extractRangeTo1D
from experiment_time_utilities import handleExperimentTime
from d22data import getStoredData
from input_output_utilities import unpackQHistogramFile, unpackRawQListFile

def getPlotRanges(datasets, xPlotRange, yPlotRange):
  """Get plot ranges. Return ranges if provided, otherwise find the minimum and
  maximum from the datasets."""
  if not xPlotRange:
    xEdgeMin = min([xEdges[0] for _,_,xEdges,_,_ in datasets])
    xEdgeMax = max([xEdges[-1] for _,_,xEdges,_,_ in datasets])
    xPlotRange = [xEdgeMin, xEdgeMax]
  if not yPlotRange:
    yEdgeMin = min([yEdges[0] for _,_,_,yEdges,_ in datasets])
    yEdgeMax = max([yEdges[-1] for _,_,_,yEdges,_ in datasets])
    yPlotRange = [yEdgeMin, yEdgeMax]
  return xPlotRange, yPlotRange

def getOverlayPlotAxes(column=2):
  """Get axes for special subplot layout for dataset comparison. Create a
  grid of subplots, replacing the bottom row with a single larger subplot"""
  fig, axes = plt.subplots(2, column, figsize=(16, 12))

  #Replace the bottom row of the grid with a single new subplot
  gs = axes[1, 0].get_gridspec() #Get GridSpec from the bottom left subplot
  for i in range(column): #Remove all bottom subplots
    axes[1, i].remove()
  axesBottom = fig.add_subplot(gs[1:, :]) #cover the row with a new subplot

  axes = axes.flatten()
  axesTop = [axes[i] for i in range(column)]
  return axesTop, axesBottom

def getDatasets(args):
  """Prepare the datasets to be plotted. Process input files, and scale to
  experiment time if required"""
  datasets = []
  xDataRange = args.x_range
  yDataRange = args.y_range

  if args.nxs:
    hist, histError, xEdges, yEdges = getStoredData(args.nxs)
    label = 'D22 measurement'
    datasets.append((hist, histError, xEdges, yEdges, label))
    xDataRange = [xEdges[0], xEdges[-1]]
    yDataRange = [yEdges[0], yEdges[-1]]

  if args.filename:
    labels = args.label if args.label else args.filename #default to filenames
    for filename, label in zip(args.filename, labels):
      with np.load(filename) as npFile:
        if 'hist' in npFile.files: #new file with histograms
          hist, histError, xEdges, yEdges, _ = unpackQHistogramFile(npFile)
          hist = np.sum(hist, axis=2)
          histError = np.sum(histError, axis=2)
          xDataRange = [xEdges[0], xEdges[-1]]
          yDataRange = [yEdges[0], yEdges[-1]]
        else: #old 'raw data' file with a list of unhistogrammed qEvents
          x, y, _, weights = unpackRawQListFile(npFile)
          bins_hor = args.bins[0] if not args.nxs else len(xEdges)-1 #override bin number to match stored data for better comparison
          bins_vert = args.bins[1] if not args.nxs else len(yEdges)-1
          hist, histError, xEdges, yEdges = create2dHistogram(x, y, weights, xBins=bins_hor, yBins=bins_vert, xRange=xDataRange, yRange=yDataRange)

      qyIndex = np.digitize(args.q_min, yEdges) - 1

      hist, histError = handleExperimentTime(hist, histError, qyIndex, args.experiment_time, args.find_experiment_time, args.minimum_count_number, args.minimum_count_fraction, args.iterate, args.maximum_iteration_number, args.verbose)
      datasets.append((hist, histError, xEdges, yEdges, label))

      # #TODO experimental
      # detectionEfficiency = 0.7
      # hist = hist * detectionEfficiency
      # histError = histError * detectionEfficiency
      # #TODO experimental

  return datasets

def main(args):
  datasets = getDatasets(args)

  if args.dual_plot:
    _, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))
    plotOutput = 'none'
    matchXAxes = True
  else:
    matchXAxes = False
    if args.overlay:
      axesTop, axesBottom = getOverlayPlotAxes(len(datasets))
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
    isUpscaled = args.experiment_time or args.find_experiment_time
    intensityMin = 1e-9 if not isUpscaled else 1

  if args.overlay:
    xPlotRange, yPlotRange = getPlotRanges(datasets, args.x_plot_range, args.y_plot_range)
    lineColors = ['blue', 'green', 'orange','purple']
    for datasetIndex, dataset in enumerate(datasets):
      plot2DAxes = axesTop[datasetIndex]
      lineColor = lineColors[datasetIndex]
      hist, histError, xEdges, yEdges, label = dataset
      logPlot2d(hist, xEdges, yEdges, label, ax=plot2DAxes, intensityMin=intensityMin, xRange=xPlotRange, yRange=yPlotRange, savename=args.savename, output='none')

      qyMinIndex = np.digitize(args.q_min, yEdges) - 1
      qyMaxIndex = np.digitize(args.q_max, yEdges)
      values, errors, xBins, yLimits = extractRangeTo1D(hist, histError, xEdges, yEdges, [qyMinIndex, qyMaxIndex])
      plotQ1D(values, errors, xBins, yLimits, intensityMin=intensityMin, color=lineColor, titleText='', label=label, ax=axesBottom, xRange=xPlotRange, savename=args.savename, output='none')
      plot2DAxes.axhline(yEdges[qyMinIndex], color='magenta', linestyle='--', label='q_y = 0')
      plot2DAxes.axhline(yEdges[qyMaxIndex], color='magenta', linestyle='--', label='q_y = 0')

      # ### TEMP manual work
      # xFirstPeakMin = 0.04 #TODO
      # xFirstPeakMax = 0.085 #TODO
      # qFirstPeakMinIndex = np.digitize(xFirstPeakMin, xBins) - 1
      # qFirstPeakMaxIndex = np.digitize(xFirstPeakMax, xBins)
      # axesBottom.axvline(xBins[qFirstPeakMinIndex], color='magenta', linestyle='--')
      # axesBottom.axvline(xBins[qFirstPeakMaxIndex], color='magenta', linestyle='--')

      # firstPeakSumIntensity = sum(values[qFirstPeakMinIndex:qFirstPeakMaxIndex])
      # print(f"{label} - {qFirstPeakMinIndex=}, {qFirstPeakMaxIndex=}")
      # print(f"{label} - first peak sum intensity: {firstPeakSumIntensity}")
      # ### TEMP manual work

    axesBottom.set_ylim(bottom=intensityMin)
    axesBottom.grid()
    axesBottom.legend()
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
    for hist, histError, xEdges, yEdges, label in datasets:
      xPlotRange = args.x_plot_range if args.x_plot_range else [xEdges[0], xEdges[-1]]
      yPlotRange = args.y_plot_range if args.y_plot_range else [yEdges[0], yEdges[-1]]
      logPlot2d(hist, xEdges, yEdges, '', ax=ax1, intensityMin=intensityMin, xRange=xPlotRange, yRange=yPlotRange, savename=args.savename, matchXAxes=matchXAxes, output=plotOutput)

      qyMinIndex_exp = np.digitize(args.q_min, yEdges) - 1
      qyMaxIndex_exp = np.digitize(args.q_max, yEdges)
      values, errors, xBins, yLimits = extractRangeTo1D(hist, histError, xEdges, yEdges, [qyMinIndex_exp, qyMaxIndex_exp])
      plotQ1D(values, errors, xBins, yLimits, intensityMin=intensityMin, color='blue', titleText='', label=label, ax=ax2, xRange=xPlotRange, savename=args.savename, output=plotOutput)

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

  def zeroToOne(x):
    """Argparser type check function for float number in range [0.0, 1.0]"""
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
  parser.add_argument('-t', '--experiment_time', default=None, type=int, help = 'Experiment time in seconds to scale the results up to. (e.g. 10800). Must be a positive integer.')
  parser.add_argument('-v', '--verbose', action='store_true', help = 'Verbose output.')

  plotParamGroup = parser.add_argument_group('Control plotting', 'Parameters and options for plotting.')
  plotParamGroup.add_argument('-d', '--dual_plot', default=False, action='store_true', help = 'Create a dual plot in a single figure.')
  plotParamGroup.add_argument('-m', '--intensity_min', default=None, help = 'Intensity minimum for the 2D q plot colorbar.')
  plotParamGroup.add_argument('-q', '--q_min', default=0.09, type=float, help = 'Vertical component of the Q values of interest. Used as the minimum of the range is q_max is provided as well.')
  plotParamGroup.add_argument('--q_max', default=0.10, type=float, help = 'Maximum of the vertical component of the Q range of interest.')
  plotParamGroup.add_argument('--x_plot_range', nargs=2, type=float, help = 'Plot x range.')
  plotParamGroup.add_argument('--y_plot_range', nargs=2, type=float, help = 'Plot y range.')

  findTimeParamGroup = parser.add_argument_group('Find experiment time', 'Parameters and options for finding the experiment time to scale up to.')
  findTimeParamGroup.add_argument('--find_experiment_time', action='store_true', help = 'Find the minimum experiment time the results need to be upscaled to in order to get a certain minimum number of counts in the bins.')
  findTimeParamGroup.add_argument('-i', '--iterate', action='store_true', help = 'Iteratively find the experiment time for which the bin count criterion is fulfilled after adding Gaussian noise.')
  findTimeParamGroup.add_argument('--maximum_iteration_number', type=int, default=50, help = 'Maximum number of iterations.')
  findTimeParamGroup.add_argument('--minimum_count_number', default=36, type=int, help = 'Minimum number of counts expected in the bins.')
  findTimeParamGroup.add_argument('--minimum_count_fraction', type=zeroToOne, default=0.8, help = 'The fraction of bins that are required to fulfill the minimum count number criterion. [0,1]')

  rawFormat = parser.add_argument_group('Raw Q events data', 'Use (old) raw data format with Q event list in the file instead of an already histogrammed data.')
  rawFormat.add_argument('--bins', nargs=2, type=int, default=[256, 128], help='Number of histogram bins in x,y directions.')
  rawFormat.add_argument('--x_range', nargs=2, type=float, default=[-0.55, 0.55], help='Qx range of the histogram. (In horizontal plane right to left)')
  rawFormat.add_argument('--y_range', nargs=2, type=float, default=[-0.5, 0.6], help='Qy range of the histogram. (In vertical plane bottom to top)')

  storedDataParamGroup = parser.add_argument_group('Stored data', 'Use stored data files for plotting or comparison.')
  storedDataParamGroup.add_argument('--nxs', default=None, help = 'Full path to the D22 Nexus file. (Using automatic D22 measurement label for it.)')
  storedDataParamGroup.add_argument('--overlay', action='store_true', help = 'Overlay stored data with simulated data.') #TODO isn't it more general than that?

  args = parser.parse_args()

  if args.filename is None and args.nxs is None:
    parser.error('No input file provided! This is only allowed when the --nxs option is used.')

  if args.label and len(args.label) != len(args.filename):
    parser.error(f"The number of labels(${len(args.label)}) doesn't agree with the number of files(${len(args.filename)})")

  if (args.experiment_time is not None) and args.experiment_time <= 0:
    parser.error('The --experiment_time must be a positive integer.')

  if args.minimum_count_number < 0:
    parser.error('The --minimum_count_number must be a non-negative integer.')

  if args.iterate and not args.find_experiment_time:
    parser.error('The --iterate option can only be used when --find_experiment_time is also in use.')

  main(args)
