#!/usr/bin/env python3

"""
Main plotting script to create 2D/1D Q plots from simulation results
"""

import numpy as np
import matplotlib.pyplot as plt

from .plotting_utils import plotQ1D, logPlot2d, create2dHistogram, extractRangeTo1D, plotQ1D_vert, extractRangeTo1D_vert
from .experiment_time import handleExperimentTime
from .input_output import unpackQHistogramFile, unpackRawQListFile

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
  minValue = min([hist.min().min() for hist,_,_,_,_ in datasets])
  maxValue = max([hist.max().max() for hist,_,_,_,_ in datasets])
  return xPlotRange, yPlotRange, minValue, maxValue

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
    from .read_d22 import getStoredData
    hist, histError, xEdges, yEdges = getStoredData(args.nxs)
    label = 'D22 measurement'
    nxs_sum = np.sum(hist)
    if args.verbose:
      print(f"NXS sum: {nxs_sum}")
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

      hist, histError = handleExperimentTime(hist, histError, qyIndex, args.experiment_time, args.find_experiment_time, args.minimum_count_number, args.minimum_count_fraction, args.iterate, args.maximum_iteration_number, args.verbose, args.background)

      hist_sum = np.sum(hist)
      if args.verbose:
        print(f"{filename} sum: {hist_sum}")
      if args.normalise_to_nxs:
        hist *= nxs_sum / hist_sum #normalise total intensity of the sim to the nxs data
        histError *= nxs_sum / hist_sum
      datasets.append((hist, histError, xEdges, yEdges, label))

      if args.csv:
        csvFilename = f"{filename.rsplit('.', 1)[0]}.csv"
        np.savetxt(csvFilename, hist, delimiter=',')
        print(f"Created {csvFilename}")

  return datasets

def main():
  from .plot_cli import create_argparser, parse_args
  parser = create_argparser()
  args = parse_args(parser)

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
    xPlotRange, yPlotRange, _ , maxValue = getPlotRanges(datasets, args.x_plot_range, args.y_plot_range)
    lineColors = ['blue', 'green', 'orange', 'purple', 'cyan', 'brown']
    for datasetIndex, dataset in enumerate(datasets):
      plot2DAxes = axesTop[datasetIndex]
      lineColor = lineColors[datasetIndex]
      hist, histError, xEdges, yEdges, label = dataset
      commonMaximum = maxValue if args.individual_colorbars is False else None
      logPlot2d(hist, xEdges, yEdges, label, ax=plot2DAxes, intensityMin=intensityMin, intensityMax=commonMaximum, xRange=xPlotRange, yRange=yPlotRange, savename=args.savename, output='none')

      ### TODO in dev temp OFF ###
      qyMinIndex = np.digitize(args.q_min, yEdges) - 1
      qyMaxIndex = np.digitize(args.q_max, yEdges)
      values, errors, xBins, yLimits = extractRangeTo1D(hist, histError, xEdges, yEdges, [qyMinIndex, qyMaxIndex])
      plotQ1D(values, errors, xBins, yLimits, intensityMin=intensityMin, color=lineColor, titleText='', label=label, ax=axesBottom, xRange=xPlotRange, savename=args.savename, output='none')
      plot2DAxes.axhline(yEdges[qyMinIndex], color='magenta', linestyle='--', label='q_y = 0') #TODO the label seems to be unfinished
      plot2DAxes.axhline(yEdges[qyMaxIndex], color='magenta', linestyle='--', label='q_y = 0') #TODO the label seems to be unfinished
      ### TODO in dev temp OFF ###
      # ### TODO in dev ###
      # qxMinIndex = np.digitize(args.q_min, xEdges) - 1
      # qxMaxIndex = np.digitize(args.q_max, xEdges)
      # values, errors, yBins, xLimits = extractRangeTo1D_vert(hist, histError, xEdges, yEdges, [qxMinIndex, qxMaxIndex])
      # plotQ1D_vert(values, errors, yBins, xLimits, intensityMin=intensityMin, color=lineColor, titleText='', label=label, ax=axesBottom, yRange=yPlotRange, savename=args.savename, output='none')
      # plot2DAxes.axvline(xEdges[qxMinIndex], color='magenta', linestyle='--', label='q_x = 0')
      # plot2DAxes.axvline(xEdges[qxMaxIndex], color='magenta', linestyle='--', label='q_x = 0')
      # ### TODO in dev ###
  
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
  main()
