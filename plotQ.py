
import numpy as np
import matplotlib.pyplot as plt

from plotting_utilities import plotSingleQ, logPlot2d, create2dHistogram  #, createTofSliced2dQPlots
from experiment_time_utilities import scaleToExperiment, handleExperimentTime
from readD22Data import getStoredData, storedDataParameters

def unpackQEvents(qEvents):
    x = qEvents[:, 1]
    y = qEvents[:, 2]
    z = -qEvents[:, 3] #NOTE inverting z (to point up instead of down)
    weights = qEvents[:, 0]
    # time = np.array(qEvents[:, 4])
    # return x, y, z, weights, time
    return x, y, z, weights

def getRangeDefaultOrOverride(default, minOverride, maxOverride):
  return [minOverride if minOverride else default[0],
          maxOverride if maxOverride else default[1]]

def main(args):

  if args.plotStoredData:
    hist, histError, xEdges, zEdges, experimentTime = getStoredData(args.dataId)
    qz = args.q_vertical
    qzIndex = np.digitize(qz, zEdges) - 1
    xRange = getRangeDefaultOrOverride([xEdges[0], xEdges[-1]], args.x_min, args.x_max)
    zRange = getRangeDefaultOrOverride([zEdges[0], zEdges[-1]], args.y_min, args.y_max)
  else:
    with np.load(args.filename) as npFile:
      npFileArrayKey = npFile.files[0]
      q_events = npFile[npFileArrayKey]
      # x, _, z, weights, _ = unpackQEvents(q_events)
      x, _, z, weights = unpackQEvents(q_events)
    xRange = getRangeDefaultOrOverride([-0.55, 0.55], args.x_min, args.x_max)
    zRange = getRangeDefaultOrOverride([-0.5, 0.6], args.y_min, args.y_max)
    bins_hor = 256 #150
    bins_vert = 128 #100
    hist, xEdges, zEdges, histError = create2dHistogram(x, z, weights, bins_hor, bins_vert, xRange=xRange, yRange=zRange)
    qz = args.q_vertical
    qzIndex = np.digitize(qz, zEdges) - 1
    experimentTime = args.experiment_time

    hist, histError = handleExperimentTime(hist, histError, qzIndex, experimentTime, args.find_experiment_time, args.minimum_count_number, args.minimum_count_fraction, args.iterate, args.maximum_iteration_number, args.verbose)

  if args.dual_plot:
    _, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))
    plotOutput = 'none'
  else:
    ax1, ax2 = None, None
    if args.pdf:
      plotOutput = ".pdf"
    elif args.png:
      plotOutput = ".png"
    else:
      plotOutput = 'show'

  #TODO if plotSingleQ
  plotSingleQ(qz, hist, xEdges, zEdges, histError, titleText='', ax=ax2, xRange=xRange, savename=args.savename, output=plotOutput)

  #TODO if plot2D
  different2dPlotResolution = False #TODO use input value
  if(different2dPlotResolution):
    bins_hor=256 #TODO input
    bins_vert=128 #TODO input
    # bins_hor=300 #400
    # bins_vert=300 #400
    plotRangeX = getRangeDefaultOrOverride(xRange, args.x_min, args.x_max)
    plotRangeY = getRangeDefaultOrOverride(zRange, args.y_min, args.y_max)
    hist, xEdges, zEdges, histError = create2dHistogram(x, z, weights, bins_hor, bins_vert, xRange=plotRangeX, yRange=plotRangeY)
    if experimentTime is not None:
      hist, _ = scaleToExperiment(hist, histError, experimentTime)

  intensityMin = args.intensity_min
  if intensityMin is None:
    intensityMin = 1e-9 if experimentTime is None else 1
  logPlot2d(hist, xEdges, zEdges, '', ax=ax1, intensityMin=intensityMin, xRange=xRange, yRange=zRange, savename=args.savename, output=plotOutput)


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
  parser.add_argument('filename', help = 'Input filename.')
  parser.add_argument('-s', '--savename', default='qPlot', required=False, help = 'Output image filename.')
  parser.add_argument('--pdf', action='store_true', help = 'Export figure as pdf.')
  parser.add_argument('--png', action='store_true', help = 'Export figure as png.')
  parser.add_argument('-t', '--experiment_time', default=None, type=int, help = 'Experiment time in seconds to scale the results up to. (e.g. 10800)')
  parser.add_argument('-v', '--verbose', action='store_true', help = 'Verbose output.')

  plotParamGroup = parser.add_argument_group('Control plotting', 'Parameters and options for plotting.')
  plotParamGroup.add_argument('-d', '--dual_plot', default=False, action='store_true', help = 'Create dual plot in a single figure.')
  plotParamGroup.add_argument('-m', '--intensity_min', default=None, help = 'Intensity minimum for the 2D q plot colorbar.')
  plotParamGroup.add_argument('-q', '--q_vertical', default=0.12, type=float, help = 'Vertical component of the Q values of interest.')
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

  storedDataParamGroup = parser.add_argument_group('Stored data', 'Use stored data files for plotting or comparison.')
  storedDataParamGroup.add_argument('--dataId', default=None, choices=list(storedDataParameters.keys()), help = 'Stored data id.')
  storedDataParamGroup.add_argument('--plotStoredData',  action='store_true', help = 'Plot stored data.')

  args = parser.parse_args()

  main(args)
