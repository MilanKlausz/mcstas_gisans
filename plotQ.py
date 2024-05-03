
import numpy as np
from plotting_utilities import plotSingleQ, logPlot2d, create2dHistogram  #, createTofSliced2dQPlots
from experiment_time_utilities import scaleToExperiment, scaleToExperimentWithIteratedTime, checkUpscalingError, checkBinCountCriterion, findExperimentTimeMaximum, findExperimentTimeMinimum

def unpackQEvents(qEvents):
    x = qEvents[:, 1]
    y = qEvents[:, 2]
    z = -qEvents[:, 3] #NOTE inverting z (to point up instead of down)
    weights = qEvents[:, 0]
    # time = np.array(qEvents[:, 4])
    # return x, y, z, weights, time
    return x, y, z, weights


def main(args):
  data = np.load(args.filename)
  for key in data.files:
    q_events = data[key]

    # x, _, z, weights, _ = unpackQEvents(q_events)
    x, _, z, weights = unpackQEvents(q_events)

    import matplotlib.pyplot as plt

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

    experimentTime = args.experiment_time

    x_range=[-0.55, 0.55]
    z_range=[-0.5, 0.6]
    # x_range=[-0.2, 0.2]
    # z_range=[-0.12, 0.22]
    bins_hor=150
    bins_vert=100
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert, x_range=x_range, z_range=z_range)
    qz=args.q_vertical
    qz_index = np.digitize(qz, zedges) - 1

    if args.find_tmin:
      #Find experiment time that would give at least <args.minimum_count_number> hits in
      #more than <args.minimum_count_fraction> fraction of the bins
      #TODO add option to control if the basis should be the full 2D histogram or just the q of interest
      if args.verbose:
        t_max = findExperimentTimeMaximum(hist[qz_index,:], hist_error[qz_index,:])
        print(f"Maximum time the result can be correctly upscaled to: {t_max} sec")

      t_min = findExperimentTimeMinimum(hist[qz_index,:], args.minimum_count_number, args.minimum_count_fraction)
      print(f"Minimum time the result should be upscaled to in order to get {args.minimum_count_number} counts in {args.minimum_count_fraction*100} percent of the bins: {t_min} sec ({t_min/(60*60):.2f} hours). (This is the analytical result, without the effect of a Gaussian noise.)")
      experimentTime = t_min

      if args.iterate:
        initialTime = t_min * 0.5 #TODO experimental value!
        print(f"\nIteratively increasing the experiment time to find one where {args.minimum_count_fraction*100} percent of the bins have more than {args.minimum_count_number} counts after adding Gaussian noise.")

        hist, hist_error, experimentTime = scaleToExperimentWithIteratedTime(hist, hist_error, qz_index, args.minimum_count_number, args.minimum_count_fraction, initialTime, args.maximum_iteration_number, verbose=args.verbose)

        print(f'\nIteratively found experiment time is: {experimentTime:.0f} sec ({experimentTime/(60*60):.2f} hours)')
        checkBinCountCriterion(hist[qz_index,:], args.minimum_count_number, args.minimum_count_fraction, verbose=True)

    if (experimentTime is not None) and not args.iterate:
      print(f'Upscaling to experiment time: {experimentTime:.0f} sec ({experimentTime/(60*60):.2f} hours)')
      checkUpscalingError(hist[qz_index,:], hist_error[qz_index,:], experimentTime)
      hist, hist_error = scaleToExperiment(hist, hist_error, experimentTime)
      checkBinCountCriterion(hist[qz_index,:], args.minimum_count_number, args.minimum_count_fraction, verbose=True)

    #TODO if plotSingleQ
    plotSingleQ(qz, hist, xedges, zedges, hist_error, titleText = key, ax=ax2, x_range=x_range, savename=args.savename, output=plotOutput)

    #TODO if plot2D
    bins_hor=300 #400
    bins_vert=300 #400
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert, x_range=x_range, z_range=z_range)


    if experimentTime is not None:
      hist, _ = scaleToExperiment(hist, hist_error, experimentTime)

    intensityMin = args.intensity_min
    if intensityMin is None:
      intensityMin = 1e-9 if experimentTime is None else 1

    logPlot2d(hist, xedges, zedges, f"{key}", ax=ax1, intensityMin=intensityMin, x_range=x_range, z_range=z_range, savename=args.savename, output=plotOutput)


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
  parser.add_argument('-d', '--dual_plot', default=False, action='store_true', help = 'Create dual plot in a single figure.')
  parser.add_argument('-t', '--experiment_time', default=None, type=int, help = 'Experiment time in seconds to scale the results up to. (e.g. 10800)')
  parser.add_argument('-v', '--verbose', action='store_true', help = 'Verbose output.')

  parser.add_argument('-m', '--intensity_min', default=None, help = 'Intensity minimum for the 2D q plot colorbar.')
  parser.add_argument('-q', '--q_vertical', default=0.12, type=float, help = 'Vertical component of the Q values of interest.')

  #TODO add plot intensity limit

  findTimeParamGroup = parser.add_argument_group('Find experiment time', 'Parameters and options for finding the experiment time to scale up to.')
  findTimeParamGroup.add_argument('--find_tmin', action='store_true', help = 'Find the minimum experiment time the results need to be upscaled to in order to get a certain minimum number of counts in the bins.')
  findTimeParamGroup.add_argument('-i', '--iterate', action='store_true', help = 'Iteratively find the experiment time for which the bin count criterion is fulfilled after adding Gaussian noise.')
  findTimeParamGroup.add_argument('--maximum_iteration_number', type=int, default=50, help = 'Maximum number of iterations.')
  findTimeParamGroup.add_argument('--minimum_count_number', default=36, type=float, help = 'Minimum number of counts expected in the bins.')
  findTimeParamGroup.add_argument('--minimum_count_fraction', type=zeroToOne, default=0.8, help = 'The fraction of bins that are required to fulfill the minimum count number criterion. [0,1]')

      # minimumHitNumber = 36 #50
    # requiredFulfillmentRatio = 0.8

  args = parser.parse_args()

  main(args)
