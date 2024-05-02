
import numpy as np
from plotting_utilities import plotSingleQ, logPlot2d, create2dHistogram, scaleToExperiment, testTmax#, createTofSliced2dQPlots
from plotting_utilities import findExperimentTimeMaximum, findExperimentTimeMinimum

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
      plotOutput = 'show'
      if args.pdf:
        plotOutput = ".pdf"
      if args.png:
        plotOutput = ".png"

    experimentTime = args.experiment_time



    bins_hor=150
    bins_vert=100
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert)
    qz=0.12
    qz_index = np.digitize(qz, zedges) - 1
    minimumHitNumber = 36 #50
    requiredFulfillmentRatio = 0.8
    # t_max = findExperimentTimeMaximum(hist[qz_index,:], hist_error[qz_index,:])
    # t_min = findExperimentTimeMinimum(hist[qz_index,:], minimumHitNumber, requiredFulfillmentRatio)
    # experimentTime = t_min

    # #TESTING to find right experiment time
    # experimentTime = 60*60 #sec
    # minimumHitNumber = 50
    # fractionFilled = 0.90
    # #Find experiment time that would give at least <minimumHitNumber> hits in
    # #more than <fractionNotFilled> ratio of the bins
    # bins_hor=150
    # bins_vert=100
    # hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert)
    # qz=0.12
    # qz_index = np.digitize(qz, zedges) - 1
    # # numberOfBins = hist.size
    # for i in range(50):
    #   print('Experiment time (in sec): ', experimentTime)
    #   tmp_hist, tmp_hist_error = scaleToExperiment(hist, hist_error, experimentTime)
    #   # binsWithEnoughtHits = np.where(tmp_hist > minimumHitNumber)
    #   # if len(binsWithEnoughtHits[0])/numberOfBins >fractionFilled:
    #   binsWithEnoughtHits = np.where(tmp_hist[qz_index,:] > minimumHitNumber)
    #   # print("Result enough hit ration: ",len(binsWithEnoughtHits[0]), '/', numberOfBins)
    #   print("    Result enough hit ration: ",len(binsWithEnoughtHits[0]), '/', bins_hor)
    #   if len(binsWithEnoughtHits[0])/bins_hor >fractionFilled:
    #     break
    #   else:
    #     experimentTime *= 2
    if experimentTime is not None:
      print(f'#### Final experiment time: {experimentTime} sec ({experimentTime/(60*60)} hours)')
      scaleToExperiment(hist[qz_index,:], hist_error[qz_index,:], experimentTime)
    ######

    x_range=[-0.55, 0.55]
    z_range=[-0.5, 0.6]
    # x_range=[-0.2, 0.2]
    # z_range=[-0.12, 0.22]
    bins_hor=400
    bins_vert=400
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert, x_range=x_range, z_range=z_range)
    if experimentTime is not None:
      hist, _ = scaleToExperiment(hist, hist_error, experimentTime)
    logPlot2d(hist, xedges, zedges, f"{key}", ax=ax1, x_range=x_range, z_range=z_range, savename=args.savename, output=plotOutput)

    # qz=0.12
    bins_hor=150
    bins_vert=100
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert, x_range=x_range, z_range=z_range)
    if experimentTime is not None:
      hist, hist_error = scaleToExperiment(hist, hist_error, experimentTime)
    plotSingleQ(qz, hist, xedges, zedges, hist_error, titleText = key, ax=ax2, x_range=x_range, savename=args.savename, output=plotOutput)

    # testTmax(qz, x, z, weights, bins_hor, bins_vert, titleText = key, ax=ax2, savename=args.savename, output=output)


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
  import argparse
  parser = argparse.ArgumentParser(description = 'Create Q plots from an .npz file containing the derived Q values for each outgoing neutron from the BornAgain simulation.')
  parser.add_argument('filename', help = 'Input filename.')
  parser.add_argument('-s', '--savename', default='qPlot', required=False, help = 'Output image filename.')
  parser.add_argument('--pdf', action = 'store_true', help = 'Export figure as pdf.')
  parser.add_argument('--png', action = 'store_true', help = 'Export figure as png.')
  parser.add_argument('-d', '--dual_plot', default=False, action='store_true', help = 'Create dual plot in a single figure.')
  parser.add_argument('-e', 'experiment_time', default=None, help = 'Experiment time in seconds to scale the results up to. (e.g. 10800)')

  # parser.add_argument('--find_tmin', default=None, help = '')


  args = parser.parse_args()

  main(args)
