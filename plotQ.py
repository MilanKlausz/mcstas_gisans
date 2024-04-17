import sys
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


def main():
  if len(sys.argv)>1:
    qFilename = sys.argv[1]
  else:
    qFilename = 'q_arrays.npz'

  data = np.load(qFilename)
  for key in data.files:
    q_events = data[key]

    # x, _, z, weights, _ = unpackQEvents(q_events)
    x, _, z, weights = unpackQEvents(q_events)

    import matplotlib.pyplot as plt

    output='show' #'.png'
    if 'noshow' in sys.argv:
      output=None #'.png'
    if 'double' in sys.argv:
      _, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))
      output = 'showAll'
    else:
      ax1, ax2 = None, None
    experimentTime = 3*60*60 #None #sec

    bins_hor=150
    bins_vert=100
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert)
    qz=0.12
    qz_index = np.digitize(qz, zedges) - 1
    minimumHitNumber = 36 #50
    requiredFulfillmentRatio = 0.8
    # t_max = findExperimentTimeMaximum(hist[qz_index,:], hist_error[qz_index,:])
    # t_min = findExperimentTimeMinimum(hist[qz_index,:], minimumHitNumber, requiredFulfillmentRatio)
    # # break
    # experimentTime = t_min
    print('#### Final experiment time: ', experimentTime/(60*60), ' hours')
    scaleToExperiment(hist[qz_index,:], hist_error[qz_index,:], experimentTime)
    x_range=[-0.55, 0.55]
    z_range=[-0.5, 0.6]
    bins_hor=400
    bins_vert=400
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert, x_range=x_range, z_range=z_range)
    if experimentTime is not None:
      hist, _ = scaleToExperiment(hist, hist_error, experimentTime)
    logPlot2d(hist, xedges, zedges, f"{key}_fullRange", ax=ax1, x_range=x_range, z_range=z_range, output=output)

    qz=0.12
    bins_hor=150
    bins_vert=100
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert, x_range=x_range, z_range=z_range)
    if experimentTime is not None:
      hist, hist_error = scaleToExperiment(hist, hist_error, experimentTime)
    plotSingleQ(qz, hist, xedges, zedges, hist_error, titleText = key, ax=ax2, x_range=x_range, output=output)

    # bins_hor=100
    # bins_vert=150
    # hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert)
    # logPlot2d(hist, xedges, zedges, f"{key}_fullRange", ax=ax2, output=output) #TESTING
    # if experimentTime is not None:
    #   hist, _ = scaleToExperiment(hist, hist_error, experimentTime)
    # logPlot2d(hist, xedges, zedges, f"{key}_fullRange", ax=ax1, output=output)



    if output == 'showAll':
      plt.show()

    # createTofSliced2dQPlots(x, z, weights, key)

if __name__=='__main__':
  main()