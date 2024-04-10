import sys
import numpy as np
from plotting_utilities import plotSingleQ, logPlot2d, create2dHistogram, scaleToExperiment#, createTofSliced2dQPlots

def unpackQEvents(qEvents):
    x = qEvents[:, 1]
    y = qEvents[:, 2]
    z = -qEvents[:, 3] #NOTE inverting z (to point up instead of down)
    weights = qEvents[:, 0]
    time = np.array(qEvents[:, 4])

    return x, y, z, weights, time


def main():
  if len(sys.argv)>1:
    qFilename = sys.argv[1]
  else:
    qFilename = 'q_arrays.npz'

  data = np.load(qFilename)
  for key in data.files:
    q_events = data[key]

    x, _, z, weights, _ = unpackQEvents(q_events)

    import matplotlib.pyplot as plt

    output='show' #'.png'
    if 'noshow' in sys.argv:
      output=None #'.png'
    if 'double' in sys.argv:
      _, (ax1, ax2) = plt.subplots(2, figsize=(6, 12))
      output = 'showAll'
    else:
      ax1, ax2 = None, None

    experimentTime = None #24*60*60 #sec 

    bins_hor=400
    bins_vert=400
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert)
    if experimentTime is not None:
      hist, _ = scaleToExperiment(hist, hist_error, experimentTime)
    logPlot2d(hist, xedges, zedges, f"{key}_fullRange", ax=ax1, output=output)

    qz=0.12
    bins_hor=150
    bins_vert=100
    hist, xedges, zedges, hist_error = create2dHistogram(x, z, weights, bins_hor, bins_vert)
    if experimentTime is not None:
      hist, hist_error = scaleToExperiment(hist, hist_error, experimentTime)
    plotSingleQ(qz, hist, xedges, zedges, hist_error, titleText = key, ax=ax2, output=output)

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