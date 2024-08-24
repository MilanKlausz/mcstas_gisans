

from multiprocessing import shared_memory, Lock
import numpy as np

# Create a lock for thread-safe updates
update_lock = Lock()

sharedMemoryHistName = 'sharedHistogram'
sharedMemoryHistErrorName = 'sharedHistogramError'
hist_shape = (256, 1, 128) #FIXME shouldn't be hardcoded
sharedMemoryConstantsName = 'sharedConstants'

sharedConstantsKeyOrder = ['nominal_source_sample_distance', 'sample_detector_distance', 'sim_module_name', 'silicaRadius', 'pixelNr', 'wavelengthSelected', 'alpha_inc', 'angle_range', 'raw_output']

def createSharedMemory(sc):
  '''Create shared memory blocks for parallel processing simulation'''
  # Create shared memory block for constant values
  shm_const = shared_memory.ShareableList([sc[key] for key in sharedConstantsKeyOrder], name=sharedMemoryConstantsName)
  # Create shared memory block for the histogram and error array
  shm_hist = shared_memory.SharedMemory(create=True, size=np.zeros(hist_shape).nbytes, name=sharedMemoryHistName)
  shm_error = shared_memory.SharedMemory(create=True, size=np.zeros(hist_shape).nbytes, name=sharedMemoryHistErrorName)
  # Create numpy arrays backed by shared memory to initialize with zeros
  hist_shared = np.ndarray(hist_shape, dtype=np.float64, buffer=shm_hist.buf)
  error_shared = np.ndarray(hist_shape, dtype=np.float64, buffer=shm_error.buf)
  hist_shared[:] = 0
  error_shared[:] = 0
  return (shm_const.shm, shm_hist, shm_error)

def getSharedConstants():
  try:
    shm = shared_memory.ShareableList(name=sharedMemoryConstantsName)
    sharedValues = {key:shm[i] for i,key in enumerate(sharedConstantsKeyOrder)}
  finally:
    shm.shm.close()
  return sharedValues

def incrementSharedHistograms(qArray, weights):
  hist, histError, _, _, _, = create3dHistogram(qArray, weights=weights, xRange=[-0.55, 0.55], yRange=[-1000, 1000], zRange=[-0.6, 0.5]) #FIXME shouldn't rely on default values of the function, especially given that the x and z directions are in the other direction!

  # Increment histograms in shared memory
  with update_lock:
    shm_hist = shared_memory.SharedMemory(name=sharedMemoryHistName)
    shm_error = shared_memory.SharedMemory(name=sharedMemoryHistErrorName)

    # Create numpy arrays backed by shared memory
    hist_shared = np.ndarray(hist_shape, dtype=np.float64, buffer=shm_hist.buf)
    error_shared = np.ndarray(hist_shape, dtype=np.float64, buffer=shm_error.buf)

    # Increment the shared arrays
    hist_shared += hist
    error_shared += histError

    shm_hist.close()
    shm_error.close()

def getFinalHistograms(shm_hist, shm_error):
  final_hist = np.ndarray(hist_shape, dtype=np.float64, buffer=shm_hist.buf)
  final_error = np.sqrt(np.ndarray(hist_shape, dtype=np.float64, buffer=shm_error.buf))

  #TODO Only doing this for the default edge values..but we shouldn't rely on the defaults, especially given that the x and z axis are in the other direction!
  _, _, xEdges, yEdges, zEdges = create3dHistogram(np.array([[0.1, 0.1, 0.1]]), weights=np.array([1.0]), xRange=[-0.55, 0.55], yRange=[-1000, 1000], zRange=[-0.6, 0.5])

  return final_hist.copy(), final_error.copy(), xEdges, yEdges, zEdges #NOTE returning a copy, not the shared memory object

def create3dHistogram(qEvents, weights, xBins=256, yBins=1, zBins=128, xRange=[-0.55, 0.55], yRange=[-1000, 1000], zRange=[-0.5, 0.6]):
  hist, edges = np.histogramdd(qEvents, weights=weights, bins=[xBins, yBins, zBins], range=[xRange, yRange, zRange])
  hist_weight2, _ = np.histogramdd(qEvents, weights=weights**2, bins=[xBins, yBins, zBins], range=[xRange, yRange, zRange])
  histError = np.sqrt(hist_weight2)
  hist = hist
  histError = histError
  return hist, histError, edges[0], edges[1], edges[2],

def mergeHistograms(hist1, hist1error, hist2, hist2error, xEdges=None, yEdges=None):
  mergedHist = hist1 + hist2
  mergedError = np.sqrt(hist1error**2 + hist2error**2)
  if xEdges is None and yEdges is None:
    return mergedHist, mergedError
  else:
    return mergedHist, mergedError, xEdges, yEdges
