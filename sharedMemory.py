

from multiprocessing import shared_memory, Lock
from importlib import import_module
import numpy as np

# Create a lock for thread-safe updates
update_lock = Lock()

sharedMemoryHistName = 'sharedHistogram'
sharedMemoryHistErrorName = 'sharedHistogramError'
hist_shape = (256, 1, 128) #FIXME shouldn't be hardcoded

sharedMemoryConstantsName = 'sharedConstants'
defaultSampleModel = "models.silica_100nm_air"
sim_module=import_module(defaultSampleModel)

sharedTemplate = np.array([
   0.0,                # nominal_source_sample_distance
   0.0,                # sample_detector_distance
   defaultSampleModel, # sample model,
   0,                  # silica particle radius for 'Silica particles on Silicon measured in air' sample model
   0,                  # pixelNr (number of outgoing beams in x and y direction)
   0.0,                # wavelength selected (for non-TOF instruments)
   0.0,                # alpha_inc (incident angle)
   0.0,                # angle_range (outgoing angle covered by the simulation)
   False               # raw_output (use raw list of q events as output instead of histograms)
   ])

def createSharedMemory(sc):
  '''Add parameters (shared constants) to shared memory for parallel processing simulation'''
  sharedArray = np.array([
     sc['nominal_source_sample_distance'],
     sc['sample_detector_distance'],
     sc['sim_module_name'],
     sc['silicaRadius'],
     sc['pixelNr'],
     sc['wavelengthSelected'],
     sc['alpha_inc'],
     sc['angle_range'],
     sc['raw_output']
    ])

  shm = shared_memory.SharedMemory(create=True, size=sharedTemplate.nbytes, name=sharedMemoryConstantsName)
  mem = np.ndarray(sharedTemplate.shape, dtype=sharedTemplate.dtype, buffer=shm.buf) #Create a NumPy array backed by shared memory
  mem[:] = sharedArray[:] # Copy to the shared memory

  # Create shared memory objects for the histogram and error array
  shm_hist = shared_memory.SharedMemory(create=True, size=np.zeros(hist_shape).nbytes, name=sharedMemoryHistName)
  shm_error = shared_memory.SharedMemory(create=True, size=np.zeros(hist_shape).nbytes, name=sharedMemoryHistErrorName)
  # Create numpy arrays backed by shared memory
  hist_shared = np.ndarray(hist_shape, dtype=np.float64, buffer=shm_hist.buf)
  error_shared = np.ndarray(hist_shape, dtype=np.float64, buffer=shm_error.buf)
  # Initialize with zeros
  hist_shared[:] = 0
  error_shared[:] = 0

  return (shm, shm_hist, shm_error)

def getSharedConstants():
  shm = shared_memory.SharedMemory(name=sharedMemoryConstantsName)
  mem = np.ndarray(sharedTemplate.shape, dtype=sharedTemplate.dtype, buffer=shm.buf)

  sharedValues = {}
  sharedValues.update({'nominal_source_sample_distance' : float(mem[0])})
  sharedValues.update({'sample_detector_distance' : float(mem[1])})
  sharedValues.update({'sim_module_name' : str(mem[2])})
  sharedValues.update({'silicaRadius' : float(mem[3])})
  sharedValues.update({'pixelNr' : int(mem[4])})
  sharedValues.update({'wavelengthSelected' : None if mem[5] == 'None' else float(mem[5])})
  sharedValues.update({'alpha_inc' : float(mem[6])})
  sharedValues.update({'angle_range' : float(mem[7])})
  sharedValues.update({'raw_output' : mem[8].lower() == 'true'})
  shm.close()

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
