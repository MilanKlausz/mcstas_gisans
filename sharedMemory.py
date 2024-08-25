

from multiprocessing import shared_memory, Lock
import numpy as np

class sharedMemoryHandler:
  """ Intended for creating and managing shared memory blocks for parallel
  processing simulation, as a mean to pass information between the main process
  and the parallel processes.

  This class is not indented the be instantiated. All variables are static and
  the methods are class methods. All information is stored in shared memory
  blocks, reachable through the (static) variables holding their hard-coded names.

  The name of the variables holding the shared memory names, and their intended uses:
    constantsMemoryName - pass in constant values from the main process to the parallel processes
    histMemoryName - store the results (qx,qy,qz) of the parallel processes (incremented by each)
    histErrorMemoryName - accumulate weight squares to calculate uncertainty of the results
    histParamsMemoryName - store parameters of the histograms used by this class
  """
  constantsMemoryName = 'constants'
  histMemoryName = 'hist'
  histErrorMemoryName = 'histError'
  histParamsMemoryName = 'histParams'

  sharedConstantsKeyOrder = ['nominal_source_sample_distance', 'sample_detector_distance', 'sim_module_name', 'silicaRadius', 'pixelNr', 'wavelengthSelected', 'alpha_inc', 'angle_range', 'raw_output']
  #we only actually need: range = [histParams['xRange'], histParams['yRange'], histParams['zRange']]
  histParamsKeyOrder = ['bins', 'xRange', 'yRange', 'zRange']
  histParamsParamNumber = [3, 2, 2, 2]

  updateLock = Lock() # A lock for thread-safe updates

  @classmethod
  def createSharedMemoryBlocks(cls, sharedConstants, histParams):
    """Create shared memory blocks.
    It is intended for the main process to initialise all shared memory blocks.
    """
    ## Create a shared memory block for constant values
    shared_memory.ShareableList([sharedConstants[key] for key in cls.sharedConstantsKeyOrder], name=cls.constantsMemoryName)

    ## Create a shared memory block for the histogram parameters
    shared_memory.ShareableList([p for key in cls.histParamsKeyOrder for p in histParams[key]], name=cls.histParamsMemoryName)

    ## Create shared memory blocks for the histogram and error array
    shm_hist = shared_memory.SharedMemory(create=True, size=np.zeros(histParams['bins']).nbytes, name=cls.histMemoryName)
    shm_error = shared_memory.SharedMemory(create=True, size=np.zeros(histParams['bins']).nbytes, name=cls.histErrorMemoryName)
    # Initialize histogram and error arrays with zeros
    hist = np.ndarray(histParams['bins'], dtype=np.float64, buffer=shm_hist.buf)
    error = np.ndarray(histParams['bins'], dtype=np.float64, buffer=shm_error.buf)
    hist[:] = 0
    error[:] = 0

  @classmethod
  def getConstants(cls):
    """Get constant values from the shared memory.
    Intended for the parallel processes to get access to several constant values.
    """
    try:
      shm = shared_memory.ShareableList(name=cls.constantsMemoryName)
      sharedValues = {key:shm[i] for i,key in enumerate(cls.sharedConstantsKeyOrder)}
    finally:
      shm.shm.close()
    return sharedValues

  @classmethod
  def incrementSharedHistograms(cls, qArray, weights):
    """Increment the histogram and histogramError in the shared memory.
    Intended for the parallel processes, as a mean to store their results.
    """
    hist, histWeightsSquared, _, _, _, = cls._create3dHistogram(qArray, weights)
    with cls.updateLock:
      try:
        hist_shared, error_shared, shm_hist, shm_error = cls._getHistograms()
        # Increment the shared arrays
        hist_shared += hist
        error_shared += histWeightsSquared
      finally:
        shm_hist.close()
        shm_error.close()

  @classmethod
  def getFinalHistograms(cls):
    """Get the final histogram and histogramError from the shared memory.
    Intended for the main process after the parallel processing of the neutrons
    is done.
    """
    try:
      hist, error, shm_hist, shm_error = cls._getHistograms()
      # getting persistent copies of the histogram
      final_hist = hist.copy()
      final_error = np.sqrt(error.copy()) # getting sqrt at the end
    finally:
      shm_hist.close()
      shm_error.close()

    #Calling the _create3dHistogram method with dummy data to histogram in order
    #to get the bin edges of the real histogram.
    dummyEvents = np.array([[0.1, 0.1, 0.1]])
    dummyWeights = np.array([1.0])
    _, _, xEdges, yEdges, zEdges = cls._create3dHistogram(dummyEvents, dummyWeights)

    return final_hist, final_error, xEdges, yEdges, zEdges

  @classmethod
  def cleanup(cls):
    """Close and unlink all shared memory blocks.
    Intended for the main process to clean up after the parallel processes are
    done, and the results are retrieved from the shared memory.
    """
    # Cleanup ShareableList objects
    for shareableListName in [cls.histParamsMemoryName, cls.constantsMemoryName]:
      shm = shared_memory.ShareableList(name=shareableListName)
      shm.shm.close()
      shm.shm.unlink()
    # Cleanup other SharedMemory blocks
    for sharedMemoryName in [cls.histMemoryName, cls.histErrorMemoryName]:
      shm = shared_memory.SharedMemory(name=sharedMemoryName)
      shm.close()
      shm.unlink()

  @classmethod
  def _getHistParams(cls):
    """Get the parameters of the results and corresponding uncertainty histograms
    from the shared memory as a dictionary.
    """
    shm = shared_memory.ShareableList(name=cls.histParamsMemoryName)
    paramsFlat = list(shm)
    shm.shm.close()
    params = {key:'' for key in cls.histParamsKeyOrder} #init dict with keys
    indexBegin = 0
    for key, paramNr in zip(cls.histParamsKeyOrder, cls.histParamsParamNumber):
      params[key] = paramsFlat[indexBegin:indexBegin+paramNr]
      indexBegin+=paramNr
    return params

  @classmethod
  def _getHistograms(cls):
    """Get the result and corresponding uncertainty histograms from shared memory.
    Return numpy arrays backed by shared memory to facilitate handling of the
    memory block as a 3d array (e.g. for incrementing), and the shared memory
    objects to enable closing up after.
    """
    shm_hist = shared_memory.SharedMemory(name=cls.histMemoryName)
    shm_error = shared_memory.SharedMemory(name=cls.histErrorMemoryName)
    # Create numpy arrays backed by shared memory (facilitating )
    bins = cls._getHistParams()['bins']
    hist = np.ndarray(bins, dtype=np.float64, buffer=shm_hist.buf)
    error = np.ndarray(bins, dtype=np.float64, buffer=shm_error.buf)
    return  hist, error, shm_hist, shm_error

  @classmethod
  def _create3dHistogram(cls, qEvents, weights):
    """Create 3d histogram from a list of (qx,qy,qy,weight) values.
    The parameters of the histogram are retrieved from the shared memory.
    """
    histParams = cls._getHistParams()
    bins = histParams['bins']
    range = [histParams['xRange'], histParams['yRange'], histParams['zRange']]
    hist, edges = np.histogramdd(qEvents, weights=weights, bins=bins, range=range)
    histWeightsSquared, _ = np.histogramdd(qEvents, weights=weights**2, bins=bins, range=range)
    return hist, histWeightsSquared, edges[0], edges[1], edges[2],
