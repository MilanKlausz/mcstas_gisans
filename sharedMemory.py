

from multiprocessing import shared_memory
from importlib import import_module
import numpy as np

sharedMemoryName = 'sharedProcessMemory'
defaultSampleModel = "models.silica_100nm_air"
sim_module=import_module(defaultSampleModel)

sharedTemplate = np.array([
   0.0,                # nominal_source_sample_distance
   0.0,                # sample_detector_distance
   defaultSampleModel, # sample model,
   0,                  # silica particle radius for 'Silica particles on Silicon measured in air' sample model
   0,                  # bins (~detector resolution)
   0.0                 # wavelength selected (for non-TOF instruments)
   ])

def createSharedMemory(nominal_source_sample_distance, sample_detector_distance, sim_module_name, silicaRadius, bins, wavelengthSelected):
  '''Add parameters to a shared memory for parallel processing simulation'''
  shared = np.array([
     nominal_source_sample_distance,
     sample_detector_distance,
     sim_module_name,
     silicaRadius,
     bins,
     wavelengthSelected
    ])

  shm = shared_memory.SharedMemory(create=True, size=sharedTemplate.nbytes, name=sharedMemoryName)
  mem = np.ndarray(sharedTemplate.shape, dtype=sharedTemplate.dtype, buffer=shm.buf) #Create a NumPy array backed by shared memory
  mem[:] = shared[:] # Copy to the shared memory

  return shm

def getSharedMemoryValues():
  shm = shared_memory.SharedMemory(name=sharedMemoryName)
  mem = np.ndarray(sharedTemplate.shape, dtype=sharedTemplate.dtype, buffer=shm.buf)

  sharedValues = {}
  sharedValues.update({'nominal_source_sample_distance' : float(mem[0])})
  sharedValues.update({'sample_detector_distance' : float(mem[1])})
  sharedValues.update({'sim_module_name' : str(mem[2])})
  sharedValues.update({'silicaRadius' : float(mem[3])})
  sharedValues.update({'bins' : int(mem[4])})
  sharedValues.update({'wavelengthSelected' : None if mem[5] == 'None' else float(mem[5])})
  shm.close()

  return sharedValues
