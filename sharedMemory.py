

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
   0.0,                # wavelength selected (for non-TOF instruments)
   0.0,                # alpha_inc (incident angle)
   0.0                 # angle_range (outgoing angle covered by the simulation)
   ])

def createSharedMemory(sc):
  '''Add parameters (shared constants) to shared memory for parallel processing simulation'''
  sharedArray = np.array([
     sc['nominal_source_sample_distance'],
     sc['sample_detector_distance'],
     sc['sim_module_name'],
     sc['silicaRadius'],
     sc['bins'],
     sc['wavelengthSelected'],
     sc['alpha_inc'],
     sc['angle_range']
    ])

  shm = shared_memory.SharedMemory(create=True, size=sharedTemplate.nbytes, name=sharedMemoryName)
  mem = np.ndarray(sharedTemplate.shape, dtype=sharedTemplate.dtype, buffer=shm.buf) #Create a NumPy array backed by shared memory
  mem[:] = sharedArray[:] # Copy to the shared memory

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
  sharedValues.update({'alpha_inc' : float(mem[6])})
  sharedValues.update({'angle_range' : float(mem[7])})
  shm.close()

  return sharedValues
