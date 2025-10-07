
"""
Read data from measurements at D22(ILL) from nxs files (hard-coded)
"""

import h5py
import numpy as np
from numpy import pi, sin, arctan

from .instrument import Instrument
from .instrument_defaults import instrument_defaults

def getStoredData(filepath='073174.nxs'):
  """
  Read data from measurements at D22(ILL) from nxs files.It is hardcoded
  and not meant to be used for any other data in the current state.
  """
  # Constants
  wavelength_selected = 6.0 # Wavelength in angstroms

  if filepath.endswith("073174.nxs"): #silica_100nm_air
    alpha_inc_deg = 0.24
  elif filepath.endswith("73378.nxs"):  #silica_100nm_D2O
    alpha_inc_deg = 0.35 #0.35
  else:
    alpha_inc_deg = 0.0

  # Open the NeXus file
  with h5py.File(filepath, 'r') as file:
    detector_data = file['entry0/D22/Detector 1/data1'][:]
  hist = detector_data[:,:,0]
  hist_error = np.sqrt(hist)

  inst_params = instrument_defaults['d22']

  alpha_inc = float(np.deg2rad(alpha_inc_deg))
  sample_inclination = alpha_inc
  nominal_source_sample_distance = inst_params['nominal_source_sample_distance']
  sample_detector_distance = inst_params['sample_detector_distance']
  instrument = Instrument(inst_params['detector'], sample_detector_distance, sample_inclination, alpha_inc, nominal_source_sample_distance, wavelength_selected)

  q_min, q_max = instrument.calculate_q_limits()

  q_x = np.linspace(q_min[0], q_max[0], num=(detector_data.shape[1]+1))
  q_y = np.linspace(q_min[1], q_max[1], num=(detector_data.shape[0]+1))

  return hist.T, hist_error.T, q_x, q_y
