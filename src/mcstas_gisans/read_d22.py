
"""
Read data from measurements at D22(ILL) from nxs files (hard-coded)
"""

import h5py
import numpy as np

from .instrument import Instrument
from .instrument_defaults import instrument_defaults

def read_nexus_data(filepath='073174.nxs', sample_orientation=1, scale_factor=None):
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
  elif filepath.endswith("281836.nxs") or filepath.endswith("silica_100nm_air_D2O_3mNm_3600s.nxs") or filepath.endswith("Direct_beam_60s.nxs") or filepath.endswith("reflected_beam_D2O_794s.nxs") or filepath.endswith("345282.nxs") or filepath.endswith("345959.nxs") :
    alpha_inc_deg = 0.44
    # monitor_data_path= 'entry0/data1/MultiDetector1_data'
  elif filepath.endswith("348110.nxs"): #beam density sim (should be alpha=0.0)
    alpha_inc_deg = 0.44
    # monitor_data_path= 'entry0/data1/MultiDetector1_data'
  elif filepath.endswith("281839.nxs") or filepath.endswith("281841.nxs"): #NP with shell high pressure
    alpha_inc_deg = 0.44
    # scale_factor=3.55921e7 #41722317/0.95109135 #experimentally determined scale factor to match the direct beam intensity (measured with 60s) to the simulated beam density (measured with 3600s) FIXME: this is a hack, should be replaced with proper normalization using monitor counts and measurement times
    # monitor_data_path= 'entry0/data1/MultiDetector1_data'
  else:
    alpha_inc_deg = 0.0

  # Open the NeXus file
  with h5py.File(filepath, 'r') as file:
    try:
        detector_data = file['entry0/D22/Detector 1/data1'][:]
    except:
        detector_data = file['entry0/data1/MultiDetector1_data'][:]
  hist = detector_data[:,:,0]
  if scale_factor is not None:
    hist = hist * scale_factor
  if sample_orientation == 0:
      # Sample rotated +90 deg CCW -> Rotate detector -90 deg CW in sample frame
      hist = np.rot90(hist, -1)
  elif sample_orientation == 2:
      # Sample rotated -90 deg CW -> Rotate detector +90 deg CCW in sample frame
      hist= np.rot90(hist, 1)
  hist_error = np.sqrt(hist)

  instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg, wavelength_selected, sample_orientation)
  q_y, q_z = instrument.get_q_pixel_limits()


  return hist, hist_error, q_y, q_z
