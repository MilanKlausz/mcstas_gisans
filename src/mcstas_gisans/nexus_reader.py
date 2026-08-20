
"""
Read data from measurements at D22(ILL) from nxs files (hard-coded)
"""

import h5py
import numpy as np

from .instrument import Instrument
from .instrument_defaults import instrument_defaults

def read_nexus_data(filepath, instrument, scale_factor=None):
  """
  Read data from measurements at D22(ILL) from nxs files.
  """

  # Open the NeXus file
  with h5py.File(filepath, 'r') as file:
    try:
        detector_data = file['entry0/D22/Detector 1/data1'][:]
    except:
        detector_data = file['entry0/data1/MultiDetector1_data'][:]
  hist = detector_data[:,:,0]
  if scale_factor is not None:
    hist = hist * scale_factor
  hist_error = np.sqrt(hist)

  # Rotate the NeXus detector image to match the BornAgain sample frame
  hist = instrument.detector.coords.rotate_detector_image(hist)
  hist_error = instrument.detector.coords.rotate_detector_image(hist_error)

  q_y, q_z = instrument.get_q_pixel_limits()


  return hist, hist_error, q_y, q_z
