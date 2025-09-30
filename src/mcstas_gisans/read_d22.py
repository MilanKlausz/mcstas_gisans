
"""
Read data from measurements at D22(ILL) from nxs files (hard-coded)
"""

import h5py
import numpy as np
from numpy import pi, sin, arctan

def getStoredData(filepath='073174.nxs'):
  """Read data from measurements at D22(ILL) from nxs files.It is hardcoded
  and not meant to be used for any other data in the current state."""
  # Constants
  lambda_angstrom = 6  # Wavelength in angstroms
  detector_distance_mm = 17600  # Distance from sample to detector in mm
  pixel_size_x_mm = 4  # Pixel size in x-direction in mm
  pixel_size_y_mm = 8  # Pixel size in y-direction in mm

  # Open the NeXus file
  with h5py.File(filepath, 'r') as file:
    detector_data = file['entry0/D22/Detector 1/data1'][:]
  hist = detector_data[:,:,0]
  histError = np.sqrt(hist)

  # Compute q-values for x and y axes
  x_pixels = np.arange(0, detector_data.shape[1] + 1) # Start from 0
  y_pixels = np.arange(0, detector_data.shape[0] + 1) # Start from 0

  # Convert pixel index to mm
  x_mm = x_pixels * pixel_size_x_mm
  y_mm = y_pixels * pixel_size_y_mm

# Apply offset to centre the direct beam (based on 073162.nxs)
  x_mm = x_mm - x_mm.max()/2 - (pixel_size_x_mm * 3)
  # y_mm = y_mm - y_mm.max()/2 + (pixel_size_y_mm * 37) #no gravity
  y_mm = y_mm - y_mm.max()/2 + (pixel_size_y_mm * 36) #with gravity

  # Convert mm to q
  theta_x = arctan(x_mm / detector_distance_mm)/2
  theta_y = arctan(y_mm / detector_distance_mm)/2
  q_x = (4 * pi / lambda_angstrom) * sin(theta_x)
  q_y = (4 * pi / lambda_angstrom) * sin(theta_y)

  q_x *= 10 #from 1/A to 1/nm
  q_y *= 10 #from 1/A to 1/nm

  return hist.T, histError.T, q_x, q_y
