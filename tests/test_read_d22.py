"""
Tests for read_d22 module
"""
import os
import numpy as np
from mcstas_gisans.read_d22 import read_nexus_data


def test_read_nexus_data_scaling():
  filepath = os.path.join("data", "paper", "d22_measurement", "073174.nxs")
  hist, _, q_y, q_z = read_nexus_data(filepath)

  # Scale by a factor of 2.5
  factor = 2.5
  hist_scaled, hist_error_scaled, q_y_scaled, q_z_scaled = read_nexus_data(filepath, scale_factor=factor)

  # Assert scaling is applied correctly
  assert np.allclose(hist_scaled, hist * factor)
  assert np.allclose(hist_error_scaled, np.sqrt(hist * factor))
  assert np.allclose(q_y_scaled, q_y)
  assert np.allclose(q_z_scaled, q_z)

if __name__ == "__main__":
  test_read_nexus_data_scaling()
  print("All scaling tests passed!")
