"""
Tests for beam_centre_correction module
"""
import os
import numpy as np
from mcstas_gisans.beam_centre_correction import find_required_centre_offset

def test_find_required_centre_offset_073162():
  filepath = os.path.join("data", "paper", "d22_measurement", "073162.nxs")
  assert os.path.exists(filepath), f"Nexus file {filepath} not found"

  offset = find_required_centre_offset(filepath)

  assert isinstance(offset, np.ndarray)
  assert offset.shape == (2,)

  # Check if it matches expected value
  assert np.isclose(offset[0], 0.29018731062878)
  assert np.isclose(offset[1], -0.01568863004422238)

def test_find_required_centre_offset_073174():
  filepath = os.path.join("data", "paper", "d22_measurement", "073174.nxs")
  assert os.path.exists(filepath), f"Nexus file {filepath} not found"

  offset = find_required_centre_offset(filepath)

  assert isinstance(offset, np.ndarray)
  assert offset.shape == (2,)

  # Check if it matches expected value
  assert np.isclose(offset[0], 0.15351910619853737)
  assert np.isclose(offset[1], -0.01634266155600934)

if __name__ == "__main__":
  test_find_required_centre_offset_073162()
  test_find_required_centre_offset_073174()
  print("All beam_centre_correction tests passed!")
