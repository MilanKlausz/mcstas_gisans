"""
Tests for beam_centre_correction module
"""
import os
import pytest
import numpy as np
from mcstas_gisans.beam_centre_correction import find_required_centre_offset

@pytest.mark.parametrize("filename, expected_x, expected_y", [
    ("073162.nxs", 0.29018731062878, -0.019189374284965064),
    ("073174.nxs", 0.15351910619853737, -0.01977701711532396)
])
def test_find_required_centre_offset_normal(filename, expected_x, expected_y):
    filepath = os.path.join("data", "paper", "d22_measurement", filename)
    assert os.path.exists(filepath), f"Nexus file {filepath} not found"

    offset = find_required_centre_offset(filepath)

    assert isinstance(offset, np.ndarray)
    assert offset.shape == (2,)

    # Check if it matches expected value
    assert np.isclose(offset[0], expected_x)
    assert np.isclose(offset[1], expected_y)

def test_find_required_centre_offset_declination_override():
    filepath = os.path.join("data", "paper", "d22_measurement", "073162.nxs")
    assert os.path.exists(filepath), f"Nexus file {filepath} not found"
    offset = find_required_centre_offset(filepath, beam_angle=0.44)
    assert isinstance(offset, np.ndarray)
    assert offset.shape == (2,)
    assert np.isclose(offset[0], 0.29026488888918295)
    assert np.isclose(offset[1], 0.11608554, atol=1e-5)

def test_find_required_centre_offset_file_not_found():
    filepath = "non_existent_file.nxs"
    with pytest.raises(FileNotFoundError):
        find_required_centre_offset(filepath)
