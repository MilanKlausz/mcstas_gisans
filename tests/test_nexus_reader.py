"""
Tests for nexus_reader module
"""
import os
import numpy as np
import pytest
from mcstas_gisans.nexus_reader import read_nexus_data
from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults

def test_read_nexus_data_scaling():
  filepath = os.path.join("data", "paper", "d22_measurement", "073174.nxs")
  instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=1)
  hist, _, q_y, q_z = read_nexus_data(filepath, instrument)
 
  # Scale by a factor of 2.5
  factor = 2.5
  hist_scaled, hist_error_scaled, q_y_scaled, q_z_scaled = read_nexus_data(filepath, instrument, scale_factor=factor)

  # Assert scaling is applied correctly
  assert np.allclose(hist_scaled, hist * factor)
  assert np.allclose(hist_error_scaled, np.sqrt(hist * factor))
  assert np.allclose(q_y_scaled, q_y)
  assert np.allclose(q_z_scaled, q_z)

def test_read_nexus_data_file_not_found():
  filepath = "non_existent_file.nxs"
  instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=1)
  with pytest.raises(FileNotFoundError):
    read_nexus_data(filepath, instrument)

