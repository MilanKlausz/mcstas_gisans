"""
Tests for command line overrides of instrument parameters in the plot CLI
"""
import sys
import copy
from mcstas_gisans.plot_cli import create_argparser, parse_args
from mcstas_gisans.instrument_defaults import instrument_defaults

def test_plot_instrument_overrides():
  # We need to pass required args: --nxs and instrument overrides
  argv = [
    "--nxs", "data/paper/d22_measurement/073174.nxs",
    "-i", "d22",
    "--instrument_sample_detector_distance", "15.5",
    "--instrument_detector_pixels", "512", "256",
    "--instrument_detector_size", "2.048", "2.048",
    "--instrument_detector_centre_offset", "0.1", "-0.2",
  ]

  # Save clean copy of defaults
  original_defaults = copy.deepcopy(instrument_defaults['d22'])

  try:
    parser = create_argparser()
    # Temporarily patch sys.argv
    sys_argv_backup = sys.argv
    sys.argv = [sys_argv_backup[0]] + argv
    try:
      parsed_args = parse_args(parser)
    finally:
      sys.argv = sys_argv_backup

    # Check if the overrides mutated the dictionary in instrument_defaults
    current_d22 = instrument_defaults['d22']
    assert current_d22['sample_detector_distance'] == 15.5
    assert current_d22['detector']['pixels'] == [512, 256]
    assert current_d22['detector']['size'] == [2.048, 2.048]
    assert current_d22['detector']['direct_beam_centre_offset'] == [0.1, -0.2]

    # Verify that calling read_nexus_data loads the overriden settings
    from mcstas_gisans.read_d22 import read_nexus_data
    hist, hist_error, q_y, q_z = read_nexus_data("data/paper/d22_measurement/073174.nxs")
    
    print("DEBUG: len(q_y) =", len(q_y), "len(q_z) =", len(q_z))
    assert len(q_y) == 513
    assert len(q_z) == 257

  finally:
    # Restore original defaults
    instrument_defaults['d22'] = original_defaults

if __name__ == "__main__":
  test_plot_instrument_overrides()
  print("All plot instrument overrides tests passed!")
