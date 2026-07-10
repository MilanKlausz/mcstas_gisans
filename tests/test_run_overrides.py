"""
Tests for command line overrides of instrument parameters
"""
import os
import sys
import copy
from mcstas_gisans.run_cli import create_argparser, parse_args
from mcstas_gisans.instrument_defaults import instrument_defaults
from mcstas_gisans.parameters import pack_parameters

def test_instrument_overrides():
  # We need to pass required args: filename and -i/--instrument
  argv = [
    "data/paper/d22_measurement/073174.nxs",
    "-i", "d22",
    "--instrument_sample_detector_distance", "15.5",
    "--instrument_detector_pixels", "512", "256",
    "--instrument_detector_size", "2.048", "2.048",
    "--instrument_detector_centre_offset", "0.1", "-0.2",
    "--wavelength_selected", "6.0"  # required for non-TOF
  ]

  # Save clean copy of defaults
  original_defaults = copy.deepcopy(instrument_defaults['d22'])

  try:
    parser = create_argparser()
    # Note: parse_args inside run_cli calls parser.parse_args() internally (without args argument),
    # which reads sys.argv. To make it read our custom argv, we temporarily patch sys.argv.
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

    # Check if they propagate to pack_parameters and the Instrument object
    params = pack_parameters(parsed_args, 'neutron')
    inst = params['instrument']

    assert inst.sample_detector_distance == 15.5
    assert inst.detector.pixels_x == 512
    assert inst.detector.pixels_y == 256
    assert inst.detector.size_x == 2.048
    assert inst.detector.size_y == 2.048
    assert inst.detector.direct_beam_centre_offset_x == 0.1
    assert inst.detector.direct_beam_centre_offset_y == -0.2

  finally:
    # Restore original defaults
    instrument_defaults['d22'] = original_defaults

if __name__ == "__main__":
  test_instrument_overrides()
  print("All instrument overrides tests passed!")
