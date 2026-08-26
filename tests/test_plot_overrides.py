"""
Tests for command line overrides of instrument parameters in the plot CLI
"""
import pytest
import sys
import copy
from mcstas_gisans.plot_cli import create_argparser, parse_args
from mcstas_gisans.instrument_defaults import instrument_defaults
from mcstas_gisans.nexus_reader import read_nexus_data
from mcstas_gisans.instrument import Instrument

@pytest.fixture
def patch_instrument_defaults(monkeypatch):
    """Fixture to ensure instrument_defaults are restored after each test."""
    original = copy.deepcopy(instrument_defaults)
    monkeypatch.setattr("mcstas_gisans.instrument_defaults.instrument_defaults", instrument_defaults)
    yield instrument_defaults
    # restore
    instrument_defaults.clear()
    instrument_defaults.update(original)

@pytest.mark.parametrize("instrument_name, sample_dist, pixels, size, centre_offset", [
    ("d22", "15.5", ["512", "256"], ["2.048", "2.048"], ["0.1", "-0.2"]),
    ("skadi", "10.0", ["1024", "1024"], ["4.0", "4.0"], ["0.0", "0.0"]),
])
def test_plot_instrument_overrides(patch_instrument_defaults, monkeypatch, instrument_name, sample_dist, pixels, size, centre_offset):
    # We need to pass required args: --nxs and instrument overrides
    argv = [
        "scan",
        "--nxs", "data/paper/d22_measurement/073174.nxs",
        "-i", instrument_name,
        "--instrument_sample_detector_distance", sample_dist,
        "--instrument_detector_pixels", *pixels,
        "--instrument_detector_size", *size,
        "--instrument_detector_centre_offset", *centre_offset,
    ]

    monkeypatch.setattr(sys, "argv", argv)
    
    # Trigger the parsing to apply overrides
    _ = parse_args(create_argparser())

    # Check if the overrides mutated the dictionary in instrument_defaults
    current_inst = patch_instrument_defaults[instrument_name]
    assert current_inst['sample_detector_distance'] == float(sample_dist)
    assert current_inst['detector']['pixels'] == [int(p) for p in pixels]
    assert current_inst['detector']['size'] == [float(s) for s in size]
    assert current_inst['detector']['direct_beam_centre_offset'] == [float(c) for c in centre_offset]

    if instrument_name == "d22":
        # Verify that calling read_nexus_data loads the overriden settings
        instrument = Instrument(current_inst, alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=2)
        _, _, q_y, q_z = read_nexus_data("data/paper/d22_measurement/073174.nxs", instrument=instrument)
        
        assert {len(q_y), len(q_z)} == {int(pixels[0]) + 1, int(pixels[1]) + 1}
