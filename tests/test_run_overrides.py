"""
Tests for command line overrides of instrument parameters
"""
import pytest
import copy
from mcstas_gisans.run_cli import create_argparser, parse_args
from mcstas_gisans.instrument_defaults import instrument_defaults, set_instrument_parameters, reset_instrument_defaults
from mcstas_gisans.parameters import pack_parameters

@pytest.fixture
def clean_defaults():
    original_defaults = copy.deepcopy(instrument_defaults)
    yield
    instrument_defaults.clear()
    instrument_defaults.update(original_defaults)

@pytest.mark.parametrize("overrides, expected_params", [
    (
        ["--instrument_sample_detector_distance", "15.5"],
        {"sample_detector_distance": 15.5}
    ),
    (
        ["--instrument_detector_pixels", "512", "256"],
        {"pixels_y_bornagain": 512, "pixels_z_bornagain": 256}
    ),
    (
        ["--instrument_detector_size", "2.048", "2.048"],
        {"size_y_bornagain": 2.048, "size_z_bornagain": 2.048}
    ),
    (
        ["--instrument_detector_centre_offset", "0.1", "-0.2"],
        {"direct_beam_centre_offset_y_bornagain": 0.1, "direct_beam_centre_offset_z_bornagain": -0.2}
    ),
    (
        [
            "--instrument_sample_detector_distance", "10.0",
            "--instrument_detector_pixels", "100", "200",
            "--instrument_detector_size", "1.0", "2.0",
            "--instrument_detector_centre_offset", "0.5", "0.5"
        ],
        {
            "sample_detector_distance": 10.0,
            "pixels_y_bornagain": 100, "pixels_z_bornagain": 200,
            "size_y_bornagain": 1.0, "size_z_bornagain": 2.0,
            "direct_beam_centre_offset_y_bornagain": 0.5, "direct_beam_centre_offset_z_bornagain": 0.5
        }
    )
])
def test_instrument_overrides(clean_defaults, monkeypatch, overrides, expected_params):
    argv = [
        "data/paper/d22_measurement/073174.nxs",
        "-i", "d22",
        "--wavelength_selected", "6.0"
    ] + overrides

    parser = create_argparser()
    monkeypatch.setattr("sys.argv", ["run"] + argv)
    parsed_args = parse_args(parser)

    params = pack_parameters(parsed_args, 'neutron')
    inst = params['instrument']

    for attr, val in expected_params.items():
        if attr == "sample_detector_distance":
            assert inst.sample_detector_distance == val
        elif attr.startswith("pixels"):
            assert getattr(inst.detector, attr) == val
        elif attr.startswith("size"):
            assert getattr(inst.detector, attr) == val
        elif attr.startswith("direct_beam_centre_offset"):
            assert getattr(inst.detector, attr) == val

def test_set_and_reset_instrument_parameters(clean_defaults, monkeypatch):
    parser = create_argparser()
    args = parser.parse_args(["dummy.mcpl.gz", "-i", "d22", "--instrument_sample_detector_distance", "18.2"])
    
    set_instrument_parameters(args)
    assert instrument_defaults['d22']['sample_detector_distance'] == 18.2
    
    reset_instrument_defaults()
    assert instrument_defaults['d22']['sample_detector_distance'] == 17.6

if __name__ == "__main__":
    pytest.main([__file__])
