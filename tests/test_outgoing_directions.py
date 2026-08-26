import pytest
import sys
from mcstas_gisans.run_cli import create_argparser, parse_args
from mcstas_gisans.parameters import pack_parameters

@pytest.fixture
def parser():
    return create_argparser()

@pytest.mark.parametrize("argv,expected_horiz,expected_vert,expected_out", [
    (["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0"], 20, 20, 20),
    (["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions", "45"], 45, 45, 45),
    (["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions", "10"], 10, 10, 10),
    (["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions_horizontal", "50", "--outgoing_directions_vertical", "30"], 50, 30, None),
    (["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions_horizontal", "15", "--outgoing_directions_vertical", "25"], 15, 25, None),
])
def test_outgoing_directions_success(monkeypatch, parser, argv, expected_horiz, expected_vert, expected_out):
    monkeypatch.setattr(sys, 'argv', ["run"] + argv)
    args = parse_args(parser)
    assert args.outgoing_directions == expected_out
    
    if expected_out is None:
        assert args.outgoing_directions_horizontal == expected_horiz
        assert args.outgoing_directions_vertical == expected_vert
        
    params = pack_parameters(args, "neutron")
    assert params["outgoing_directions_horizontal"] == expected_horiz
    assert params["outgoing_directions_vertical"] == expected_vert

@pytest.mark.parametrize("argv", [
    ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions", "25", "--outgoing_directions_horizontal", "50"],
    ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions_horizontal", "50"],
    ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions_vertical", "30"],
])
def test_outgoing_directions_validation_errors(monkeypatch, parser, argv):
    monkeypatch.setattr(sys, 'argv', ["run"] + argv)
    with pytest.raises(SystemExit):
        parse_args(parser)
