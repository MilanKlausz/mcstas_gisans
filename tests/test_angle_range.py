import pytest
import numpy as np
from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults
from mcstas_gisans.run_cli import create_argparser, parse_args
from mcstas_gisans.parameters import pack_parameters

def test_detector_angle_maximum_returns_four_values():
    d22 = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=2)
    angles = d22.get_detector_angle_maximum()
    assert len(angles) == 4
    horiz_min, horiz_max, vert_min, vert_max = angles
    # Verify bounds are ordered
    assert horiz_min < horiz_max
    assert vert_min < vert_max

def test_angle_range_cli_four_values():
    parser = create_argparser()
    args = parser.parse_args(['dummy.mcpl.gz', '-i', 'd22', '--wavelength_selected', '6.0', '--angle_range', '-2.5', '3.5', '-1.0', '4.0'])
    assert args.angle_range == [-2.5, 3.5, -1.0, 4.0]
    
    d22 = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=2)
    params = pack_parameters(args, d22)
    assert params['angle_range'] == [-2.5, 3.5, -1.0, 4.0]

def test_verbose_prints_angle_range(capsys):
    parser = create_argparser()
    args = parser.parse_args(['dummy.mcpl.gz', '-i', 'd22', '--wavelength_selected', '6.0', '-v'])
    d22 = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=2)
    params = pack_parameters(args, d22)
    captured = capsys.readouterr()
    assert "Simulated angle range [deg]:" in captured.out

def test_get_masked_angle_range():
    d22 = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=2)
    full_angles = d22.get_detector_angle_maximum()
    
    # Mask half of the detector horizontally
    mask = np.ones((256, 128), dtype=bool)
    mask[:128, :] = False  # exclude left half
    
    mask_angles = d22.get_masked_angle_range(mask, len_y_centres=256)
    assert len(mask_angles) == 4
    # Horizontal min should be tighter than full detector min
    assert mask_angles[0] > full_angles[0]
    assert np.isclose(mask_angles[1], full_angles[1], atol=1e-3)

