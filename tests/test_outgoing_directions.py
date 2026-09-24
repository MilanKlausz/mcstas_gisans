import pytest
import sys
from mcstas_gisans.run_cli import create_argparser, parse_args
from mcstas_gisans.parameters import pack_parameters

def test_outgoing_directions_default():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        args = parse_args(parser)
        assert args.outgoing_directions == 20
        params = pack_parameters(args, "neutron")
        assert params["outgoing_directions_horizontal"] == 20
        assert params["outgoing_directions_vertical"] == 20
    finally:
        sys.argv = sys_argv_backup

def test_outgoing_directions_explicit():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions", "45"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        args = parse_args(parser)
        assert args.outgoing_directions == 45
        params = pack_parameters(args, "neutron")
        assert params["outgoing_directions_horizontal"] == 45
        assert params["outgoing_directions_vertical"] == 45
    finally:
        sys.argv = sys_argv_backup

def test_outgoing_directions_asymmetric():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions_horizontal", "50", "--outgoing_directions_vertical", "30"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        args = parse_args(parser)
        assert args.outgoing_directions is None
        assert args.outgoing_directions_horizontal == 50
        assert args.outgoing_directions_vertical == 30
        params = pack_parameters(args, "neutron")
        assert params["outgoing_directions_horizontal"] == 50
        assert params["outgoing_directions_vertical"] == 30
    finally:
        sys.argv = sys_argv_backup

def test_outgoing_directions_validation_mutually_exclusive():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions", "25", "--outgoing_directions_horizontal", "50"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        with pytest.raises(SystemExit):
            parse_args(parser)
    finally:
        sys.argv = sys_argv_backup

def test_outgoing_directions_validation_both_required():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions_horizontal", "50"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        with pytest.raises(SystemExit):
            parse_args(parser)
    finally:
        sys.argv = sys_argv_backup

@pytest.mark.parametrize("rand_y, rand_z", [(0.0, 0.0), (1.0, -1.0), (-0.3, 0.7)])
def test_rays_use_the_directions_bornagain_evaluates(rand_y, rand_z):
  """The ray directions must be the bin centres of the BornAgain detector (with the same shift)."""
  import numpy as np
  import bornagain as ba
  from bornagain import deg
  from mcstas_gisans.run import get_outgoing_grid, get_simulation
  angle_range = [-1.0, 1.5, -0.2, 2.0]
  n_h, n_v = 7, 5
  sim = get_simulation(ba.MultiLayer(), n_h, n_v, angle_range, 6.0, 0.24, 1.0, rand_y, rand_z, None, None, 1.0, 0.5)
  detector = sim.detector()
  (_, phi, alpha) = get_outgoing_grid(angle_range, n_h, n_v, rand_y, rand_z)
  np.testing.assert_allclose(phi, np.array(detector.axis(0).binCenters()) / deg, atol=1e-12)
  np.testing.assert_allclose(alpha[::-1], np.array(detector.axis(1).binCenters()) / deg, atol=1e-12)
  # the shifted grid never leaves the angle range
  assert angle_range[0] <= phi.min() and phi.max() <= angle_range[1]
  assert angle_range[2] <= alpha.min() and alpha.max() <= angle_range[3]
