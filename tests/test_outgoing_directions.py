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
        # default: the 'quick' sampling preset, the grid is chosen from the particles
        assert args.sampling == 'quick' and args.rays_per_pixel == 250
        assert args.outgoing_directions is None and args.outgoing_directions_horizontal is None
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
  empty_sample = ba.MultiLayer() if hasattr(ba, "MultiLayer") else ba.Sample()  # ba.Sample from BornAgain 22 on
  sim = get_simulation(empty_sample, n_h, n_v, angle_range, 6.0, 0.24, 1.0, rand_y, rand_z, None, None, 1.0, 0.5)
  detector = sim.detector()
  (_, phi, alpha) = get_outgoing_grid(angle_range, n_h, n_v, rand_y, rand_z)
  np.testing.assert_allclose(phi, np.array(detector.axis(0).binCenters()) / deg, atol=1e-12)
  np.testing.assert_allclose(alpha[::-1], np.array(detector.axis(1).binCenters()) / deg, atol=1e-12)
  # the shifted grid never leaves the angle range
  assert angle_range[0] <= phi.min() and phi.max() <= angle_range[1]
  assert angle_range[2] <= alpha.min() and alpha.max() <= angle_range[3]


def _d22_instrument(sample_orientation):
  from mcstas_gisans.instrument import Instrument
  from mcstas_gisans.instrument_defaults import instrument_defaults
  return Instrument(instrument_defaults['d22'], 0.24, 6.0, sample_orientation=sample_orientation)

def test_outgoing_directions_for_sampling_follow_the_detector_pixels():
  """D22 at 17.6 m: 8 mm tubes = 0.0260436 deg, 4 mm pixels = 0.0130218 deg. For a vertical sample the tubes lie
  along the sample normal (BornAgain vertical), for a horizontal sample across it. The directions per pixel along
  each axis are sqrt(R / n_hit), the same on both axes."""
  from mcstas_gisans.parameters import outgoing_directions_for_sampling
  # vertical sample, paper fit mask range: pixels 2.4061/0.0130218 = 184.78 x 0.1719/0.0260436 = 6.600;
  # sqrt(2000/16963) = 0.34337 -> ceil(63.45) x ceil(2.266) = 64 x 3
  paper = _d22_instrument(2)
  angle_range = [-1.2122, 1.1939, 0.1547, 0.3266]
  assert outgoing_directions_for_sampling(angle_range, paper.detector, 17.6, 16963, 2000) == (64, 3)
  # horizontal sample, NP fit mask range: pixels 2.6249/0.0260436 = 100.79 x 1.0468/0.0130218 = 80.39;
  # sqrt(2000/16963) -> ceil(34.61) x ceil(27.60) = 35 x 28; sqrt(250/100000) = 0.05 -> ceil(5.04) x ceil(4.02) = 6 x 5
  liquid = _d22_instrument(1)
  angle_range = [-1.3030, 1.3219, 0.2256, 1.2724]
  assert outgoing_directions_for_sampling(angle_range, liquid.detector, 17.6, 16963, 2000) == (35, 28)
  assert outgoing_directions_for_sampling(angle_range, liquid.detector, 17.6, 100000, 250) == (6, 5)
  # at least one direction per axis
  assert outgoing_directions_for_sampling(angle_range, liquid.detector, 17.6, 10**9, 250) == (1, 1)
  # no neutron hitting the sample is treated as one: sqrt(2000) = 44.72 -> ceil(4507.4) x ceil(3595.1)
  assert outgoing_directions_for_sampling(angle_range, liquid.detector, 17.6, 0, 2000) == (4508, 3596)

D22_RUN_ARGV = ["mg_run", "dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--alpha", "0.24",
                "--sample_orientation", "2", "--angle_range", "-1.2122", "1.1939", "0.1547", "0.3266"]

def _particles(n_hit, n_miss):
  """Preconditioned particles (p, x, y, z, vx, vy, vz, wavelength, t): n_hit on the default 6 cm x 8 cm sample
  and n_miss next to it."""
  import numpy as np
  rows = [[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 400.0, 6.0, 0.0]] * n_hit + [[1.0, 0.05, 0.0, 0.0, 0.0, -1.0, 400.0, 6.0, 0.0]] * n_miss
  return np.array(rows).reshape(-1, 9)

@pytest.mark.parametrize("option, rays_per_pixel", [(["--sampling", "quick"], 250), (["--sampling", "standard"], 2000),
                                                    (["--sampling", "long"], 10000), (["--rays_per_pixel", "2000"], 2000)])
def test_sampling_cli_mg_run(monkeypatch, option, rays_per_pixel):
  from mcstas_gisans.parameters import set_outgoing_directions_from_sampling, outgoing_directions_for_sampling
  monkeypatch.setattr(sys, "argv", D22_RUN_ARGV + option)
  args = parse_args(create_argparser())
  assert args.rays_per_pixel == rays_per_pixel
  assert args.outgoing_directions is None
  with pytest.raises(ValueError):
    pack_parameters(args, "neutron")  # the grid needs the particles first
  set_outgoing_directions_from_sampling(args, _particles(16963, 500), "neutron")  # the missing ones are not counted
  params = pack_parameters(args, "neutron")
  expected = outgoing_directions_for_sampling(args.angle_range, params['instrument'].detector, 17.6, 16963, rays_per_pixel)
  assert (params['outgoing_directions_horizontal'], params['outgoing_directions_vertical']) == expected
  if rays_per_pixel == 2000:
    assert expected == (64, 3)

def test_sampling_cli_mg_fit(monkeypatch):
  """mg_fit uses the run parser; the grid follows the (mask) angle range set into args.angle_range."""
  from mcstas_gisans.fit import create_fit_parser
  from mcstas_gisans.parameters import set_outgoing_directions_from_sampling
  monkeypatch.setattr(sys, "argv", D22_RUN_ARGV + ["--sampling", "standard", "--nxs", "dummy.nxs", "--scan", "radius", "51"])
  args = parse_args(create_fit_parser())
  assert args.sampling == 'standard' and args.rays_per_pixel == 2000
  set_outgoing_directions_from_sampling(args, _particles(16963, 0), "neutron")
  assert (args.outgoing_directions_horizontal, args.outgoing_directions_vertical) == (64, 3)
  params = pack_parameters(args, "neutron")  # every fit evaluation uses the same grid
  assert (params['outgoing_directions_horizontal'], params['outgoing_directions_vertical']) == (64, 3)

@pytest.mark.parametrize("extra", [["--sampling", "quick", "--rays_per_pixel", "100"],
                                   ["--sampling", "quick", "-n", "30"],
                                   ["--rays_per_pixel", "100", "--outgoing_directions", "30"],
                                   ["--sampling", "long", "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "10"],
                                   ["--rays_per_pixel", "100", "--outgoing_directions_vertical", "10"],
                                   ["--rays_per_pixel", "0"],
                                   ["--sampling", "huge"]])
def test_sampling_cli_exclusive(monkeypatch, extra):
  monkeypatch.setattr(sys, "argv", D22_RUN_ARGV + extra)
  with pytest.raises(SystemExit):
    parse_args(create_argparser())

def test_explicit_directions_are_not_changed_by_the_sampling(monkeypatch):
  from mcstas_gisans.parameters import set_outgoing_directions_from_sampling
  monkeypatch.setattr(sys, "argv", D22_RUN_ARGV + ["--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "7"])
  args = parse_args(create_argparser())
  assert args.rays_per_pixel is None
  set_outgoing_directions_from_sampling(args, _particles(100, 0), "neutron")
  params = pack_parameters(args, "neutron")
  assert (params['outgoing_directions_horizontal'], params['outgoing_directions_vertical']) == (10, 7)

def test_default_sampling_constant(monkeypatch):
  import mcstas_gisans.run_cli as run_cli
  monkeypatch.setattr(sys, "argv", D22_RUN_ARGV)
  args = parse_args(create_argparser())
  assert run_cli.DEFAULT_SAMPLING == 'quick'
  assert args.sampling == 'quick' and args.rays_per_pixel == 250 and args.outgoing_directions is None
  monkeypatch.setattr(run_cli, "DEFAULT_SAMPLING", None)  # without a default preset: the fixed default grid
  args = parse_args(create_argparser())
  assert args.rays_per_pixel is None and args.outgoing_directions == run_cli.DEFAULT_OUTGOING_DIRECTIONS
  monkeypatch.setattr(run_cli, "DEFAULT_SAMPLING", "quick")
  monkeypatch.setattr(sys, "argv", D22_RUN_ARGV + ["-n", "30"])  # explicit directions still win over the default
  args = parse_args(create_argparser())
  assert args.rays_per_pixel is None and args.outgoing_directions == 30
