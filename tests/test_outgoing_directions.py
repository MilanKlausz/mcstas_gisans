import pytest
import sys
from mcstas_gisans.run_cli import create_argparser, parse_args
from mcstas_gisans.parameters import pack_parameters

@pytest.fixture
def parser():
    return create_argparser()

def test_default_is_the_quick_sampling_preset(monkeypatch, parser):
    monkeypatch.setattr(sys, 'argv', ["run", "dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0"])
    args = parse_args(parser)
    assert args.sampling == 'quick' and args.rays_per_pixel == 250
    assert args.outgoing_directions is None and args.outgoing_directions_horizontal is None


@pytest.mark.parametrize("argv,expected_horiz,expected_vert,expected_out", [
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


def _substrate():
    import bornagain as ba
    sample = ba.MultiLayer() if hasattr(ba, 'MultiLayer') else ba.Sample()  # ba.Sample from BornAgain 22 on
    sample.addLayer(ba.Layer(ba.RefractiveMaterial("Vacuum", 0.0, 0.0)))
    sample.addLayer(ba.Layer(ba.RefractiveMaterial("Si", 7.6e-6, 1.7e-10)))
    return sample


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rays_use_the_directions_bornagain_evaluates(monkeypatch, seed):
    """Every intensity must go to the direction at which BornAgain evaluated it: the ray angles are
    the bin centres of the (randomly shifted) BornAgain detector, in the order of the intensities."""
    import numpy as np
    from bornagain import deg
    from mcstas_gisans import run
    detectors = []
    original = run.get_simulation
    def capture(*args, **kwargs):
        sim = original(*args, **kwargs)
        detector = sim.detector()  # read now: the detector does not outlive the simulation
        detectors.append((np.array(detector.axis(0).binCenters()) / deg, np.array(detector.axis(1).binCenters()) / deg))
        return sim
    monkeypatch.setattr(run, "get_simulation", capture)
    np.random.seed(seed)
    alpha_i, n_h, n_v = 1.0, 7, 9
    params = {'angle_range': [-0.6, 0.8, 0.1, 2.0], 'outgoing_directions_horizontal': n_h,
              'outgoing_directions_vertical': n_v, 'specular': 'include_specular'}
    weights, alpha_f, phi_f = run._execute_bornagain_simulation(_substrate(), 6.0, alpha_i, 1.0, [], params)

    phi_centres, alpha_centres = detectors[0]
    np.testing.assert_allclose(phi_f, phi_centres, atol=1e-12)
    np.testing.assert_allclose(alpha_f[::-1], alpha_centres, atol=1e-12)
    # ordering: the specular reflection of a bare substrate is the brightest ray, at alpha_f = alpha_i, phi_f = 0
    alpha_grid, phi_grid = np.meshgrid(alpha_f, phi_f)
    brightest = np.argmax(weights)
    assert abs(alpha_grid.flatten()[brightest] - alpha_i) <= 0.5 * (2.0 - 0.1) / n_v
    assert abs(phi_grid.flatten()[brightest]) <= 0.5 * 1.4 / n_h


D22_RUN_ARGV = ["mg_run", "dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--alpha", "0.24",
                "--sample_orientation", "2", "--angle_range", "-1.2122", "1.1939", "0.1547", "0.3266"]


def _detector(monkeypatch, sample_orientation):
    argv = D22_RUN_ARGV[:8] + ["--sample_orientation", str(sample_orientation), "-n", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    return pack_parameters(parse_args(create_argparser()), "neutron")['instrument'].detector


def test_outgoing_directions_for_sampling_follow_the_detector_pixels(monkeypatch):
    """D22 at 17.6 m: 8 mm tubes = 0.0260436 deg, 4 mm pixels = 0.0130218 deg. For a vertical sample the tubes lie
    along the sample normal (BornAgain vertical), for a horizontal sample across it. The directions per pixel along
    each axis are sqrt(R / n_hit), the same on both axes."""
    from mcstas_gisans.parameters import outgoing_directions_for_sampling
    # vertical sample, paper fit mask range: pixels 2.4061/0.0130218 = 184.78 x 0.1719/0.0260436 = 6.600;
    # sqrt(2000/16963) = 0.34337 -> ceil(63.45) x ceil(2.266) = 64 x 3
    paper = _detector(monkeypatch, 2)
    assert outgoing_directions_for_sampling([-1.2122, 1.1939, 0.1547, 0.3266], paper, 17.6, 16963, 2000) == (64, 3)
    # horizontal sample, NP fit mask range: pixels 2.6249/0.0260436 = 100.79 x 1.0468/0.0130218 = 80.39;
    # sqrt(2000/16963) -> ceil(34.61) x ceil(27.60) = 35 x 28; sqrt(250/100000) = 0.05 -> ceil(5.04) x ceil(4.02) = 6 x 5
    liquid = _detector(monkeypatch, 1)
    angle_range = [-1.3030, 1.3219, 0.2256, 1.2724]
    assert outgoing_directions_for_sampling(angle_range, liquid, 17.6, 16963, 2000) == (35, 28)
    assert outgoing_directions_for_sampling(angle_range, liquid, 17.6, 100000, 250) == (6, 5)
    assert outgoing_directions_for_sampling(angle_range, liquid, 17.6, 10**9, 250) == (1, 1)  # at least one per axis
    # no neutron hitting the sample is treated as one: sqrt(2000) = 44.72 -> ceil(4507.4) x ceil(3595.1)
    assert outgoing_directions_for_sampling(angle_range, liquid, 17.6, 0, 2000) == (4508, 3596)


def _particles(n_hit, n_miss):
    """Preconditioned particles in the BornAgain frame (p, x, y, z, vx, vy, vz, wavelength, t): n_hit on the surface
    of the default 6 cm x 8 cm sample, n_miss next to it (y outside the 3 cm half-width)."""
    import numpy as np
    rows = [[1.0, 0.0, 0.0, 0.0, 400.0, 0.0, -1.0, 6.0, 0.0]] * n_hit + [[1.0, 0.0, 0.05, 0.0, 400.0, 0.0, -1.0, 6.0, 0.0]] * n_miss
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
