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


def _substrate():
    import bornagain as ba
    sample = ba.MultiLayer()
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
