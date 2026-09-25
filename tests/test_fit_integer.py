import pytest
import sys
import numpy as np
from mcstas_gisans.fit import create_fit_parser, run_automated_fit
import mcstas_gisans.fit as fit

def test_fit_integer_parsing():
    parser = create_fit_parser()
    argv = [
        "dummy.mcpl.gz",
        "-i", "d22",
        "--wavelength_selected", "6.0",
        "--nxs", "data/paper/d22_measurement/073174.nxs",
        "--fit", "layerNumber", "3", "1", "10",
        "--fit_integer", "layerNumber"
    ]
    sys_argv_backup = sys.argv
    sys.argv = ["scan"] + argv
    try:
        from mcstas_gisans.run_cli import parse_args
        args = parse_args(parser)
        assert args.fit_integer == [["layerNumber"]]
    finally:
        sys.argv = sys_argv_backup

def test_fit_integer_rounding(monkeypatch):
    # Mock run_simulation_evaluation to check if parameter was rounded
    called_points = []
    def mock_run_simulation_evaluation(grid_point, *args, **kwargs):
        called_points.append(grid_point)
        return {"poisson_deviance": 1.0, "reduced_chi2": 1.0, "log_residual": 1.0, "mc_to_poisson_variance": 0.0}, {**grid_point}, {}
        
    monkeypatch.setattr(fit, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(fit, "save_and_print_summary", lambda *args, **kwargs: None)
    
    class DummyArgs:
        fit = [["layerNumber", "3.2", "1.0", "10.0"]]
        fit_integer = [["layerNumber"]]
        optimizer = "nelder-mead"
        poisson_sampling = False
        max_evals = 2
        loss_function = "reduced_chi2"
        xatol = 0.01
        fatol = 0.05
        output_dir = "dummy_output"
        gif = False

    args = DummyArgs()
    run_automated_fit(args, particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None, y_edges_nxs=None, z_edges_nxs=None, mask=None)
    
    # Check that initial evaluation point was rounded to 3 (instead of 3.2)
    assert len(called_points) > 0
    assert called_points[0]["layerNumber"] == 3
    assert isinstance(called_points[0]["layerNumber"], int)

def test_differential_evolution_validation_missing_bounds():
    parser = create_fit_parser()
    # Missing bounds for layerNumber (only initial value provided)
    argv = [
        "dummy.mcpl.gz",
        "-i", "d22",
        "--wavelength_selected", "6.0",
        "--nxs", "data/paper/d22_measurement/073174.nxs",
        "--fit", "layerNumber", "3",
        "--optimizer", "differential-evolution"
    ]
    sys_argv_backup = sys.argv
    sys.argv = ["scan"] + argv
    try:
        from mcstas_gisans.run_cli import parse_args
        args = parse_args(parser)
        with pytest.raises(SystemExit):
            fit.validate_fit_args(args, parser)
    finally:
        sys.argv = sys_argv_backup

def test_differential_evolution_execution(monkeypatch):
    called_points = []
    def mock_run_simulation_evaluation(grid_point, *args, **kwargs):
        called_points.append(grid_point)
        return {"poisson_deviance": 1.0, "reduced_chi2": 1.0, "log_residual": 1.0, "mc_to_poisson_variance": 0.0}, {**grid_point}, {}
        
    monkeypatch.setattr(fit, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(fit, "save_and_print_summary", lambda *args, **kwargs: None)
    
    class DummyArgs:
        fit = [["layerNumber", "3.0", "1.0", "10.0"], ["radius", "50.0", "40.0", "60.0"]]
        fit_integer = [["layerNumber"]]
        optimizer = "differential-evolution"
        popsize = 2
        poisson_sampling = False
        max_evals = 2
        loss_function = "reduced_chi2"
        xatol = 0.01
        fatol = 0.05
        output_dir = "dummy_output"
        gif = False

    args = DummyArgs()
    run_automated_fit(args, particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None, y_edges_nxs=None, z_edges_nxs=None, mask=None)
    
    assert len(called_points) > 0
    # Ensure it parsed and passed integer parameters as actual integers
    for point in called_points:
        assert isinstance(point["layerNumber"], int)
        assert isinstance(point["radius"], float)

def test_joint_fit_execution(monkeypatch):
    called_points1 = []
    called_points2 = []
    def mock_run_simulation_evaluation(grid_point, args_eval, *args, **kwargs):
        if getattr(args_eval, 'nxs', None) == "nxs2.nxs":
            called_points2.append(grid_point)
        else:
            called_points1.append(grid_point)
        return {"poisson_deviance": 1.0, "reduced_chi2": 1.0, "log_residual": 1.0, "mc_to_poisson_variance": 0.0}, {**grid_point}, {}

    monkeypatch.setattr(fit, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(fit, "save_and_print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(fit, "prepare_experimental_data", lambda args: (np.zeros((5, 5)), np.zeros((5, 5)), np.linspace(-1, 1, 6), np.linspace(0, 1, 6), np.ones((5, 5), dtype=bool), np.zeros((5, 5)), np.zeros((5, 5))))
    monkeypatch.setattr(fit, "load_and_precondition_particles", lambda args: ([], "neutron"))

    class DummyJointArgs:
        filename = "dummy.mcpl.gz"
        nxs = "nxs1.nxs"
        fit_common = [["radius", "51.0", "40.0", "60.0"]]
        fit = [["latticeParameter", "114.0", "100.0", "130.0"]]
        nxs2 = "nxs2.nxs"
        fit2 = [["latticeParameter", "120.0", "100.0", "130.0"]]
        sample_arguments2 = "radius=51;interferenceRange=5"
        optimizer = "nelder-mead"
        fit_integer = [["latticeParameter"]]
        poisson_sampling = False
        max_evals = 2
        loss_function = "reduced_chi2"
        xatol = 0.01
        fatol = 0.05
        output_dir = "dummy_output"
        gif = False

    args = DummyJointArgs()
    run_automated_fit(args, particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None, y_edges_nxs=None, z_edges_nxs=None, mask=None)

    assert len(called_points1) > 0
    assert len(called_points2) > 0
    assert "radius" in called_points1[0]
    assert "radius" in called_points2[0]
    assert called_points1[0]["radius"] == called_points2[0]["radius"]
    assert isinstance(called_points2[0]["latticeParameter"], int)
    assert called_points1[0]["latticeParameter"] == 114.0
    assert called_points2[0]["latticeParameter"] == 120.0

@pytest.mark.parametrize("max_evals, popsize", [(40, 2), (100, 3), (7, 2)])
def test_differential_evolution_respects_the_evaluation_budget(monkeypatch, max_evals, popsize):
    """DE evaluates (maxiter + 1) populations; the total must not exceed --max_evals (except that
    the initial population is always evaluated)."""
    rng = np.random.default_rng(0)
    evaluations = []
    def mock_run_simulation_evaluation(grid_point, *args, **kwargs):
        evaluations.append(grid_point)
        loss = (grid_point["radius"] - 47.0) ** 2 + rng.normal(0, 1.0)  # noisy: never converges early
        return {"poisson_deviance": loss, "reduced_chi2": loss, "log_residual": loss, "mc_to_poisson_variance": 0.0}, {**grid_point}, {}
    monkeypatch.setattr(fit, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(fit, "save_and_print_summary", lambda *args, **kwargs: None)

    class DummyArgs:
        fit = [["radius", "50.0", "40.0", "60.0"], ["height", "10.0", "5.0", "20.0"]]
        fit_integer = None
        optimizer = "differential-evolution"
        poisson_sampling = False
        loss_function = "poisson_deviance"
        xatol = 0.01
        fatol = 1e-9
        gif = False
    args = DummyArgs()
    args.max_evals, args.popsize = max_evals, popsize
    args.output_dir = "dummy_output"
    run_automated_fit(args, particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None, y_edges_nxs=None, z_edges_nxs=None, mask=None)
    population = max(5, popsize * 2)
    assert len(evaluations) == max(population, (max_evals // population) * population)


def test_differential_evolution_stops_when_the_population_loss_spread_is_below_fatol(monkeypatch):
    """A flat loss: the population's loss spread is 0 < --fatol, so DE stops after the first generation."""
    evaluations = []
    def mock_run_simulation_evaluation(grid_point, *args, **kwargs):
        evaluations.append(grid_point)
        return {"poisson_deviance": 1.0, "reduced_chi2": 1.0, "log_residual": 1.0, "mc_to_poisson_variance": 0.0}, {**grid_point}, {}
    monkeypatch.setattr(fit, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(fit, "save_and_print_summary", lambda *args, **kwargs: None)

    class DummyArgs:
        fit = [["radius", "50.0", "40.0", "60.0"], ["height", "10.0", "5.0", "20.0"]]
        fit_integer = None
        optimizer = "differential-evolution"
        popsize = 3
        poisson_sampling = False
        max_evals = 600
        loss_function = "poisson_deviance"
        xatol = 0.01
        fatol = 0.05
        output_dir = "dummy_output"
        gif = False
    run_automated_fit(DummyArgs(), particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None, y_edges_nxs=None, z_edges_nxs=None, mask=None)
    assert len(evaluations) < 600


def _run_fit(monkeypatch, fit_params, loss_of_point, optimizer="nelder-mead", max_evals=60, fatol=1e-6, xatol=0.01, fit_integer=None):
    evaluations = []
    def mock_run_simulation_evaluation(grid_point, *args, **kwargs):
        evaluations.append(dict(grid_point))
        loss = loss_of_point(grid_point)
        return {"poisson_deviance": loss, "reduced_chi2": loss, "log_residual": loss, "mc_to_poisson_variance": 0.0}, {**grid_point}, {}
    monkeypatch.setattr(fit, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(fit, "save_and_print_summary", lambda *args, **kwargs: None)
    class DummyArgs:
        pass
    args = DummyArgs()
    args.fit, args.fit_integer, args.optimizer, args.max_evals = fit_params, fit_integer, optimizer, max_evals
    args.fatol, args.xatol, args.popsize = fatol, xatol, 15
    args.poisson_sampling, args.loss_function, args.output_dir, args.gif = False, "poisson_deviance", "dummy_output", False
    run_automated_fit(args, particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None, y_edges_nxs=None, z_edges_nxs=None, mask=None)
    return evaluations


def test_tiny_valued_parameter_is_fitted(monkeypatch):
    """Parameters on the 1e-6 scale (SLDs) must converge; with absolute tolerances they stopped at once."""
    target = 5.3e-6
    evaluations = _run_fit(monkeypatch, [["sld", "4.5e-6", "3e-6", "7e-6"]], lambda p: ((p["sld"] - target) / 1e-6) ** 2, xatol=1e-4)
    best = min(evaluations, key=lambda p: (p["sld"] - target) ** 2)["sld"]
    assert len(evaluations) > 5
    assert best == pytest.approx(target, rel=0.01)


def test_parameters_of_very_different_magnitude_are_fitted_together(monkeypatch):
    """Like the microgel fits: a volume fraction, a position [nm] and an SLD in one fit."""
    target = {"vf": 0.62, "z_pos": -150.0, "sld": 5.0e-6}
    loss = lambda p: ((p["vf"] - 0.62) / 0.1) ** 2 + ((p["z_pos"] + 150.0) / 20.0) ** 2 + ((p["sld"] - 5.0e-6) / 1e-6) ** 2
    evaluations = _run_fit(monkeypatch, [["vf", "0.5", "0.4", "0.9"], ["z_pos", "-120", "-180", "-10"], ["sld", "5.56e-6", "2e-6", "6.35e-6"]],
                           loss, max_evals=300)
    best = min(evaluations, key=loss)
    assert best["vf"] == pytest.approx(target["vf"], abs=0.01)
    assert best["z_pos"] == pytest.approx(target["z_pos"], abs=2.0)
    assert best["sld"] == pytest.approx(target["sld"], rel=0.02)


def test_integer_parameter_is_explored_by_nelder_mead(monkeypatch):
    """The initial simplex must move integer parameters by at least one unit (no rounding plateau)."""
    evaluations = _run_fit(monkeypatch, [["n", "2", "0", "10"]], lambda p: (p["n"] - 6) ** 2, max_evals=40, fit_integer=[["n"]])
    visited = {p["n"] for p in evaluations}
    assert len(visited) > 1 and 6 in visited


@pytest.mark.parametrize("optimizer", ["nelder-mead", "powell"])
def test_evaluation_budget_is_respected(monkeypatch, optimizer):
    evaluations = _run_fit(monkeypatch, [["a", "1", "0", "5"], ["b", "5", "0", "10"]],
                           lambda p: (p["a"] - 1.3) ** 2 + (p["b"] - 7.0) ** 2, optimizer=optimizer, max_evals=7)
    assert len(evaluations) <= 7


@pytest.mark.parametrize("optimizer", ["nelder-mead", "powell"])
def test_bounds_are_passed_to_the_optimizer(monkeypatch, optimizer):
    """The optimum lies outside the bounds: no evaluation may leave them (no 1e9-penalty evaluations)."""
    evaluations = _run_fit(monkeypatch, [["a", "4", "0", "5"]], lambda p: (p["a"] - 9.0) ** 2, optimizer=optimizer, max_evals=40)
    assert all(0 <= p["a"] <= 5 for p in evaluations)
    assert max(p["a"] for p in evaluations) == pytest.approx(5.0, abs=0.05)
