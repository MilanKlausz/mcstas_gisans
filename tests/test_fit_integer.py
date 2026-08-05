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
        return 1.0, 1.0, {"reduced_chi2": 1.0, "log_residual": 1.0, **grid_point}
        
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
        return 1.0, 1.0, {"reduced_chi2": 1.0, "log_residual": 1.0, **grid_point}
        
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
        return 1.0, 1.0, {"reduced_chi2": 1.0, "log_residual": 1.0, **grid_point}, {}

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
        fit_integer = None
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
    assert called_points1[0]["latticeParameter"] == 114.0
    assert called_points2[0]["latticeParameter"] == 120.0
