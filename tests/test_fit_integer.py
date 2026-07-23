import pytest
import sys
from mcstas_gisans.scan import create_scan_parser, run_automated_fit
import mcstas_gisans.scan as scan

def test_fit_integer_parsing():
    parser = create_scan_parser()
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
        
    monkeypatch.setattr(scan, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(scan, "save_and_print_summary", lambda *args, **kwargs: None)
    
    class DummyArgs:
        fit = [["layerNumber", "3.2", "1.0", "10.0"]]
        fit_integer = [["layerNumber"]]
        optimizer = "nelder-mead"
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
    parser = create_scan_parser()
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
            scan.validate_scan_args(args, parser)
    finally:
        sys.argv = sys_argv_backup

def test_differential_evolution_execution(monkeypatch):
    called_points = []
    def mock_run_simulation_evaluation(grid_point, *args, **kwargs):
        called_points.append(grid_point)
        return 1.0, 1.0, {"reduced_chi2": 1.0, "log_residual": 1.0, **grid_point}
        
    monkeypatch.setattr(scan, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(scan, "save_and_print_summary", lambda *args, **kwargs: None)
    
    class DummyArgs:
        fit = [["layerNumber", "3.0", "1.0", "10.0"], ["radius", "50.0", "40.0", "60.0"]]
        fit_integer = [["layerNumber"]]
        optimizer = "differential-evolution"
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
