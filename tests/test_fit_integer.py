"""
Tests for the optimizer plumbing of mg_fit (integer parameters, evaluation budgets,
parameter scaling, joint fits), using a mocked simulation.
"""
import sys

import numpy as np
import pytest

import mcstas_gisans.fit as fit
from mcstas_gisans.fit import create_fit_parser, run_automated_fit


def _metrics(loss):
    return {'poisson_deviance': loss, 'reduced_chi2': loss, 'log_residual': loss, 'mc_to_poisson_variance': 0.0}


def _mock_evaluation(monkeypatch, loss_of_point=lambda point: 1.0, calls=None):
    calls = [] if calls is None else calls

    def mock_run_simulation_evaluation(grid_point, args_eval, *args, **kwargs):
        calls.append((grid_point, args_eval))
        loss = loss_of_point(grid_point)
        return _metrics(loss), {**grid_point, **_metrics(loss)}, {}

    monkeypatch.setattr(fit, "run_simulation_evaluation", mock_run_simulation_evaluation)
    monkeypatch.setattr(fit, "save_and_print_summary", lambda *args, **kwargs: None)
    return calls


def _args(tmp_path, **kwargs):
    class Args:
        fit = None
        fit2 = None
        fit_common = None
        nxs2 = None
        fit_integer = None
        optimizer = "nelder-mead"
        popsize = 15
        max_evals = 10
        loss_function = "poisson_deviance"
        xatol = 0.01
        fatol = 0.05
        gif = False
        seed = 1
    args = Args()
    args.output_dir = str(tmp_path)
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def _run(args):
    run_automated_fit(args, particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None,
                      y_edges_nxs=None, z_edges_nxs=None, mask=None)


def _parse(argv, monkeypatch):
    from mcstas_gisans.run_cli import parse_args
    parser = create_fit_parser()
    monkeypatch.setattr(sys, "argv", ["mg_fit"] + argv)
    return parser, parse_args(parser)


BASE_ARGV = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0",
             "--nxs", "data/paper/d22_measurement/073174.nxs", "--experiment_time", "60"]


def test_fit_integer_parsing(monkeypatch):
    _, args = _parse(BASE_ARGV + ["--fit", "layerNumber", "3", "1", "10", "--fit_integer", "layerNumber"], monkeypatch)
    assert args.fit_integer == [["layerNumber"]]


def test_fit_integer_must_name_a_fitted_parameter(monkeypatch):
    parser, args = _parse(BASE_ARGV + ["--fit", "layerNumber", "3", "1", "10", "--fit_integer", "layernumber"], monkeypatch)
    with pytest.raises(SystemExit):
        fit.validate_fit_args(args, parser)


@pytest.mark.parametrize("fit_spec", [
    ["radius", "70", "40", "60"],        # x0 outside the bounds
    ["radius", "50", "60", "40"],        # min > max
    ["radius", "50", "40", "60", "1"],   # extra token
])
def test_invalid_fit_specifications_are_rejected(monkeypatch, fit_spec):
    parser, args = _parse(BASE_ARGV + ["--fit", *fit_spec], monkeypatch)
    with pytest.raises(SystemExit):
        fit.validate_fit_args(args, parser)


def test_duplicate_fit_parameter_is_rejected(monkeypatch):
    parser, args = _parse(BASE_ARGV + ["--fit", "radius", "50", "40", "60", "--fit", "radius", "55", "40", "60"], monkeypatch)
    with pytest.raises(SystemExit):
        fit.validate_fit_args(args, parser)


def test_experiment_time_is_required(monkeypatch):
    argv = [a for a in BASE_ARGV if a not in ("--experiment_time", "60")]
    parser, args = _parse(argv + ["--fit", "radius", "50", "40", "60"], monkeypatch)
    with pytest.raises(SystemExit):
        fit.validate_fit_args(args, parser)


def test_scan_and_fit_are_exclusive(monkeypatch):
    parser, args = _parse(BASE_ARGV + ["--fit", "radius", "50", "40", "60", "--scan", "radius", "50", "55"], monkeypatch)
    with pytest.raises(SystemExit):
        fit.validate_fit_args(args, parser)


def test_differential_evolution_validation_missing_bounds(monkeypatch):
    parser, args = _parse(BASE_ARGV + ["--fit", "layerNumber", "3", "--optimizer", "differential-evolution"], monkeypatch)
    with pytest.raises(SystemExit):
        fit.validate_fit_args(args, parser)


def test_fit_integer_rounding(monkeypatch, tmp_path):
    calls = _mock_evaluation(monkeypatch)
    _run(_args(tmp_path, fit=[["layerNumber", "3.6", "1.0", "10.0"]], fit_integer=[["layerNumber"]], max_evals=2))
    first_point = calls[0][0]
    assert first_point["layerNumber"] == 4  # rounded, not truncated
    assert isinstance(first_point["layerNumber"], int)


def test_integer_parameter_is_explored_by_nelder_mead(monkeypatch, tmp_path):
    """The initial simplex must move integer parameters by at least one unit (no rounding plateau)."""
    calls = _mock_evaluation(monkeypatch, loss_of_point=lambda p: (p["n"] - 6) ** 2)
    _run(_args(tmp_path, fit=[["n", "2", "0", "10"]], fit_integer=[["n"]], max_evals=40))
    visited = {point["n"] for point, _ in calls}
    assert len(visited) > 1
    assert 6 in visited


@pytest.mark.parametrize("optimizer", ["nelder-mead", "powell"])
def test_evaluation_budget_is_respected(monkeypatch, tmp_path, optimizer):
    calls = _mock_evaluation(monkeypatch, loss_of_point=lambda p: (p["a"] - 1.3) ** 2 + (p["b"] - 7.0) ** 2)
    _run(_args(tmp_path, fit=[["a", "1", "0", "5"], ["b", "5", "0", "10"]], optimizer=optimizer, max_evals=7))
    assert len(calls) <= 7


def test_differential_evolution_budget_and_integers(monkeypatch, tmp_path):
    calls = _mock_evaluation(monkeypatch, loss_of_point=lambda p: (p["radius"] - 52.0) ** 2)
    _run(_args(tmp_path, fit=[["layerNumber", "3.0", "1.0", "10.0"], ["radius", "50.0", "40.0", "60.0"]],
               fit_integer=[["layerNumber"]], optimizer="differential-evolution", popsize=2, max_evals=20))
    assert 0 < len(calls) <= 20
    for point, _ in calls:
        assert isinstance(point["layerNumber"], int)
        assert isinstance(point["radius"], float)


@pytest.mark.parametrize("max_evals, popsize", [(40, 2), (100, 3), (7, 2)])
def test_differential_evolution_respects_the_evaluation_budget(monkeypatch, tmp_path, max_evals, popsize):
    """DE evaluates (maxiter + 1) populations; the total must not exceed --max_evals (except that
    the initial population is always evaluated), and a noisy loss (never converging) uses all of it."""
    rng = np.random.default_rng(0)
    calls = _mock_evaluation(monkeypatch, loss_of_point=lambda p: (p["radius"] - 47.0) ** 2 + rng.normal(0, 1.0))
    _run(_args(tmp_path, fit=[["radius", "50.0", "40.0", "60.0"], ["height", "10.0", "5.0", "20.0"]],
               optimizer="differential-evolution", popsize=popsize, max_evals=max_evals, fatol=1e-9))
    population = max(5, popsize * 2)  # scipy uses at least 5 population members
    assert len(calls) == max(population, (max_evals // population) * population)


@pytest.mark.parametrize("optimizer", ["nelder-mead", "powell"])
def test_bounds_are_passed_to_the_optimizer(monkeypatch, tmp_path, capsys, optimizer):
    """The optimum lies outside the bounds: no evaluation may leave them (no 1e9-penalty evaluations),
    and the optimizer walks up to the bound."""
    calls = _mock_evaluation(monkeypatch, loss_of_point=lambda p: (p["a"] - 9.0) ** 2)
    _run(_args(tmp_path, fit=[["a", "4", "0", "5"]], optimizer=optimizer, max_evals=40, fatol=1e-6))
    # an out-of-bounds point never reaches the simulation: run_automated_fit returns the penalty instead
    assert "Penalty applied" not in capsys.readouterr().out
    assert all(0 <= point["a"] <= 5 for point, _ in calls)
    assert max(point["a"] for point, _ in calls) == pytest.approx(5.0, abs=0.05)


def test_parameters_of_very_different_magnitude_are_fitted_together(monkeypatch, tmp_path):
    """Like the microgel fits: a volume fraction, a position [nm] and an SLD in one fit."""
    target = {"vf": 0.62, "z_pos": -150.0, "sld": 5.0e-6}
    loss = lambda p: ((p["vf"] - 0.62) / 0.1) ** 2 + ((p["z_pos"] + 150.0) / 20.0) ** 2 + ((p["sld"] - 5.0e-6) / 1e-6) ** 2
    calls = _mock_evaluation(monkeypatch, loss_of_point=loss)
    _run(_args(tmp_path, fit=[["vf", "0.5", "0.4", "0.9"], ["z_pos", "-120", "-180", "-10"], ["sld", "5.56e-6", "2e-6", "6.35e-6"]],
               max_evals=300, fatol=1e-6))
    best = min((point for point, _ in calls), key=loss)
    assert best["vf"] == pytest.approx(target["vf"], abs=0.01)
    assert best["z_pos"] == pytest.approx(target["z_pos"], abs=2.0)
    assert best["sld"] == pytest.approx(target["sld"], rel=0.02)


def test_tiny_valued_parameter_is_fitted(monkeypatch, tmp_path):
    """Parameters on the 1e-6 scale (SLDs) must converge; with absolute tolerances they stopped at once."""
    target = 5.3e-6
    calls = _mock_evaluation(monkeypatch, loss_of_point=lambda p: ((p["sld"] - target) / 1e-6) ** 2)
    _run(_args(tmp_path, fit=[["sld", "4.5e-6", "3e-6", "7e-6"]], max_evals=60, fatol=1e-6, xatol=1e-4))
    best = min(calls, key=lambda c: (c[0]["sld"] - target) ** 2)[0]["sld"]
    assert len(calls) > 5
    assert best == pytest.approx(target, rel=0.01)


def _joint_setup(monkeypatch, tmp_path, **kwargs):
    calls = _mock_evaluation(monkeypatch)
    loads = []
    monkeypatch.setattr(fit, "prepare_experimental_data", lambda args: (np.zeros((5, 5)), np.zeros((5, 5)), np.linspace(-1, 1, 6), np.linspace(0, 1, 6), np.ones((5, 5), dtype=bool), np.zeros((5, 5)), np.zeros((5, 5))))
    monkeypatch.setattr(fit, "load_and_precondition_particles", lambda args: loads.append(args) or ([], "neutron", None))
    args = _args(tmp_path, filename="dummy.mcpl.gz", nxs="nxs1.nxs",
                 fit_common=[["radius", "51.0", "40.0", "60.0"]], fit=[["latticeParameter", "114.0", "100.0", "130.0"]],
                 nxs2="nxs2.nxs", fit2=[["latticeParameter", "120.0", "100.0", "130.0"]],
                 sample_arguments2="radius=51;interferenceRange=5", fit_integer=[["latticeParameter"]], max_evals=2,
                 alpha=0.24, intensity_factor=0.2, **kwargs)
    _run(args)
    return calls, loads


def test_joint_fit_execution(monkeypatch, tmp_path):
    calls, loads = _joint_setup(monkeypatch, tmp_path)
    points1 = [p for p, a in calls if getattr(a, 'nxs', None) != "nxs2.nxs"]
    points2 = [p for p, a in calls if getattr(a, 'nxs', None) == "nxs2.nxs"]
    assert points1 and points2
    assert points1[0]["radius"] == points2[0]["radius"]
    assert isinstance(points2[0]["latticeParameter"], int)
    assert points1[0]["latticeParameter"] == 114
    assert points2[0]["latticeParameter"] == 120
    assert loads == []  # nothing differs for sample 2: particles are shared


def test_joint_fit_reloads_particles_for_a_different_intensity_factor(monkeypatch, tmp_path):
    _, loads = _joint_setup(monkeypatch, tmp_path, intensity_factor2=0.001)
    assert len(loads) == 1 and loads[0].intensity_factor == 0.001


def test_differential_evolution_stops_when_the_population_loss_spread_is_below_fatol(monkeypatch, tmp_path):
    """--fatol is DE's convergence criterion: a loss spread just above it keeps DE running, a flat loss stops it."""
    rng = np.random.default_rng(1)
    fit_params = [["radius", "50.0", "40.0", "60.0"], ["height", "10.0", "5.0", "20.0"]]
    # loss ~10 with spread 0.05: above --fatol 0.01 (keep going), below scipy's default 1% of the mean (0.1)
    noisy = _mock_evaluation(monkeypatch, loss_of_point=lambda p: 10.0 + rng.normal(0, 0.05))
    _run(_args(tmp_path, fit=fit_params, optimizer="differential-evolution", popsize=3, max_evals=120, fatol=0.01))
    assert len(noisy) == 120                    # the whole budget is used
    flat = _mock_evaluation(monkeypatch, loss_of_point=lambda p: 1.0)
    _run(_args(tmp_path, fit=fit_params, optimizer="differential-evolution", popsize=3, max_evals=120, fatol=0.01))
    assert len(flat) < 120


def test_differential_evolution_accepts_the_default_63_bit_seed(monkeypatch, tmp_path):
    """run_cli draws the default --seed below 2**63; scipy's legacy seeding only takes seeds below 2**32."""
    fit_params = [["radius", "50.0", "40.0", "60.0"], ["height", "10.0", "5.0", "20.0"]]
    runs = []
    for _ in range(2):
        runs.append(_mock_evaluation(monkeypatch, loss_of_point=lambda p: (p["radius"] - 52.0) ** 2 + (p["height"] - 12.0) ** 2))
        _run(_args(tmp_path, fit=fit_params, optimizer="differential-evolution", popsize=3, max_evals=30,
                   seed=7187904184844630382))  # the seed of a failed cluster run
    assert [p for p, _ in runs[0]] == [p for p, _ in runs[1]]  # reproducible for a given seed
