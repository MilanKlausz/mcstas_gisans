"""
Tests for continuing a Differential Evolution fit (mg_fit --de_continue): the replay of the evaluations must
recover scipy's population exactly, and a continued fit must start from it without simulating it again.
"""
import csv
import os

import numpy as np
import pytest
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

import mcstas_gisans.fit as fit
from mcstas_gisans.fit import replay_de_population, run_automated_fit


def _toy_loss(x):
    return float((x[0] - 1.3) ** 2 + 0.5 * (x[1] + 0.4) ** 2 + 0.1 * (x[2] - 3) ** 2 + 0.05 * np.sin(7 * x[0]))


@pytest.mark.parametrize("maxiter, maxfun", [(6, np.inf), (6, 41)])  # complete generations / stopped within one
def test_replay_reproduces_the_scipy_population(maxiter, maxfun):
    evaluations = {}

    def recorded(x):
        loss = _toy_loss(x)
        evaluations[len(evaluations) + 1] = (np.array(x, dtype=float), loss)
        return loss

    bounds = [(-2, 3), (-1, 1), (1, 6)]
    # the solver class of scipy.optimize.differential_evolution, for maxfun (a stop within a generation)
    with DifferentialEvolutionSolver(recorded, bounds, x0=[0.5, 0.2, 4], maxiter=maxiter, popsize=3, tol=0, atol=0,
                                     integrality=[False, False, True], polish=False, rng=np.random.default_rng(3),
                                     maxfun=maxfun) as solver:
        res = solver.solve()
    size = len(res.population)                            # popsize 3 x 3 parameters
    assert len(evaluations) == (maxfun + 1 if np.isfinite(maxfun) else size * (maxiter + 1))
    population, energies = replay_de_population(evaluations, len(res.population))
    np.testing.assert_array_equal(population, res.population)
    np.testing.assert_array_equal(energies, res.population_energies)


def test_replay_skips_evaluations_without_record():
    """Out-of-bounds points get the penalty without a record (a gap in eval_index): never accepted."""
    evaluations = {i: (np.array([float(i)]), 10.0 - i) for i in range(1, 6)}   # initial population, best = #5 (loss 5)
    # after the promotion the members are #5, #2, #3, #4, #1; #6 (trial of member 0) has no record,
    # #7 is the trial of member 1 (#2, loss 8) and better
    evaluations[7] = (np.array([70.0]), 7.5)
    population, energies = replay_de_population(evaluations, 5)
    np.testing.assert_array_equal(population[:, 0], [5.0, 70.0, 3.0, 4.0, 1.0])
    np.testing.assert_array_equal(energies, [5.0, 7.5, 7.0, 6.0, 9.0])


def test_replay_needs_the_whole_initial_population():
    with pytest.raises(ValueError, match="initial DE population is incomplete"):
        replay_de_population({1: (np.zeros(1), 1.0), 3: (np.zeros(1), 1.0)}, 5)


# --- the continuation in run_automated_fit, with a mocked simulation ---

def _metrics(loss):
    return {'poisson_deviance': loss, 'reduced_chi2': loss, 'log_residual': loss, 'mc_to_poisson_variance': 0.0}


class _Stop(Exception):
    pass


def _mock(monkeypatch, calls, stop_after=None):
    def evaluation(grid_point, args_eval, *args, **kwargs):
        if stop_after is not None and len(calls) == stop_after:
            raise _Stop()                                   # like a job killed by its time limit
        calls.append(dict(grid_point))
        loss = (grid_point["radius"] - 52.0) ** 2 / 10 + (grid_point["vf"] - 0.6) ** 2 * 50 + 0.2 * grid_point["size"]
        return _metrics(loss), {}, {}
    monkeypatch.setattr(fit, "run_simulation_evaluation", evaluation)


def _args(output_dir, **kwargs):
    class Args:
        fit = [["radius", "50.0", "40.0", "60.0"], ["vf", "0.5", "0.3", "0.9"], ["size", "3", "2", "6"]]
        fit2 = None
        fit_common = None
        nxs2 = None
        fit_integer = [["size"]]
        optimizer = "differential-evolution"
        popsize = 2
        max_evals = 60
        loss_function = "poisson_deviance"
        xatol = 0.01
        fatol = 1e-9
        gif = False
        seed = 1
        de_continue = None
    args = Args()
    args.output_dir = str(output_dir)
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def _run(args):
    run_automated_fit(args, particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None,
                      y_edges_nxs=None, z_edges_nxs=None, mask=None)


def _summary(directory):
    with open(os.path.join(directory, "fit_summary.csv"), newline='') as f:
        return [r for r in csv.DictReader(f) if (r.get('eval_index') or '').isdigit()]


def test_continued_fit_starts_from_the_stopped_population(monkeypatch, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    calls = []
    _mock(monkeypatch, calls, stop_after=15)              # population 6: initial 6 + 9 trials, stopped within generation 2
    with pytest.raises(_Stop):
        _run(_args(first))
    assert len(_summary(first)) == 15

    calls.clear()
    _mock(monkeypatch, calls)
    _run(_args(second, de_continue=str(first), max_evals=18))
    rows = _summary(second)
    indices = sorted(int(r['eval_index']) for r in rows)
    assert indices == list(range(1, 16 + len(calls)))     # the earlier evaluations are kept, the numbering continues
    assert len(calls) == 18                               # the recovered population is not simulated again
    recovered = fit.load_de_continuation(str(first), ["radius", "vf", "size"], "poisson_deviance", 6)
    first_points = [tuple(p) for p in recovered['population']]
    assert all(tuple(float(c[name]) for name in ("radius", "vf", "size")) not in first_points for c in calls[:6])
    best_first = min(float(r['poisson_deviance']) for r in _summary(first))
    assert min(float(r['poisson_deviance']) for r in rows) <= best_first


def test_continuation_reads_a_finished_summary_with_the_printed_table(tmp_path):
    """A finished fit appends its printed results table and runtimes to fit_summary.csv; only the rows of
    evaluations count."""
    path = tmp_path / "fit_summary.csv"
    lines = ["radius,vf,eval_index,poisson_deviance"] + [f"{50 + i},{0.5},{i},{10 - i}" for i in range(1, 6)] + [
        "", "--- Optimization Results (Sorted by poisson_deviance) ---", "radius | vf | eval_index", "Total Runtime: 1h 0m 0s (3600.00s)",
        "  background = 1.0 (sample 1), 2.0 (sample 2) [counts per pixel]"]
    path.write_text("\n".join(lines) + "\n")
    c = fit.load_de_continuation(str(path), ["radius", "vf"], "poisson_deviance", 5)
    assert c['last_eval_index'] == 5 and len(c['records']) == 5
    assert c['energies'][0] == 5.0 and c['population'][0, 0] == 55.0


def test_de_continue_needs_differential_evolution(monkeypatch, tmp_path):
    _mock(monkeypatch, [])
    with pytest.raises(ValueError, match="requires --optimizer differential-evolution"):
        _run(_args(tmp_path, optimizer="nelder-mead", de_continue=str(tmp_path)))
