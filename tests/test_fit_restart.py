"""
Tests of the output-directory state of mg_fit (fit_restart): an interrupted fit continued in the same directory
retraces its evaluations from fit_evaluations.jsonl without simulating them and ends exactly as an uninterrupted
fit; finished fits and fits with other settings are not touched; --overwrite starts from scratch.
"""
import csv
import json
import os
import signal
from types import SimpleNamespace

import numpy as np
import pytest

import mcstas_gisans.fit as fit
from mcstas_gisans import fit_restart
from mcstas_gisans.fit import run_automated_fit


class _Stop(Exception):
    pass


def _loss(p):
    return (p["radius"] - 52.3) ** 2 / 10 + (p["vf"] - 0.61) ** 2 * 50 + 0.2 * (p["size"] - 4) ** 2 + 0.1 * np.sin(p["radius"])


def _mock(monkeypatch, calls, stop_after=None, action=None):
    """The simulation: a deterministic loss; after stop_after simulations it stops (raises, or runs action)."""
    def evaluation(grid_point, args_eval, *args, **kwargs):
        if stop_after is not None and len(calls) == stop_after:
            if action is not None:
                action()
            raise _Stop()
        calls.append(dict(grid_point))
        loss = _loss(grid_point)
        return {'poisson_deviance': loss, 'reduced_chi2': loss, 'log_residual': loss, 'mc_to_poisson_variance': 0.0}, {}, {}
    monkeypatch.setattr(fit, "run_simulation_evaluation", evaluation)


def _args(output_dir, **kwargs):
    values = dict(fit=[["radius", "50.0", "40.0", "60.0"], ["vf", "0.5", "0.3", "0.9"], ["size", "3", "2", "6"]],
                  fit2=None, fit_common=None, nxs2=None, fit_integer=[["size"]], optimizer="differential-evolution",
                  popsize=2, max_evals=48, loss_function="poisson_deviance", xatol=0.01, fatol=1e-9, gif=False,
                  seed=11, seed_given=True, de_continue=None, polish=None, polish_max_evals=40, polish_xatol=0.002,
                  polish_fatol=0.002, overwrite=False, output_dir=str(output_dir))
    values.update(kwargs)
    return SimpleNamespace(**values)


def _run(args):
    run_automated_fit(args, particles=[], particle_type="neutron", hist_nxs=None, hist_nxs_error=None,
                      y_edges_nxs=None, z_edges_nxs=None, mask=None)


def _summary(directory):
    with open(os.path.join(directory, "fit_summary.csv"), newline='') as f:
        rows = [r for r in csv.DictReader(f) if (r.get('eval_index') or '').isdigit()]
    return sorted(rows, key=lambda r: int(r['eval_index']))


def _state(directory):
    with open(os.path.join(directory, fit_restart.STATE_FILE)) as f:
        return json.load(f)


def _points(rows):
    return [(r['phase'], float(r['radius']), float(r['vf']), int(r['size']), float(r['poisson_deviance'])) for r in rows]


CASES = {
    "de": dict(optimizer="differential-evolution"),
    "de+nelder-mead polish": dict(optimizer="differential-evolution", polish="nelder-mead"),
    "de+powell polish": dict(optimizer="differential-evolution", polish="powell"),
    "nelder-mead": dict(optimizer="nelder-mead", max_evals=60),
    "powell": dict(optimizer="powell", max_evals=60),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("stop_fraction", [0.2, 0.75])
def test_interrupted_fit_continues_exactly_as_an_uninterrupted_one(monkeypatch, tmp_path, case, stop_fraction):
    kwargs = CASES[case]
    reference_calls = []
    _mock(monkeypatch, reference_calls)
    _run(_args(tmp_path / "reference", **kwargs))
    reference = _points(_summary(tmp_path / "reference"))
    total = len(reference_calls)
    assert total == len(reference) > 10

    stop = int(total * stop_fraction)          # within the DE or within the polish
    first = []
    _mock(monkeypatch, first, stop_after=stop)
    with pytest.raises(_Stop):
        _run(_args(tmp_path / "interrupted", **kwargs))
    assert _state(tmp_path / "interrupted")['status'] == 'failed'
    assert len(_summary(tmp_path / "interrupted")) == stop

    second = []
    _mock(monkeypatch, second)
    _run(_args(tmp_path / "interrupted", **kwargs))       # the same command again
    assert len(second) == total - stop                    # only the missing evaluations are simulated
    assert _points(_summary(tmp_path / "interrupted")) == reference
    state = _state(tmp_path / "interrupted")
    assert state['status'] == 'finished'
    assert [s['cached_evaluations'] for s in state['segments']] == [0, stop]
    assert [s['new_evaluations'] for s in state['segments']] == [stop, total - stop]


def test_polish_improves_on_the_de_best_and_keeps_the_integers(monkeypatch, tmp_path):
    calls = []
    _mock(monkeypatch, calls)
    _run(_args(tmp_path, polish="nelder-mead", fatol=0.05))
    rows = _summary(tmp_path)
    de = [r for r in rows if r['phase'] == 'de']
    polish = [r for r in rows if r['phase'] == 'polish']
    assert de and polish
    best_de = min(de, key=lambda r: float(r['poisson_deviance']))
    assert {r['size'] for r in polish} == {best_de['size']}           # the integer stays at its DE value
    assert min(float(r['poisson_deviance']) for r in polish) <= float(best_de['poisson_deviance'])
    assert _state(tmp_path)['result']['best_eval_index'] in [int(r['eval_index']) for r in rows]


def test_finished_fit_is_not_run_again(monkeypatch, tmp_path):
    _mock(monkeypatch, [])
    _run(_args(tmp_path))
    calls = []
    _mock(monkeypatch, calls)
    with pytest.raises(SystemExit, match="FINISHED"):
        _run(_args(tmp_path))
    assert calls == []


def test_other_settings_are_refused_with_the_differences(monkeypatch, tmp_path):
    _mock(monkeypatch, [], stop_after=10)
    with pytest.raises(_Stop):
        _run(_args(tmp_path))
    calls = []
    _mock(monkeypatch, calls)
    with pytest.raises(SystemExit, match=r"other settings(.|\n)*fit:"):
        _run(_args(tmp_path, fit=[["radius", "50.0", "40.0", "70.0"], ["vf", "0.5", "0.3", "0.9"], ["size", "3", "2", "6"]]))
    assert calls == []
    _run(_args(tmp_path, max_evals=60))      # a larger budget is not another setting


def test_overwrite_starts_from_scratch(monkeypatch, tmp_path):
    first = []
    _mock(monkeypatch, first)
    _run(_args(tmp_path))
    second = []
    _mock(monkeypatch, second)
    _run(_args(tmp_path, overwrite=True))
    assert len(second) == len(first)
    assert len(_summary(tmp_path)) == len(first)
    assert len(_state(tmp_path)['segments']) == 1


def test_summary_without_state_is_not_continued(monkeypatch, tmp_path):
    (tmp_path / "fit_summary.csv").write_text("radius,eval_index,poisson_deviance\n50,1,3.0\n")
    _mock(monkeypatch, [])
    with pytest.raises(SystemExit, match="without fit_state.json"):
        _run(_args(tmp_path))


def test_continuation_without_seed_uses_the_seed_of_the_fit(monkeypatch, tmp_path):
    reference = []
    _mock(monkeypatch, reference)
    _run(_args(tmp_path / "reference"))
    _mock(monkeypatch, [], stop_after=15)
    with pytest.raises(_Stop):
        _run(_args(tmp_path / "fit"))
    second = []
    _mock(monkeypatch, second)
    _run(_args(tmp_path / "fit", seed=987654321, seed_given=False))   # a random default seed: replaced by the stored one
    assert len(second) == len(reference) - 15
    assert _points(_summary(tmp_path / "fit")) == _points(_summary(tmp_path / "reference"))


def test_sigterm_records_interrupted_and_the_fit_continues(monkeypatch, tmp_path):
    reference = []
    _mock(monkeypatch, reference)
    _run(_args(tmp_path / "reference"))
    _mock(monkeypatch, [], stop_after=20, action=lambda: os.kill(os.getpid(), signal.SIGTERM))
    with pytest.raises(SystemExit) as stop:
        _run(_args(tmp_path / "fit"))
    assert stop.value.code == 128 + signal.SIGTERM
    assert _state(tmp_path / "fit")['status'] == 'interrupted'
    second = []
    _mock(monkeypatch, second)
    _run(_args(tmp_path / "fit"))
    assert len(second) == len(reference) - 20
    assert _points(_summary(tmp_path / "fit")) == _points(_summary(tmp_path / "reference"))


def test_incomplete_last_cache_line_is_redone(monkeypatch, tmp_path):
    reference = []
    _mock(monkeypatch, reference)
    _run(_args(tmp_path / "reference"))
    _mock(monkeypatch, [], stop_after=12)
    with pytest.raises(_Stop):
        _run(_args(tmp_path / "fit"))
    cache = tmp_path / "fit" / fit_restart.CACHE_FILE
    text = cache.read_text()
    cache.write_text(text[:-40])               # the last line cut, as by a kill while it was written
    second = []
    _mock(monkeypatch, second)
    _run(_args(tmp_path / "fit"))
    assert len(second) == len(reference) - 11
    assert _points(_summary(tmp_path / "fit")) == _points(_summary(tmp_path / "reference"))


def test_diverging_path_drops_the_rest_of_the_cache(monkeypatch, tmp_path, capsys):
    _mock(monkeypatch, [], stop_after=20)
    with pytest.raises(_Stop):
        _run(_args(tmp_path))
    cache = tmp_path / fit_restart.CACHE_FILE
    entries = [json.loads(line) for line in cache.read_text().splitlines()]
    entries[10]['x'][0] += 1e-6                # as if the optimiser asked for another point there
    cache.write_text("".join(json.dumps(e) + "\n" for e in entries))
    second = []
    _mock(monkeypatch, second)
    _run(_args(tmp_path))
    assert "its path differs from the earlier run" in capsys.readouterr().out
    state = _state(tmp_path)
    assert state['status'] == 'finished' and state['segments'][-1]['cached_evaluations'] == 10


@pytest.mark.parametrize("polish", [None, "nelder-mead"])
def test_fit_that_ran_out_of_evaluations_continues_with_a_larger_budget(monkeypatch, tmp_path, polish):
    reference = []
    _mock(monkeypatch, reference)
    _run(_args(tmp_path / "reference", max_evals=96, polish=polish))
    first = []
    _mock(monkeypatch, first)
    _run(_args(tmp_path / "fit", max_evals=48, polish=polish))
    assert _state(tmp_path / "fit")['result']['phases'][0]['budget_exhausted']
    _mock(monkeypatch, [])
    with pytest.raises(SystemExit, match="ran out of evaluations"):
        _run(_args(tmp_path / "fit", max_evals=48, polish=polish))     # the same budget: nothing to do
    second = []
    _mock(monkeypatch, second)
    _run(_args(tmp_path / "fit", max_evals=96, polish=polish))         # larger budget: the DE goes on
    assert _points(_summary(tmp_path / "fit")) == _points(_summary(tmp_path / "reference"))
    de_first = len([c for c in first]) if polish is None else 48
    assert len(second) == len(reference) - de_first


def test_sigterm_in_a_worker_process_only_exits(monkeypatch, tmp_path):
    """The simulation workers inherit the handler (and slurm signals every process): they must not touch the state."""
    run = fit_restart.prepare_output_dir(_args(tmp_path), fit_restart.settings_fingerprint(_args(tmp_path)))
    installed = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.setdefault(sig, handler))
    run.install_signal_handlers()
    exits = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))
    real_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)       # as in a forked worker
    installed[signal.SIGTERM](signal.SIGTERM, None)
    assert exits == [128 + signal.SIGTERM]
    assert _state(tmp_path)['status'] == 'running'                # untouched
