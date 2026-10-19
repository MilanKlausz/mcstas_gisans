"""
Run state of an mg_fit output directory: automatic continuation of an unfinished fit, --overwrite, and the
evaluation cache that lets an interrupted optimisation retrace its path without simulating again.

An output directory holds one fit. Next to fit_summary.csv, mg_fit keeps

- fit_state.json: the settings of the fit (every option that can change its result), its status (running,
  interrupted, failed, finished) and one entry per run segment (start, end, host, slurm job, evaluations);
- fit_evaluations.jsonl: every evaluation in the order made (phase, the exact parameter vector passed by the
  optimiser, the loss returned to it, the summary record).

When mg_fit is started with an --output_dir that holds an unfinished fit with the same settings, it continues it:
the optimiser is started again exactly as before, and the evaluations it requests in the same order with the
same parameter vectors are answered from fit_evaluations.jsonl instead of being simulated. The scipy optimisers
used (differential evolution with a seeded generator, Nelder-Mead, Powell) are deterministic given the losses they
receive, so they retrace their earlier path exactly and then go on with new evaluations; the result is the same as
that of an uninterrupted run. The simulation itself need not be deterministic for this: the cached losses are used.
If a requested vector differs from the cached one (the path diverged, e.g. after a scipy update), the cache is
cut there and the fit goes on with simulations from that point (with a warning).

A finished fit, or an unfinished one with other settings, is not touched: mg_fit stops with a message. --overwrite
starts the fit from scratch (it removes the three files above; the comparison plots are overwritten as they are
made).
"""
import datetime
import json
import os
import platform
import signal
import socket
import sys
from typing import Any, Dict, List, Optional

import numpy as np

STATE_FILE = "fit_state.json"
CACHE_FILE = "fit_evaluations.jsonl"
SUMMARY_FILE = "fit_summary.csv"

# Options that do not change the result of a fit: the evaluation budgets (a continued fit may get a larger one),
# parallelisation, plotting and bookkeeping. Every other option is part of the settings of a fit.
NON_RESULT_OPTIONS = {
    'output_dir', 'max_evals', 'polish_max_evals', 'parallel_processes', 'no_parallel', 'bornagain_number_of_threads',
    'verbose', 'png', 'gif', 'split_view', 'y_plot_range', 'z_plot_range', 'intensity_min', 'mask_view',
    'mask_view_style', 'overwrite', 'seed_given',
}
# File options compared by file name only (a fit continued on another machine has other paths)
PATH_OPTIONS = {'filename', 'filename2', 'nxs', 'nxs2', 'model', 'model2', 'intensity_factor_spectrum'}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


def _builtin(value: Any) -> Any:
    """JSON-compatible form of an option value (numpy scalars and arrays, tuples)."""
    if isinstance(value, (list, tuple)):
        return [_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _builtin(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return f"<array {value.shape}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return f"<{type(value).__name__}>"


def _file_names(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_file_names(v) for v in value]
    return os.path.basename(value) if isinstance(value, str) and value else value


def settings_fingerprint(args: Any) -> Dict[str, Any]:
    """
    The settings of a fit: every option except those in NON_RESULT_OPTIONS and private attributes, file options
    by file name. Computed right after parsing the command line (before mg_fit adds derived attributes).
    """
    settings = {}
    for key, value in sorted(vars(args).items()):
        if key.startswith('_') or key in NON_RESULT_OPTIONS:
            continue
        settings[key] = _builtin(_file_names(value) if key in PATH_OPTIONS else value)
    return settings


def _differences(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    keys = sorted(set(old) | set(new))
    return [f"  {key}: {old.get(key, '(not set)')!r} -> {new.get(key, '(not set)')!r}" for key in keys if old.get(key) != new.get(key)]


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"  # one per process
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=1, default=_builtin)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class EvaluationCache:
    """
    The evaluations of a fit in the order made (fit_evaluations.jsonl), answered again in that order.

    next_cached() returns the cached evaluation for a request if it is the next one of the cached sequence (same
    phase, exactly the same parameter vector); the first request that differs ends the replay, and the file is cut
    to the replayed part, so that it always describes one optimisation path.
    """

    def __init__(self, path: str, entries: Optional[List[Dict[str, Any]]] = None):
        self.path = path
        self.entries = entries or []
        self.position = 0
        self.diverged = False

    @classmethod
    def load(cls, path: str) -> "EvaluationCache":
        entries = []
        if os.path.exists(path):
            with open(path) as f:
                for number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:  # a line cut by a kill while it was written: the evaluation is redone
                        print(f"Evaluation cache: line {number} of {path} is incomplete and is ignored (its evaluation is made again)")
                        break
        return cls(path, entries)

    @property
    def remaining(self) -> int:
        return len(self.entries) - self.position

    def next_cached(self, phase: str, x: Any) -> Optional[Dict[str, Any]]:
        if self.position >= len(self.entries):
            return None
        entry = self.entries[self.position]
        if entry['phase'] == phase and entry['x'] == [float(v) for v in x]:
            self.position += 1
            return entry
        self.diverged = True
        print(f"Evaluation cache: the optimiser requested another point than the cached evaluation #{entry['eval_index']} "
              f"(phase {entry['phase']}); its path differs from the earlier run from here on. This is expected when a larger "
              f"budget lets an earlier phase go on (the later phase is then made again); otherwise (e.g. a changed scipy or "
              f"numpy version) it is a WARNING. The remaining {self.remaining} cached evaluations are discarded and the fit goes "
              f"on with simulations.")
        self._truncate()
        return None

    def _truncate(self) -> None:
        self.entries = self.entries[:self.position]
        tmp = self.path + ".tmp"
        with open(tmp, 'w') as f:
            for entry in self.entries:
                f.write(json.dumps(entry, default=_builtin) + "\n")
        os.replace(tmp, self.path)

    def append(self, entry: Dict[str, Any]) -> None:
        if self.position < len(self.entries):  # a new evaluation while cached ones are left: should not happen
            self._truncate()
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry, default=_builtin) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.entries.append(entry)
        self.position = len(self.entries)


class FitRun:
    """The state of the fit in an output directory for this mg_fit run (see the module docstring)."""

    def __init__(self, output_dir: str, state: Dict[str, Any], cache: EvaluationCache, continued: bool):
        self.output_dir = output_dir
        self.state = state
        self.cache = cache
        self.continued = continued
        self.cached_used = 0
        self.new_evaluations = 0
        self._terminated = False

    @property
    def segment(self) -> Dict[str, Any]:
        return self.state['segments'][-1]

    def save(self) -> None:
        self.state['last_update'] = _now()
        self.segment.update(cached_evaluations=self.cached_used, new_evaluations=self.new_evaluations)
        _write_json_atomic(os.path.join(self.output_dir, STATE_FILE), self.state)

    def record(self, entry: Dict[str, Any], cached: bool) -> None:
        """Count an evaluation (cached or new); new ones are appended to the cache."""
        if cached:
            self.cached_used += 1
        else:
            self.cache.append(entry)
            self.new_evaluations += 1
        self.save()

    def finish(self, status: str, message: str = '', result: Optional[Dict[str, Any]] = None) -> None:
        self.state['status'] = status
        self.segment.update(end=_now(), end_status=status, message=message)
        if result is not None:
            self.state['result'] = result
        self.save()
        print(f"Fit state: {status}" + (f" ({message})" if message else '') + f" -> {os.path.join(self.output_dir, STATE_FILE)}")

    def install_signal_handlers(self) -> None:
        """
        SIGTERM (slurm at the time limit, scancel) and SIGUSR1: record 'interrupted' and stop. Only the main process
        does that: the worker processes of the simulation inherit the handler (and slurm signals every process of the
        job), and they just exit.
        """
        main_pid = os.getpid()

        def handler(signum, frame):
            if os.getpid() != main_pid:
                os._exit(128 + signum)
                return
            if self._terminated:
                return
            self._terminated = True
            print(f"\nSignal {signal.Signals(signum).name} received: stopping after {self.new_evaluations} new evaluations "
                  f"of this run (everything evaluated so far is kept; run the same command again to continue).")
            self.finish('interrupted', f"signal {signal.Signals(signum).name}")
            sys.exit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # not in the main thread (tests)
                pass

    def print_retrace_done(self) -> None:
        if self.continued and self.cached_used and not getattr(self, '_retrace_reported', False):
            self._retrace_reported = True
            print(f"--- Continuation: {self.cached_used} evaluations retraced from the cache; new evaluations from here on ---")


def _remove_fit_files(output_dir: str) -> None:
    for name in (STATE_FILE, CACHE_FILE, SUMMARY_FILE):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"  removed {path}")


def prepare_output_dir(args: Any, settings: Dict[str, Any]) -> FitRun:
    """
    Decide what to do with --output_dir and print the decision: start a new fit, continue an unfinished one with
    the same settings, or stop (finished fit, other settings, or a summary without state from an older mg_fit).
    With --overwrite the earlier fit files are removed and a new fit starts. Sets args.seed to the stored seed when
    an unfinished fit is continued without an explicit --seed.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, STATE_FILE)
    summary_path = os.path.join(output_dir, SUMMARY_FILE)
    print(f"\n========== Output directory: {output_dir} ==========")
    previous = None
    if getattr(args, 'overwrite', False):
        print("--overwrite: any earlier fit in this directory is replaced")
        _remove_fit_files(output_dir)
    elif os.path.exists(state_path):
        with open(state_path) as f:
            previous = json.load(f)
        segments = previous.get('segments', [])
        last = segments[-1] if segments else {}
        done = sum(s.get('new_evaluations', 0) for s in segments)
        print(f"Found a fit: status {previous.get('status')!r}, started {previous.get('created')}, {len(segments)} run segment(s), "
              f"{done} evaluations; last run on {last.get('host')} (slurm job {last.get('slurm_job_id')}), last update {previous.get('last_update')}")
        if previous.get('status') == 'finished':
            result = previous.get('result', {})
            exhausted = [p['name'] for p in result.get('phases', []) if p.get('budget_exhausted')]
            larger = [name for name, old in (('max_evals', last.get('max_evals')), ('polish_max_evals', last.get('polish_max_evals')))
                      if old is not None and getattr(args, name, None) is not None and getattr(args, name) > old]
            if exhausted and larger:
                print(f"The fit FINISHED by running out of evaluations ({', '.join(exhausted)}); with the larger "
                      f"{' and '.join('--' + n for n in larger)} it is continued from where it stopped")
            else:
                hint = (f" It ran out of evaluations ({', '.join(exhausted)}): give a larger --max_evals/--polish_max_evals than "
                        f"before ({last.get('max_evals')}/{last.get('polish_max_evals')}) to continue it." if exhausted else '')
                raise SystemExit(f"\nThe fit in {output_dir} is FINISHED ({result.get('message', '')}; best {result.get('objective', 'loss')} "
                                 f"{result.get('best_loss')} at evaluation #{result.get('best_eval_index')}). Nothing to do.{hint} To run "
                                 f"it again from scratch use --overwrite, or choose another --output_dir.")
        old_settings = dict(previous.get('settings', {}))
        new_settings = dict(settings)
        if old_settings.get('seed') != new_settings.get('seed') and not getattr(args, 'seed_given', True):
            print(f"Seed: no --seed given; the seed of the fit, {old_settings.get('seed')}, is used")
            args.seed = old_settings.get('seed')
            new_settings['seed'] = args.seed
            settings['seed'] = args.seed
        differences = _differences(old_settings, new_settings)
        if differences:
            raise SystemExit(f"\nThe unfinished fit in {output_dir} has other settings than this command:\n" + "\n".join(differences)
                             + "\nContinue it with the same settings (only the budgets --max_evals/--polish_max_evals and the parallel and "
                               "plot options may change), start it again with --overwrite, or choose another --output_dir.")
        if previous.get('status') == 'running':
            print("Note: its status is 'running': the earlier run ended without recording it (killed, node failure). Make sure "
                  "that no other job is still writing to this directory.")
    elif os.path.exists(summary_path):
        raise SystemExit(f"\n{output_dir} holds a {SUMMARY_FILE} without {STATE_FILE} (a fit of an older mg_fit, or a scan). It is not "
                         f"continued automatically: use --overwrite to replace it, choose another --output_dir, or start a new "
                         f"DE fit from its population with --de_continue {summary_path} --output_dir <new dir>.")
    if previous is None:
        cache = EvaluationCache(os.path.join(output_dir, CACHE_FILE))
        open(cache.path, 'w').close()
        state = {'created': _now(), 'settings': settings, 'segments': [],
                 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': _scipy_version()}}
        print("Starting a new fit")
    else:
        cache = EvaluationCache.load(os.path.join(output_dir, CACHE_FILE))
        state = previous
        env = state.get('environment', {})
        if env.get('scipy') != _scipy_version() or env.get('numpy') != np.__version__:
            print(f"WARNING: the fit was started with scipy {env.get('scipy')}, numpy {env.get('numpy')}; now scipy {_scipy_version()}, "
                  f"numpy {np.__version__}: the optimiser may not retrace its path (the evaluations are then simulated again)")
        print(f"CONTINUING the unfinished fit: {len(cache.entries)} cached evaluations will be retraced without simulation, "
              f"then the fit goes on with new evaluations")
    state['status'] = 'running'
    state['segments'].append({'start': _now(), 'host': socket.gethostname(), 'pid': os.getpid(),
                              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'cached_available': len(cache.entries),
                              'max_evals': getattr(args, 'max_evals', None), 'polish_max_evals': getattr(args, 'polish_max_evals', None)})
    run = FitRun(output_dir, state, cache, continued=previous is not None)
    run.save()
    print(f"Fit files: {os.path.join(output_dir, SUMMARY_FILE)}, {cache.path}, {state_path}")
    print("=" * 60)
    return run


def _scipy_version() -> str:
    import scipy
    return scipy.__version__
