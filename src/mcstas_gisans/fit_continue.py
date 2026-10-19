"""
Start a Differential Evolution fit from the population of another fit (mg_fit --de_continue).

The evaluations of the earlier fit (its fit_summary.csv) are replayed with the update rules of scipy (immediate
updating) to recover its population at the point where it stopped; the new fit starts from that population (in its
own output directory). For an unfinished fit in the same output directory, the automatic continuation of
fit_restart retraces the earlier run exactly instead and needs no option.
"""
import csv
import os
from typing import Any, Dict, List, Tuple

import numpy as np


def replay_de_population(
    evaluations: Dict[int, Tuple[np.ndarray, float]],
    population_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover the population of a scipy Differential Evolution run from its evaluations, in the order made.

    scipy (updating='immediate', one worker) evaluates the initial population (members 0 .. S-1, member 0
    is x0), swaps the best member into position 0, and then makes one trial per member and generation:
    evaluation k (1-based, k > S) is the trial of member (k - S - 1) mod S; it replaces the member if its
    loss is lower or equal, and the best member is swapped into position 0 again whenever the trial is at
    least as good as member 0.

    Parameters
    ----------
    evaluations : dict
        eval_index (1-based) -> (parameter vector, loss). Evaluations without an entry (out-of-bounds
        points, penalised without a record) are never accepted.
    population_size : int
        The DE population size S.

    Returns
    -------
    population : np.ndarray
        Shape (S, N), in scipy's member order.
    energies : np.ndarray
        The losses of the members.
    """
    missing = [i for i in range(1, population_size + 1) if i not in evaluations]
    if missing:
        raise ValueError(f"The initial DE population is incomplete: no record of evaluation(s) {missing}")
    population = np.array([evaluations[i][0] for i in range(1, population_size + 1)], dtype=float)
    energies = np.array([evaluations[i][1] for i in range(1, population_size + 1)], dtype=float)

    def promote_lowest_energy():
        best = int(np.argmin(energies))
        energies[[0, best]] = energies[[best, 0]]
        population[[0, best], :] = population[[best, 0], :]

    promote_lowest_energy()
    for index in range(population_size + 1, max(evaluations) + 1):
        if index not in evaluations:
            continue
        x, energy = evaluations[index]
        candidate = (index - population_size - 1) % population_size
        if energy <= energies[candidate]:
            population[candidate] = x
            energies[candidate] = energy
            if energy <= energies[0]:
                promote_lowest_energy()
    return population, energies


def load_de_continuation(
    path: str,
    param_names: List[str],
    objective_key: str,
    population_size: int
) -> Dict[str, Any]:
    """
    Read the fit_summary.csv of an earlier DE fit (--de_continue) and recover its last population.

    Returns a dict with 'population' and 'energies' (see replay_de_population), 'records' (the earlier
    evaluations, numbers converted, in evaluation order) and 'last_eval_index'.
    """
    summary_path = os.path.join(path, "fit_summary.csv") if os.path.isdir(path) else path
    with open(summary_path, newline='') as f:
        rows = list(csv.DictReader(f))

    def number(text: str) -> Any:
        try:
            value = float(text)
        except (TypeError, ValueError):
            return text
        return int(value) if value.is_integer() and '.' not in text and 'e' not in text.lower() else value

    # the evaluation rows only (a finished fit's summary also holds the printed results table and runtimes)
    records = [{key: number(value) for key, value in row.items() if key is not None} for row in rows]
    records = [r for r in records if isinstance(r.get('eval_index'), int)
               and all(isinstance(r.get(name), (int, float)) for name in param_names)]
    records.sort(key=lambda r: r['eval_index'])
    absent = [name for name in param_names + [objective_key] if records and name not in records[0]]
    if not records or absent:
        raise ValueError(f"--de_continue {summary_path}: " + (f"columns {absent} missing (fit the same parameters with the "
                                                              f"same loss)" if records else "no evaluations"))
    evaluations = {}
    for r in records:
        if isinstance(r[objective_key], str):
            r[objective_key] = float('nan')
        loss = r[objective_key]
        loss = 1e9 if np.isnan(loss) else float(loss)  # as returned to the optimizer
        evaluations[int(r['eval_index'])] = (np.array([float(r[name]) for name in param_names]), loss)
    population, energies = replay_de_population(evaluations, population_size)
    return {'population': population, 'energies': energies, 'records': records,
            'last_eval_index': max(evaluations), 'summary_path': summary_path}


def known_population_objective(objective: Any, population: np.ndarray, energies: np.ndarray,
                               bounds: List[Tuple[float, float]]) -> Any:
    """
    The loss for a DE started from a given population: scipy evaluates that population first, member by member;
    these requests are answered with the known losses (compared in the unit cube of the bounds, as scipy rescales
    the population), all later ones by objective.
    """
    low = np.array([b[0] for b in bounds], dtype=float)
    span = np.array([b[1] - b[0] for b in bounds], dtype=float)
    known = list(zip(population, energies))

    def wrapped(x):
        if known:
            x_known, energy = known.pop(0)
            if np.allclose((np.asarray(x, dtype=float) - low) / span, (x_known - low) / span, rtol=0, atol=1e-9):
                return energy
            print("WARNING: --de_continue: scipy's initial population differs from the recovered one; simulating it")
            known.clear()
        return objective(x)
    return wrapped
