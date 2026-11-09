"""
The optimisers of mg_fit (scipy.optimize): differential evolution (global search within the bounds), Nelder-Mead
and Powell (local, derivative-free), and the polish of a differential-evolution result with a local optimiser.

Global search followed by a local refinement of its best point is the usual procedure for fits whose loss has
several minima: differential evolution finds the right minimum, but it stops when the losses of its population
agree (--fatol), which can leave its best point a few standard errors from the bottom of that minimum, especially
in narrow, correlated valleys. The local optimiser then refines that point (--polish). Derivative-free local
optimisers are used because the loss comes from a Monte Carlo simulation: with a fixed --seed it is a smooth
function of the parameters on the scale of physical parameter changes, but not on the scale of the tiny steps of
finite-difference gradients (scipy's own polish uses L-BFGS-B with such gradients).

All functions take the loss as a function of the full parameter vector (in parameter units) and are deterministic
for a given --seed and given losses, which the continuation of an interrupted fit relies on (fit_restart).
"""
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

Bounds = Sequence[Tuple[Optional[float], Optional[float]]]


def de_population_size(n_parameters: int, popsize: int) -> int:
    """scipy uses at least 5 population members."""
    return max(5, popsize * n_parameters)


def run_differential_evolution(objective: Callable, bounds: Bounds, x0: Sequence[float], integrality: List[bool],
                               popsize: int, max_evals: int, fatol: float, seed: Optional[int],
                               init: Optional[np.ndarray] = None, stream: int = 0,
                               batch_map: Optional[Callable] = None) -> Any:
    """
    Differential evolution within the bounds, started with x0 in the initial population (or from the population
    init, see fit_continue). It evaluates the initial population and then one generation of popsize x
    (number of parameters) trial points after the other; it stops when the standard deviation of the losses of
    its population is below fatol (absolute; tol=0 disables scipy's relative criterion), or after the generations
    that fit into max_evals. It has no parameter-space criterion. The generator is seeded with seed (and stream,
    for a start from another fit's population), so the sequence of trial points is reproducible.

    batch_map (mg_fit --de_parallel): a map-like callable batch_map(func, points) that evaluates the trial points of a
    generation together; scipy then updates the population once per generation ("deferred" updating) instead of after
    every trial point ("immediate", the default), which gives other, equally reproducible trial points.
    """
    import scipy.optimize
    population = de_population_size(len(bounds), popsize)
    if max_evals < population:
        print(f"WARNING: --max_evals {max_evals} is smaller than the initial DE population ({population}); "
              f"the initial population alone will be evaluated.")
    # a start from a given population (--de_continue) needs no evaluation of it: its losses are known
    maxiter = max(0, max_evals // population - (0 if init is not None else 1))
    generations = maxiter + (0 if init is not None else 1)
    print(f"Differential Evolution: population {population}, at most {generations} generations ({generations * population} "
          f"evaluations); stops when the losses of the population agree within {fatol} (standard deviation)")
    # A Generator, because scipy seeds the legacy RandomState from an int seed, which must be below 2**32, while
    # the default --seed is a random 63-bit number
    rng = (np.random.default_rng([seed, stream] if init is not None else seed)) if seed is not None else None
    return scipy.optimize.differential_evolution(
        objective, bounds, x0=None if init is not None else x0, init=init if init is not None else 'latinhypercube',
        maxiter=maxiter, popsize=popsize, tol=0.0, atol=fatol, integrality=integrality, polish=False, seed=rng,
        **({'updating': 'deferred', 'workers': batch_map} if batch_map is not None else {}))


def run_local(objective: Callable, x0: Sequence[float], bounds: Bounds, method: str, max_evals: int, xatol: float,
              fatol: float, integer_mask: Optional[List[bool]] = None) -> Tuple[Any, np.ndarray]:
    """
    Nelder-Mead or Powell from x0, in scaled coordinates u = (x - x0) / scale with scale the bound range (or the absolute value of x0,
    or 1): all parameters are O(1), so xatol is relative to the bound ranges and tiny-valued parameters (SLDs
    ~1e-6) are handled like any other. Nelder-Mead starts with a simplex of 10 % of the scale per parameter (at
    least one unit for integer parameters) and stops when the parameters change by less than xatol and the loss
    by less than fatol; Powell uses xtol/ftol. Returns the scipy result and its best point in parameter units.
    """
    import scipy.optimize
    integer_mask = integer_mask or [False] * len(x0)
    x0_arr = np.asarray(x0, dtype=float)
    scale = np.array([(high - low) if (low is not None and high is not None) else (abs(v) if v != 0 else 1.0)
                      for v, (low, high) in zip(x0_arr, bounds)])

    def to_x(u):
        return x0_arr + scale * np.asarray(u)
    u_bounds = [((low - v) / sc if low is not None else None, (high - v) / sc if high is not None else None)
                for v, sc, (low, high) in zip(x0_arr, scale, bounds)]
    has_bounds = any(b is not None for pair in u_bounds for b in pair)
    method = 'nelder-mead' if method.lower() == 'nelder-mead' else 'powell'
    options = {'maxiter': max_evals, 'maxfev': max_evals}
    if method == 'nelder-mead':
        options['xatol'] = xatol
        options['fatol'] = fatol
        steps = [max(0.1, 1.0 / sc) if integer else 0.1 for integer, sc in zip(integer_mask, scale)]
        simplex = [np.zeros(len(x0_arr))]
        for i, step in enumerate(steps):
            vertex = np.zeros(len(x0_arr))
            high = u_bounds[i][1]
            vertex[i] = step if (high is None or step <= high) else -step
            simplex.append(vertex)
        options['initial_simplex'] = np.array(simplex)
    else:
        options['xtol'] = xatol
        options['ftol'] = fatol
    result = scipy.optimize.minimize(lambda u: objective(to_x(u)), np.zeros(len(x0_arr)), method=method,
                                     options=options, bounds=u_bounds if has_bounds else None)
    return result, to_x(result.x)


def polish(objective: Callable, x_start: Sequence[float], bounds: Bounds, integer_mask: List[bool], method: str,
           max_evals: int, xatol: float, fatol: float) -> Tuple[Any, np.ndarray]:
    """
    Local refinement (run_local) of the continuous parameters from x_start (the differential-evolution best), with
    the integer parameters fixed at their values there. Returns the scipy result and the best full parameter vector.
    """
    x_full = np.array(x_start, dtype=float)
    for i, integer in enumerate(integer_mask):
        if integer:
            x_full[i] = np.round(x_full[i])
    free = [i for i, integer in enumerate(integer_mask) if not integer]
    if not free:
        raise ValueError("--polish: all fitted parameters are integers, nothing to polish")

    def reduced_objective(x_free):
        x = x_full.copy()
        x[free] = x_free
        return objective(x)
    result, best_free = run_local(reduced_objective, x_full[free], [bounds[i] for i in free], method, max_evals, xatol, fatol)
    best = x_full.copy()
    best[free] = best_free
    return result, best
