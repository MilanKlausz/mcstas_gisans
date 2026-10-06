"""
Tests for fit.fit_flat_background (--fit_background).

The expected values come from the construction of the data, not from the implementation: Poisson
counts drawn from a known signal plus a known flat background must give back that background,
a background-free measurement must give ~0, and masked (NaN) bins must not take part.
"""
import numpy as np
import pytest

from mcstas_gisans.fit import fit_flat_background

RNG = np.random.default_rng(20261006)


def _pattern(shape=(128, 256)):
  """A scattering-like pattern: a few peaks on a falling slope, from ~0 to a few hundred counts."""
  y, z = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing='ij')
  signal = 300 * np.exp(-((np.abs(y) - 0.3) / 0.05) ** 2) * np.exp(-3 * z) + 20 * np.exp(-5 * np.abs(y))
  return signal


@pytest.mark.parametrize("true_background", [1.0, 6.1, 25.0])
def test_recovers_the_true_background(true_background):
  signal = _pattern()
  counts = RNG.poisson(signal + true_background).astype(float)
  background, loss = fit_flat_background(counts, signal, np.zeros_like(signal), 'poisson_deviance')
  # statistical precision: ~sqrt(b / n_bins) for the background-dominated bins; 0.1 is generous
  assert background == pytest.approx(true_background, abs=0.1)
  # a correct model gives ~1 per bin (slightly above 1 at a mean of ~1 count per bin)
  assert loss == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("true_background", [1.0, 6.1, 25.0])
def test_reduced_chi2_overestimates_the_background(true_background):
  """Minimising the Pearson chi^2 (variance m in the denominator) inflates m: the background comes out
  ~0.5 counts too high at any level (documented in --fit_background; use the Poisson deviance)."""
  signal = _pattern()
  counts = RNG.poisson(signal + true_background).astype(float)
  background, _ = fit_flat_background(counts, signal, np.zeros_like(signal), 'reduced_chi2')
  assert background - true_background == pytest.approx(0.5, abs=0.2)


def test_zero_background_is_found_at_the_bound():
  signal = _pattern() + 2.0
  counts = RNG.poisson(signal).astype(float)
  background, _ = fit_flat_background(counts, signal, np.zeros_like(signal), 'poisson_deviance')
  assert background < 0.05


def test_masked_bins_are_ignored():
  signal = _pattern()
  counts = RNG.poisson(signal + 6.1).astype(float)
  counts[:, :50] = 1e6                       # garbage that would dominate any fit if it were used...
  counts[:, :50] = np.nan                    # ...but is masked (NaN), as in run_simulation_evaluation
  signal_masked = signal.copy()
  signal_masked[:10, :] = np.nan             # bins masked on the simulation side as well
  background, _ = fit_flat_background(counts, signal_masked, np.zeros_like(signal), 'poisson_deviance')
  assert background == pytest.approx(6.1, abs=0.1)


def test_too_intense_pattern_gets_a_too_low_background():
  """The fitted level is conditional on the pattern (documented in --fit_background): a pattern 1.5x too
  intense leaves less room for the background."""
  signal = _pattern()
  counts = RNG.poisson(signal + 6.1).astype(float)
  background, _ = fit_flat_background(counts, 1.5 * signal, np.zeros_like(signal), 'poisson_deviance')
  assert background < 6.1 - 0.5


def test_no_valid_bins():
  nan = np.full((4, 4), np.nan)
  background, loss = fit_flat_background(nan, nan, np.zeros((4, 4)), 'poisson_deviance')
  assert background == 0.0 and np.isnan(loss)


def _validated(extra):
  from mcstas_gisans.fit import create_fit_parser, validate_fit_args
  parser = create_fit_parser()
  args = parser.parse_args(['events.mcpl', '--instrument', 'd22', '--nxs', 'meas.nxs', '--experiment_time', '100',
                            '--fit', 'radius', '50', '40', '60'] + extra)
  validate_fit_args(args, parser)
  return args


@pytest.mark.parametrize("fixed", [['--background', '6.1'], ['--background', '0'], ['--background2', '3']])
def test_fit_background_cannot_be_combined_with_a_fixed_background(fixed, capsys):
  with pytest.raises(SystemExit):
    _validated(['--fit_background'] + fixed)
  assert 'cannot be combined with a fixed --background' in capsys.readouterr().err


def test_background_options_alone_are_accepted():
  assert _validated(['--fit_background']).fit_background
  assert _validated(['--background', '6.1']).background == 6.1
  assert _validated([]).background is None   # not set: no background (0) is added
