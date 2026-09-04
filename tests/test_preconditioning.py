import numpy as np
import pytest

from mcstas_gisans.preconditioning import apply_t0_correction


def _make_particles(t_values):
    """Build a minimal particles array: columns p,x,y,z,vx,vy,vz,w,t."""
    n = len(t_values)
    particles = np.zeros((n, 9))
    particles[:, 7] = 1.0  # weight
    particles[:, 8] = t_values
    return particles


def test_t0_fixed_zero_is_honored_not_treated_as_unset(capsys):
    """
    args.t0_fixed=0.0 is a legitimate, explicit "no correction offset"
    value and must be honored, not treated as falsy/unset and silently
    replaced by monitor-based T0 correction (which requires an MCPL
    filepath / McStas monitor files that don't exist in this test and
    would raise if that branch were mistakenly taken).
    """
    class Args:
        t0_fixed = 0.0

    particles = _make_particles([100.0, 200.0])
    result = apply_t0_correction(particles, Args())

    assert np.array_equal(result[:, 8], [100.0, 200.0]), "t0_fixed=0.0 should leave TOF values unchanged"
    captured = capsys.readouterr()
    assert "McStas monitor" not in captured.out


def test_t0_fixed_nonzero_is_subtracted():
    class Args:
        t0_fixed = 15.0

    particles = _make_particles([100.0, 200.0])
    result = apply_t0_correction(particles, Args())

    assert np.array_equal(result[:, 8], [85.0, 185.0])
