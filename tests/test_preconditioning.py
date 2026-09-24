import numpy as np
import pytest

from mcstas_gisans.preconditioning import apply_t0_correction, transform_to_bornagain_coordinate_system


def _make_particles(t_values):
    """Build a minimal particles array: columns p,x,y,z,vx,vy,vz,w,t."""
    n = len(t_values)
    particles = np.zeros((n, 9))
    particles[:, 7] = 1.0  # weight
    particles[:, 8] = t_values
    return particles


def _make_particles_with_velocity(vy, vz, n=5):
    """Build a minimal particles array with a fixed (vy, vz) velocity, columns p,x,y,z,vx,vy,vz,w,t."""
    particles = np.zeros((n, 9))
    particles[:, 0] = 1.0  # weight (p)
    particles[:, 5] = vy
    particles[:, 6] = vz
    return particles


def test_beam_angle_defaults_to_zero_not_mcpl_estimate(capsys):
    """
    When no beam_angle override is given, the effective beam angle used for the
    simulation must default to 0.0, not silently fall back to the value estimated
    from the MCPL file's particle velocities. Consistency with mg_beam_centre_correction
    (which has no MCPL data and always defaults to 0.0) requires this.
    """
    # vy/vz corresponds to a ~1 deg declination, which should be ignored as a fallback.
    particles = _make_particles_with_velocity(vy=np.tan(np.deg2rad(1.0)), vz=1.0)
    _, actual_beam_angle = transform_to_bornagain_coordinate_system(particles, alpha_inc_deg=0.0, sample_orientation=1, beam_angle=None)

    assert actual_beam_angle == 0.0
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_beam_angle_explicit_value_is_honored_and_not_overridden(capsys):
    """An explicitly provided beam_angle must be used as-is, even when it disagrees with the MCPL estimate."""
    particles = _make_particles_with_velocity(vy=np.tan(np.deg2rad(1.0)), vz=1.0)
    _, actual_beam_angle = transform_to_bornagain_coordinate_system(particles, alpha_inc_deg=0.0, sample_orientation=1, beam_angle=0.44)

    assert actual_beam_angle == 0.44
    captured = capsys.readouterr()
    assert "WARNING" in captured.out  # 0.44 vs ~1.0 deg estimate differ by more than the threshold


def test_beam_angle_no_warning_when_estimate_agrees(capsys):
    """No warning should be printed when the MCPL estimate agrees with the beam angle actually used."""
    particles = _make_particles_with_velocity(vy=np.tan(np.deg2rad(0.44)), vz=1.0)
    _, actual_beam_angle = transform_to_bornagain_coordinate_system(particles, alpha_inc_deg=0.0, sample_orientation=1, beam_angle=0.44)

    assert actual_beam_angle == 0.44
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


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


def _particles(z, vz, x=0.0, y=0.0, vx=600.0):
    # columns p, x, y, z, vx, vy, vz, w, t
    return np.array([[1.0, x, y, z, vx, 0.0, vz, 6.0, 0.001]])


def test_particle_hitting_the_sample_is_propagated_onto_the_surface():
    from mcstas_gisans.preconditioning import propagate_to_sample_surface
    from mcstas_gisans.sample import Sample
    # starts 2 mm above the surface, 3 cm upstream, descending at ~5.7 deg: hits 1 cm upstream of the centre
    out = propagate_to_sample_surface(_particles(z=0.002, vz=-60.0, x=-0.03), 0.1, 0.1, allow_sample_miss=False)
    assert out.shape[0] == 1
    assert out[0, 3] == 0.0
    assert out[0, 1] == pytest.approx(-0.03 + 600.0 * 0.002 / 60.0)  # x moved by vx * t
    assert out[0, 8] == pytest.approx(0.001 + 0.002 / 60.0)          # t increased by the flight time
    assert not Sample(0.1, 0.1, 'silica_100nm_air', None).sample_missed(out[0, 1], out[0, 2], out[0, 3], out[0, 6])


@pytest.mark.parametrize("z, vz", [(0.002, -1e-6), (0.002, 0.0), (-0.001, -1.0), (0.002, 1.0)])
def test_missed_particles_are_not_moved(z, vz):
    """A grazing particle (vz ~ 0) that misses the sample must not be pushed to the far-away z=0 crossing."""
    from mcstas_gisans.preconditioning import propagate_to_sample_surface
    from mcstas_gisans.sample import Sample
    particles = _particles(z=z, vz=vz)
    out = propagate_to_sample_surface(particles.copy(), 0.1, 0.1, allow_sample_miss=True)
    np.testing.assert_array_equal(out, particles)
    assert Sample(0.1, 0.1, 'silica_100nm_air', None).sample_missed(out[0, 1], out[0, 2], out[0, 3], out[0, 6])
