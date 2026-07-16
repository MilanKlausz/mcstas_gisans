"""
Tests for checking rotation consistency of position, velocity, and polarization components.
"""
import numpy as np
from mcstas_gisans.preconditioning import sample_orientation_transform, transform_to_sample_system

def test_rotation_consistency():
  # Create a dummy particle with distinct components
  # Format of particles inside preconditioning functions is (N, 12) if polarized:
  # p, x, y, z, vx, vy, vz, w, t, polx, poly, polz
  p = 1.0
  x, y, z = 1.2, 3.4, 5.6
  vx, vy, vz = 10.0, 20.0, 30.0
  w = 6.0
  t = 0.005
  pol_vec = np.array([0.1, 0.5, -0.8])
  pol_vec = pol_vec / np.linalg.norm(pol_vec)
  polx, poly, polz = pol_vec

  particles_12 = np.array([[p, x, y, z, vx, vy, vz, w, t, polx, poly, polz]])
  try:
    sample_orientation_transform(particles_12, 0)
    has_polarization = True
    particles = particles_12
  except ValueError:
    has_polarization = False
    particles = np.array([[p, x, y, z, vx, vy, vz, w, t]])

  # Test sample_orientation_transform for cases 0, 1, 2
  for orientation in [0, 1, 2]:
    transformed = sample_orientation_transform(particles, orientation)
    # sample_orientation_transform returns a tuple of 1D numpy arrays
    unpacked = [val[0] for val in transformed]
    if has_polarization:
      tp, tx, ty, tz, tvx, tvy, tvz, tw, tt, tpolx, tpoly, tpolz = unpacked
    else:
      tp, tx, ty, tz, tvx, tvy, tvz, tw, tt = unpacked

    # Scalars/Z-only components should not change
    assert np.isclose(tp, p)
    assert np.isclose(tz, z)
    assert np.isclose(tvz, vz)
    assert np.isclose(tw, w)
    assert np.isclose(tt, t)
    if has_polarization:
      assert np.isclose(tpolz, polz)

    # Check 2D rotation of X/Y components
    if orientation == 0:
      # +90 degrees rotation: (x,y) -> (-y, x)
      assert np.isclose(tx, -y)
      assert np.isclose(ty, x)
      assert np.isclose(tvx, -vy)
      assert np.isclose(tvy, vx)
      if has_polarization:
        assert np.isclose(tpolx, -poly)
        assert np.isclose(tpoly, polx)
    elif orientation == 1:
      # No rotation
      assert np.isclose(tx, x)
      assert np.isclose(ty, y)
      assert np.isclose(tvx, vx)
      assert np.isclose(tvy, vy)
      if has_polarization:
        assert np.isclose(tpolx, polx)
        assert np.isclose(tpoly, poly)
    elif orientation == 2:
      # -90 degrees rotation: (x,y) -> (y, -x)
      assert np.isclose(tx, y)
      assert np.isclose(ty, -x)
      assert np.isclose(tvx, vy)
      assert np.isclose(tvy, -vx)
      if has_polarization:
        assert np.isclose(tpolx, poly)
        assert np.isclose(tpoly, -polx)

  # Test transform_to_sample_system for different orientations and alpha incident angles
  for orientation in [0, 1, 2]:
    for alpha in [-1.5, 0.0, 0.24, 1.0, 5.0]:
      transformed_sys = transform_to_sample_system(particles, alpha, orientation, 0.0)
      unpacked_sys = transformed_sys[0]
      if has_polarization:
        tp, tx, ty, tz, tvx, tvy, tvz, tw, tt, tpolx, tpoly, tpolz = unpacked_sys
      else:
        tp, tx, ty, tz, tvx, tvy, tvz, tw, tt = unpacked_sys

      # First apply sample orientation transform to get intermediate state
      after_orientation = sample_orientation_transform(particles, orientation)
      unpacked_after = [val[0] for val in after_orientation]
      if has_polarization:
        _, ox, oy, oz, ovx, ovy, ovz, _, _, opolx, opoly, opolz = unpacked_after
      else:
        _, ox, oy, oz, ovx, ovy, ovz, _, _ = unpacked_after

      # Rotation matrix components
      alpha_rad = np.deg2rad(alpha)
      cos_a = np.cos(-alpha_rad)
      sin_a = np.sin(-alpha_rad)
      rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

      # Verify coordinate rotation of Z/Y components under alpha_inc
      expected_z_y = np.dot(rotation_matrix, [oz, oy])
      assert np.isclose(tz, expected_z_y[0])
      assert np.isclose(ty, expected_z_y[1])

      expected_vz_vy = np.dot(rotation_matrix, [ovz, ovy])
      assert np.isclose(tvz, expected_vz_vy[0])
      assert np.isclose(tvy, expected_vz_vy[1])

      if has_polarization:
        expected_polz_poly = np.dot(rotation_matrix, [opolz, opoly])
        assert np.isclose(tpolz, expected_polz_poly[0])
        assert np.isclose(tpoly, expected_polz_poly[1])

      # The X components are not affected by transform_to_sample_system's alpha rotation
      assert np.isclose(tx, ox)
      assert np.isclose(tvx, ovx)
      if has_polarization:
        assert np.isclose(tpolx, opolx)

  print("All rotation consistency checks passed successfully!")
  
def test_declination_no_rotation():
  # Create a dummy particle
  p = 1.0
  x, y, z = 1.2, 3.4, 5.6
  vx, vy, vz = 10.0, 20.0, 30.0
  w = 6.0
  t = 0.005
  particles = np.array([[p, x, y, z, vx, vy, vz, w, t]])

  # If alpha_inc_deg == beam_declination_angle, rotation angle should be 0.0
  # Thus, transform_to_sample_system should only apply sample_orientation_transform
  # (which for orientation 1 is no rotation at all).
  alpha = 0.44
  beam_declination = 0.44
  transformed = transform_to_sample_system(particles, alpha, 1, beam_declination)
  unpacked = transformed[0]
  
  assert np.isclose(unpacked[1], x)
  assert np.isclose(unpacked[2], y)
  assert np.isclose(unpacked[3], z)
  assert np.isclose(unpacked[4], vx)
  assert np.isclose(unpacked[5], vy)
  assert np.isclose(unpacked[6], vz)
  print("Declination rotation cancellation check passed successfully!")

if __name__ == "__main__":
  test_rotation_consistency()
  test_declination_no_rotation()
