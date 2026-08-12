"""
Tests for checking rotation consistency of position, velocity, and polarization components.
"""
import numpy as np
from mcstas_gisans.preconditioning import sample_orientation_transform, transform_to_bornagain_coordinate_system

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

    # Scalars should not change
    assert np.isclose(tp, p)
    assert np.isclose(tw, w)
    assert np.isclose(tt, t)

    # In BornAgain coordinates, x_ba is forward (which was Z in NeXus)
    assert np.isclose(tx, z)
    assert np.isclose(tvx, vz)
    if has_polarization:
      assert np.isclose(tpolx, polz)

    # Check 2D rotation of Y (left) and Z (up) components, which were X and Y in NeXus
    if orientation == 0:
      # -90 degrees rotation: (x_nx, y_nx) -> (-y_nx, x_nx)
      # Y_ba is left (-y_nx)
      # Z_ba is up (x_nx)
      assert np.isclose(ty, -y)
      assert np.isclose(tz, x)
      assert np.isclose(tvy, -vy)
      assert np.isclose(tvz, vx)
      if has_polarization:
        assert np.isclose(tpoly, -poly)
        assert np.isclose(tpolz, polx)
    elif orientation == 1:
      # No rotation: Y_ba is x_nx, Z_ba is y_nx
      assert np.isclose(ty, x)
      assert np.isclose(tz, y)
      assert np.isclose(tvy, vx)
      assert np.isclose(tvz, vy)
      if has_polarization:
        assert np.isclose(tpoly, polx)
        assert np.isclose(tpolz, poly)
    elif orientation == 2:
      # +90 degrees rotation: (x_nx, y_nx) -> (y_nx, -x_nx)
      assert np.isclose(ty, y)
      assert np.isclose(tz, -x)
      assert np.isclose(tvy, vy)
      assert np.isclose(tvz, -vx)
      if has_polarization:
        assert np.isclose(tpoly, poly)
        assert np.isclose(tpolz, -polx)

  # Test transform_to_sample_system for different orientations and alpha incident angles
  for orientation in [0, 1, 2]:
    for alpha in [-1.5, 0.0, 0.24, 1.0, 5.0]:
      transformed_sys = transform_to_bornagain_coordinate_system(particles, alpha, orientation, 0.0)
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

      # Verify coordinate rotation of X/Z components under alpha_inc
      expected_x_z = np.dot(rotation_matrix, [ox, oz])
      assert np.isclose(tx, expected_x_z[0])
      assert np.isclose(tz, expected_x_z[1])
      assert np.isclose(ty, oy)

      expected_vx_vz = np.dot(rotation_matrix, [ovx, ovz])
      assert np.isclose(tvx, expected_vx_vz[0])
      assert np.isclose(tvz, expected_vx_vz[1])
      assert np.isclose(tvy, ovy)

      if has_polarization:
        expected_px_pz = np.dot(rotation_matrix, [opolx, opolz])
        assert np.isclose(tpolx, expected_px_pz[0])
        assert np.isclose(tpolz, expected_px_pz[1])
        assert np.isclose(tpoly, opoly)

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
  # (which for orientation 1 maps X_nx to Y_ba (left) and Z_nx to X_ba (forward)).
  alpha = 0.44
  beam_declination = 0.44
  transformed = transform_to_bornagain_coordinate_system(particles, alpha, 1, beam_declination)
  unpacked = transformed[0]

  assert np.isclose(unpacked[1], z) # x_ba is z_nx
  assert np.isclose(unpacked[2], x) # y_ba is x_nx
  assert np.isclose(unpacked[3], y) # z_ba is y_nx
  assert np.isclose(unpacked[4], vz)
  assert np.isclose(unpacked[5], vx)
  assert np.isclose(unpacked[6], vy)
  print("Declination rotation cancellation check passed successfully!")

if __name__ == "__main__":
  test_rotation_consistency()
  test_declination_no_rotation()
