import numpy as np


class CoordinateTransform:
  """
  Handles coordinate transformations between NeXus coordinate system
  and BornAgain sample-centric coordinate system, taking into account:
  1. Sample orientation (rotation around the beam axis Z_nexus by 0, +/-90 degrees)
  2. Sample inclination angle alpha (rotation in the vertical-longitudinal plane)
  """

  def __init__(self, sample_inclination, sample_orientation):
    self.sample_inclination = sample_inclination
    self.sample_orientation = sample_orientation

    # 2D Rotation matrices for sample inclination angle alpha
    self.inverse_sample_inclination_rotation_matrix = np.array([
        [np.cos(sample_inclination), -np.sin(sample_inclination)],
        [np.sin(sample_inclination), np.cos(sample_inclination)],
    ])

    self.sample_inclination_rotation_matrix = np.array([
        [np.cos(-sample_inclination), -np.sin(-sample_inclination)],
        [np.sin(-sample_inclination), np.cos(-sample_inclination)],
    ])

    # Pre-bind sample orientation dependent methods to eliminate branching when they are used
    match sample_orientation:
      case 0:  # Vertical sample, beam from left (-90 deg rotation)
        """Beam hitting vertical sample from the left: -90 deg rotation"""
        self.apply_sample_orientation_transform = self._transform_sample_orient_0
        self._apply_inverse_sample_orientation_transform = self._inverse_sample_orient_0
      case 1:  # Horizontal sample (no rotation)
        """Horizontal sample: no rotation"""
        self.apply_sample_orientation_transform = self._transform_sample_orient_1
        self._apply_inverse_sample_orientation_transform = self._inverse_sample_orient_1
      case 2:  # Vertical sample, beam from right (+90 deg rotation)
        """Beam hitting vertical sample from the right: -90 deg rotation"""
        self.apply_sample_orientation_transform = self._transform_sample_orient_2
        self._apply_inverse_sample_orientation_transform = self._inverse_sample_orient_2
      case _:
        raise ValueError(
            f"Unknown sample orientation: {sample_orientation}"
        )

  def _transform_sample_orient_0(self, x_nexus, y_nexus):
    return -y_nexus, x_nexus

  def _transform_sample_orient_1(self, x_nexus, y_nexus):
    return x_nexus, y_nexus

  def _transform_sample_orient_2(self, x_nexus, y_nexus):
    return y_nexus, -x_nexus

  def _inverse_sample_orient_0(self, x_uninclined, y_uninclined):
    return y_uninclined, -x_uninclined

  def _inverse_sample_orient_1(self, x_uninclined, y_uninclined):
    return x_uninclined, y_uninclined

  def _inverse_sample_orient_2(self, x_uninclined, y_uninclined):
    return -y_uninclined, x_uninclined

  def rotate_detector_image(self, hist_nexus):
    """
    Rotates a 2D NeXus detector image matrix (horizontal x vertical) to match the
    (uninclined) BornAgain sample frame.
    """
    match self.sample_orientation:
      case 0:
        return np.rot90(hist_nexus, -1)
      case 1:
        return hist_nexus
      case 2:
        return np.rot90(hist_nexus, 1)
      case _:
        raise ValueError(f"Unknown sample orientation: {self.sample_orientation}")

  def apply_inclination_angle_transformation(self, x_uninclined, z_uninclined):
    """
    Apply sample inclination angle rotation (alpha) to the 2D vertical-longitudinal plane
    (x = forward, z = up).

    Accepts both scalar and array/vector inputs.
    """
    x_arr = np.asarray(x_uninclined)
    z_arr = np.asarray(z_uninclined)

    x_flat = np.ravel(x_arr)
    z_flat = np.ravel(z_arr)

    rotated = np.matmul(
        self.sample_inclination_rotation_matrix,
        np.vstack((x_flat, z_flat)),
    )
    x_bornagain = rotated[0].reshape(x_arr.shape)
    z_bornagain = rotated[1].reshape(z_arr.shape)

    return x_bornagain, z_bornagain

  def apply_inverse_inclination_angle_transformation(self, x_bornagain, z_bornagain):
    """
    Apply inverse sample inclination angle rotation (+alpha) to the 2D vertical-longitudinal plane.

    Accepts both scalar and array/vector inputs.
    """
    x_arr = np.asarray(x_bornagain)
    z_arr = np.asarray(z_bornagain)

    x_flat = np.ravel(x_arr)
    z_flat = np.ravel(z_arr)

    rotated = np.matmul(
        self.inverse_sample_inclination_rotation_matrix,
        np.vstack((x_flat, z_flat)),
    )
    x_uninclined = rotated[0].reshape(x_arr.shape)
    z_uninclined = rotated[1].reshape(z_arr.shape)

    return x_uninclined, z_uninclined

  def nexus_to_bornagain(self, x_nexus, y_nexus, z_nexus):
    """
    Transform 3D position or velocity vector from NeXus coordinate system to BornAgain coordinate system.
    NeXus: X=horizontal left, Y=vertical up, Z=longitudinal forward.
    BornAgain: X=longitudinal forward, Y=horizontal left, Z=vertical up.
    The transformation (depends on sample orientation and inclination

    Accepts both scalar and array/vector inputs.
    """
    x_arr = np.asarray(x_nexus)
    y_arr = np.asarray(y_nexus)
    z_arr = np.asarray(z_nexus)

    x_horiz, y_vert = self.apply_sample_orientation_transform(x_arr, y_arr)

    x_bornagain_uninclined = z_arr
    y_bornagain_uninclined = x_horiz
    z_bornagain_uninclined = y_vert

    x_bornagain, z_bornagain = self.apply_inclination_angle_transformation(x_bornagain_uninclined, z_bornagain_uninclined)
    y_bornagain = y_bornagain_uninclined

    return x_bornagain, y_bornagain, z_bornagain

  def bornagain_to_nexus(self, x_bornagain, y_bornagain, z_bornagain):
    """
    Transform 3D position or velocity vector from BornAgain coordinate system to NeXus coordinate system.
    BornAgain: X=longitudinal forward, Y=horizontal left, Z=vertical up.
    NeXus: X=horizontal left, Y=vertical up, Z=longitudinal forward.

    Accepts both scalar and array/vector inputs.
    """
    x_arr = np.asarray(x_bornagain)
    y_arr = np.asarray(y_bornagain)
    z_arr = np.asarray(z_bornagain)

    x_uninclined, z_uninclined = self.apply_inverse_inclination_angle_transformation(x_arr, z_arr)
    y_uninclined = y_arr

    x_horiz = y_uninclined
    y_vert = z_uninclined
    z_nexus = x_uninclined

    x_nexus, y_nexus = self._apply_inverse_sample_orientation_transform(x_horiz, y_vert)

    return x_nexus, y_nexus, z_nexus
