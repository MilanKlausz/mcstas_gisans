import numpy as np


class CoordinateTransform:
  """
  Handles coordinate transformations between NeXus laboratory coordinate system
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

    # Pre-bind sample orientation methods for zero-branching performance
    match sample_orientation:
      case 0:  # Vertical sample, beam from left (-90 deg rotation)
        self._apply_sample_orientation = self._transform_sample_orient_0
        self._apply_inverse_sample_orientation = self._inverse_sample_orient_0
      case 1:  # Horizontal sample (no rotation)
        self._apply_sample_orientation = self._transform_sample_orient_1
        self._apply_inverse_sample_orientation = self._inverse_sample_orient_1
      case 2:  # Vertical sample, beam from right (+90 deg rotation)
        self._apply_sample_orientation = self._transform_sample_orient_2
        self._apply_inverse_sample_orientation = self._inverse_sample_orient_2
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

  def transform_inclination_plane(self, z_nexus, y_nexus):
    """
    Apply sample inclination angle rotation (alpha) to the 2D vertical-longitudinal plane
    (z_nexus = distance, y_nexus = vertical height).
    Returns (z_bornagain, y_bornagain).
    """
    z_arr = np.asarray(z_nexus)
    y_arr = np.asarray(y_nexus)
    is_scalar = (z_arr.ndim == 0) and (y_arr.ndim == 0)

    z_flat = np.ravel(z_arr)
    y_flat = np.ravel(y_arr)

    rotated = np.matmul(
        self.sample_inclination_rotation_matrix,
        np.vstack((z_flat, y_flat)),
    )
    z_ba = rotated[0].reshape(z_arr.shape)
    y_ba = rotated[1].reshape(y_arr.shape)

    if is_scalar:
      return z_ba.item(), y_ba.item()

    return z_ba, y_ba

  def inverse_transform_inclination_plane(self, z_ba, y_ba):
    """
    Apply inverse sample inclination angle rotation (+alpha) to the 2D vertical-longitudinal plane.
    Returns (z_nexus, y_nexus).
    """
    z_arr = np.asarray(z_ba)
    y_arr = np.asarray(y_ba)
    is_scalar = (z_arr.ndim == 0) and (y_arr.ndim == 0)

    z_flat = np.ravel(z_arr)
    y_flat = np.ravel(y_arr)

    rotated = np.matmul(
        self.inverse_sample_inclination_rotation_matrix,
        np.vstack((z_flat, y_flat)),
    )
    z_nexus = rotated[0].reshape(z_arr.shape)
    y_nexus = rotated[1].reshape(y_arr.shape)

    if is_scalar:
      return z_nexus.item(), y_nexus.item()

    return z_nexus, y_nexus

  def nexus_to_bornagain(self, x_nexus, y_nexus, z_nexus):
    """
    Transform 3D position or velocity vector from NeXus coordinate system to BornAgain coordinate system.
    NeXus: X=horizontal, Y=vertical up, Z=longitudinal beam.
    BornAgain: X=horizontal transverse, Y=vertical, Z=longitudinal beam.
    """
    x_arr = np.asarray(x_nexus)
    y_arr = np.asarray(y_nexus)
    z_arr = np.asarray(z_nexus)
    is_scalar = (x_arr.ndim == 0) and (y_arr.ndim == 0) and (z_arr.ndim == 0)

    x_uninclined, y_uninclined = self._apply_sample_orientation(x_arr, y_arr)
    z_uninclined = z_arr

    z_flat = np.ravel(z_uninclined)
    y_flat = np.ravel(y_uninclined)

    rotated = np.matmul(
        self.sample_inclination_rotation_matrix,
        np.vstack((z_flat, y_flat)),
    )
    z_bornagain = rotated[0].reshape(z_uninclined.shape)
    y_bornagain = rotated[1].reshape(y_uninclined.shape)
    x_bornagain = x_uninclined

    if is_scalar:
      return x_bornagain.item(), y_bornagain.item(), z_bornagain.item()

    return x_bornagain, y_bornagain, z_bornagain

  def bornagain_to_nexus(self, x_bornagain, y_bornagain, z_bornagain):
    """
    Transform 3D position or velocity vector from BornAgain coordinate system to NeXus coordinate system.
    BornAgain: X=horizontal transverse, Y=vertical, Z=longitudinal beam.
    NeXus: X=horizontal, Y=vertical up, Z=longitudinal beam.
    """
    x_arr = np.asarray(x_bornagain)
    y_arr = np.asarray(y_bornagain)
    z_arr = np.asarray(z_bornagain)
    is_scalar = (x_arr.ndim == 0) and (y_arr.ndim == 0) and (z_arr.ndim == 0)

    z_flat = np.ravel(z_arr)
    y_flat = np.ravel(y_arr)

    rotated = np.matmul(
        self.inverse_sample_inclination_rotation_matrix,
        np.vstack((z_flat, y_flat)),
    )
    z_uninclined = rotated[0].reshape(z_arr.shape)
    y_uninclined = rotated[1].reshape(y_arr.shape)
    x_uninclined = x_arr

    x_nexus, y_nexus = self._apply_inverse_sample_orientation(
        x_uninclined, y_uninclined
    )
    z_nexus = z_uninclined

    if is_scalar:
      return x_nexus.item(), y_nexus.item(), z_nexus.item()

    return x_nexus, y_nexus, z_nexus
