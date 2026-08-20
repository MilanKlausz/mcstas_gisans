
"""
This module defines the Detector class, which facilitates the determination of
detection coordinates.
"""

import numpy as np
from .coordinates import CoordinateTransform

class Detector:
  def __init__(self, det_params, sample_inclination, sample_orientation, no_gravity):

    # Expressing detector parameters in BornAgain coordinate system (Y=horizontal, Z=vertical)
    # Transform 2D vectors/magnitudes to BornAgain frame (left, up) using the sample orientation
    size_nexus = det_params['size']
    pixels_nexus = det_params['pixels']
    res_nexus = det_params['resolution']
    offset_nexus = det_params['direct_beam_centre_offset']
    # Physical detector parameters in NeXus coordinate system (X=horizontal, Y=vertical)
    self.size_x_nexus = size_nexus[0]
    self.size_y_nexus = size_nexus[1]
    self.pixels_x_nexus = pixels_nexus[0]
    self.pixels_y_nexus = pixels_nexus[1]
    self.pixel_size_x_nexus = self.size_x_nexus / self.pixels_x_nexus
    self.pixel_size_y_nexus = self.size_y_nexus / self.pixels_y_nexus
    self.direct_beam_centre_offset_x_nexus = offset_nexus[0]
    self.direct_beam_centre_offset_y_nexus = offset_nexus[1]
    self.min_edge_x_nexus = self.direct_beam_centre_offset_x_nexus - 0.5 * self.size_x_nexus
    self.min_edge_y_nexus = self.direct_beam_centre_offset_y_nexus - 0.5 * self.size_y_nexus
    self.max_edge_x_nexus = self.direct_beam_centre_offset_x_nexus + 0.5 * self.size_x_nexus
    self.max_edge_y_nexus = self.direct_beam_centre_offset_y_nexus + 0.5 * self.size_y_nexus

    self.coords = CoordinateTransform(sample_inclination, sample_orientation)
    self.sample_orientation = sample_orientation

    self.size_y_bornagain, self.size_z_bornagain = np.abs(self.coords.apply_sample_orientation_transform(size_nexus[0], size_nexus[1]))
    self.pixels_y_bornagain, self.pixels_z_bornagain = np.abs(self.coords.apply_sample_orientation_transform(pixels_nexus[0], pixels_nexus[1]))
    self.resolution_y_bornagain, self.resolution_z_bornagain = np.abs(self.coords.apply_sample_orientation_transform(res_nexus[0], res_nexus[1]))
    
    self.direct_beam_centre_offset_y_bornagain, self.direct_beam_centre_offset_z_bornagain = self.coords.apply_sample_orientation_transform(offset_nexus[0], offset_nexus[1])

    #derived parameters
    self.pixel_size_y_bornagain = self.size_y_bornagain / self.pixels_y_bornagain
    self.pixel_size_z_bornagain = self.size_z_bornagain / self.pixels_z_bornagain
    self.min_edge_y_bornagain = self.direct_beam_centre_offset_y_bornagain - 0.5 * self.size_y_bornagain
    self.min_edge_z_bornagain = self.direct_beam_centre_offset_z_bornagain - 0.5 * self.size_z_bornagain
    self.max_edge_y_bornagain = self.direct_beam_centre_offset_y_bornagain + 0.5 * self.size_y_bornagain
    self.max_edge_z_bornagain = self.direct_beam_centre_offset_z_bornagain + 0.5 * self.size_z_bornagain
    self.sigma_y_bornagain = self.resolution_y_bornagain / 2.355
    self.sigma_z_bornagain = self.resolution_z_bornagain / 2.355

    # Dedicated coordinate transformation helper
    self.no_gravity = no_gravity

    if not no_gravity:
      self.gravity_acceleration_vector = self.calculate_gravity_vector()

  def calculate_gravity_vector(self):
    """ Calculate the gravity vector in BornAgain coord system for different sample orientations """
    gravity_acceleration = 9.80665 #m/s2
    gravity_vector_nexus = [0.0, -gravity_acceleration, 0.0]

    gx, gy, gz = self.coords.nexus_to_bornagain(
        gravity_vector_nexus[0], gravity_vector_nexus[1], gravity_vector_nexus[2]
    )

    return np.array([gx, gy, gz])

  def apply_position_smearing(self, y_bornagain, z_bornagain):
    """ Apply Gaussian smearing to coordinates in BornAgain frame. """
    y_smeared = np.random.normal(y_bornagain, self.sigma_y_bornagain, size=y_bornagain.shape)
    z_smeared = np.random.normal(z_bornagain, self.sigma_z_bornagain, size=z_bornagain.shape)
    return y_smeared, z_smeared

  def get_pixel_indices_from_position(self, x_nexus, y_nexus):
    """
    Find 0-indexed integer pixel indices (idx_x, idx_y) corresponding to positions (x, y)
    in the raw physical NeXus detector frame.
    Returns (idx_x, idx_y, valid_mask).
    """
    idx_x = np.floor((x_nexus - self.min_edge_x_nexus) / self.pixel_size_x_nexus).astype(int)
    idx_y = np.floor((y_nexus - self.min_edge_y_nexus) / self.pixel_size_y_nexus).astype(int)
    valid_mask = (idx_x >= 0) & (idx_x < self.pixels_x_nexus) & (idx_y >= 0) & (idx_y < self.pixels_y_nexus)
    return idx_x, idx_y, valid_mask

  def calculate_pixel_hit(self, x_intersection_bornagain, y_intersection_bornagain, z_intersection_bornagain):
    """
    Calculate physical detector pixel indices (idx_x_nexus, idx_y_nexus) in raw NeXus frame for intersection coordinates.
    All operations are vectorized across outgoing rays.
    """
    x_intersection_bornagain_uninclined, z_intersection_bornagain_uninclined = self.coords.apply_inverse_inclination_angle_transformation(x_intersection_bornagain, z_intersection_bornagain)
    y_smeared_bornagain, z_smeared_bornagain_uninclined = self.apply_position_smearing(y_intersection_bornagain, z_intersection_bornagain_uninclined)

    # Pixel hit in BA sample frame detector grid (y is horizontal, z is vertical)
    idx_y_bornagain = np.floor((y_smeared_bornagain - self.min_edge_y_bornagain) / self.pixel_size_y_bornagain).astype(int)
    idx_z_bornagain = np.floor((z_smeared_bornagain_uninclined - self.min_edge_z_bornagain) / self.pixel_size_z_bornagain).astype(int)
    valid_mask = (idx_y_bornagain >= 0) & (idx_y_bornagain < self.pixels_y_bornagain) & (idx_z_bornagain >= 0) & (idx_z_bornagain < self.pixels_z_bornagain)

    match self.sample_orientation:
      case 0:
        idx_x_nexus = self.pixels_x_nexus - 1 - idx_z_bornagain
        idx_y_nexus = idx_y_bornagain
      case 1:
        idx_x_nexus = idx_y_bornagain
        idx_y_nexus = idx_z_bornagain
      case 2:
        idx_x_nexus = idx_z_bornagain
        idx_y_nexus = self.pixels_y_nexus - 1 - idx_y_bornagain

    return idx_x_nexus, idx_y_nexus, valid_mask

  def get_pixel_centre_from_position(self, y_bornagain, z_bornagain):
    """ Find the centre of the pixel corresponding to the y, z coordinates in BornAgain frame."""
    #TODO doesn't check for missing the detector
    y_pixel_centre = np.floor((y_bornagain - self.min_edge_y_bornagain) / self.pixel_size_y_bornagain) * self.pixel_size_y_bornagain + 0.5*self.pixel_size_y_bornagain + self.min_edge_y_bornagain
    z_pixel_centre = np.floor((z_bornagain - self.min_edge_z_bornagain) / self.pixel_size_z_bornagain) * self.pixel_size_z_bornagain + 0.5*self.pixel_size_z_bornagain + self.min_edge_z_bornagain
    return y_pixel_centre, z_pixel_centre

  def calculate_gravity_drop(self, t_propagate):
    """Calculate the effect of gravity during the propagation to detector surface"""
    t_propagate_square_half = 0.5 * t_propagate**2
    x_drop = self.gravity_acceleration_vector[0] * t_propagate_square_half
    y_drop = self.gravity_acceleration_vector[1] * t_propagate_square_half
    z_drop = self.gravity_acceleration_vector[2] * t_propagate_square_half
    return x_drop, y_drop, z_drop

  def detector_plane_intersection(self, x, y, z, VX, VY, VZ, sample_detector_distance):
    """
    Calculate x,y,z position on the detector surface and the corresponding TOF
    for the sample to detector propagation.
    NOTE: under the assumption that the detector surface is vertical in the Nexus coord system
    """
    # Calculate propagation time until the detector surface in the nexus
    # coordinate system, where the z velocity component is perpendicular to it
    z_nexus, _ = self.coords.apply_inverse_inclination_angle_transformation(x, z)
    vz_nexus, _ = self.coords.apply_inverse_inclination_angle_transformation(VX, VZ)
    t_propagate = (sample_detector_distance - z_nexus) / vz_nexus

    x_intersection = VX * t_propagate + x
    y_intersection = VY * t_propagate + y
    z_intersection = VZ * t_propagate + z
    if not self.no_gravity:
      x_drop, y_drop, z_drop = self.calculate_gravity_drop(t_propagate)
      x_intersection += x_drop
      y_intersection += y_drop
      z_intersection += z_drop

    return t_propagate, x_intersection, y_intersection, z_intersection

  def calculate_detection_coordinate(self, x_intersection_bornagain, y_intersection_bornagain, z_intersection_bornagain):
    """
    Get the coordinate of the detection event from the position where the path of
    the particle intersects the plane of the detector surface. Using the exact
    position of intersection means infinite detector resolution.
    Note: in the Bornagain coordinate system, xDet is forward, yDet is left, zDet is up.
    """
    # transform to the uninclined coordinate system where the detector is vertical
    x_intersection_bornagain_uninclined, z_intersection_bornagain_uninclined = self.coords.apply_inverse_inclination_angle_transformation(x_intersection_bornagain, z_intersection_bornagain)
    #note: x_intersection_bornagain_uninclined is a fixed value due to the propagation to detector surface

    # apply gaussian randomisation to mimic the detection process
    y_detection_bornagain, z_detection_bornagain_uninclined = self.apply_position_smearing(y_intersection_bornagain, z_intersection_bornagain_uninclined)

    #get the coordinates of the centre of the pixel where the particle is detected
    y_pixel_centre_bornagain, z_pixel_centre_bornagain_uninclined = self.get_pixel_centre_from_position(y_detection_bornagain, z_detection_bornagain_uninclined)

    #transform to the inclined bornagain coordinate system
    x_pixel_centre_bornagain, z_pixel_centre_bornagain = self.coords.apply_inclination_angle_transformation(x_intersection_bornagain_uninclined, z_pixel_centre_bornagain_uninclined)

    return x_pixel_centre_bornagain, y_pixel_centre_bornagain, z_pixel_centre_bornagain

  def calculate_angles_from_spatial_bounds(self, sample_detector_distance, y_min_bornagain, y_max_bornagain, z_min_bornagain_uninclined, z_max_bornagain_uninclined):
    """
    Calculate opening angles [horiz_min, horiz_max, vert_min, vert_max] in degrees
    for given horizontal (y_min_bornagain, y_max_bornagain) and vertical (z_min_bornagain_uninclined, z_max_bornagain_uninclined) spatial boundaries.
    Note: the vertical boundaries are expressed in the uninclined coordinate system, where the detector is vertical for user convenience, so it needs to be transformed to the (inclined) BornAgain coordinate system for the angle calculation.
    """
    angle_horiz_min_deg = np.rad2deg(np.arctan2(y_min_bornagain, sample_detector_distance))
    angle_horiz_max_deg = np.rad2deg(np.arctan2(y_max_bornagain, sample_detector_distance))

    # Transform to BornAgain coordinates to calculate the vertical angles
    x_top_bornagain, z_top_bornagain = self.coords.apply_inclination_angle_transformation(sample_detector_distance, z_max_bornagain_uninclined)
    x_bottom_bornagain, z_bottom_bornagain = self.coords.apply_inclination_angle_transformation(sample_detector_distance, z_min_bornagain_uninclined)

    z_angle_top = np.arctan2(z_top_bornagain, x_top_bornagain)
    z_angle_bottom = np.arctan2(z_bottom_bornagain, x_bottom_bornagain)

    if isinstance(z_angle_top, np.ndarray):
      z_angle_top = z_angle_top[0]
    if isinstance(z_angle_bottom, np.ndarray):
      z_angle_bottom = z_angle_bottom[0]

    angle_vert_min_deg = np.rad2deg(min(z_angle_bottom, z_angle_top))
    angle_vert_max_deg = np.rad2deg(max(z_angle_bottom, z_angle_top))

    return angle_horiz_min_deg, angle_horiz_max_deg, angle_vert_min_deg, angle_vert_max_deg

  def get_detector_angle_maximum(self, sample_detector_distance):
    """Calculate the 4 opening angles [horiz_min, horiz_max, vert_min, vert_max] covered by the detector (in deg)"""
    return self.calculate_angles_from_spatial_bounds(
        sample_detector_distance,
        self.min_edge_y_bornagain, self.max_edge_y_bornagain,
        self.min_edge_z_bornagain, self.max_edge_z_bornagain
    )

  def get_masked_angle_range(self, sample_detector_distance, mask, factor=1.0):
    """
    Calculate the minimum opening angles [horiz_min, horiz_max, vert_min, vert_max] in degrees
    enclosing all unmasked (True) pixels in mask. Uses exact pixel outer boundaries.
    Optional factor scales the angular span symmetrically around the center (e.g., 1.05 for 5% margin).
    """
    x_edges = np.linspace(self.min_edge_y_bornagain, self.max_edge_y_bornagain, self.pixels_y_bornagain + 1)
    y_nexus_edges = np.linspace(self.min_edge_z_bornagain, self.max_edge_z_bornagain, self.pixels_z_bornagain + 1)

    j_indices = np.where(np.any(mask, axis=1))[0]
    i_indices = np.where(np.any(mask, axis=0))[0]

    if len(j_indices) == 0 or len(i_indices) == 0:
      return self.get_detector_angle_maximum(sample_detector_distance)

    j_min, j_max = int(np.min(j_indices)), int(np.max(j_indices))
    i_min, i_max = int(np.min(i_indices)), int(np.max(i_indices))

    y_min_bornagain = x_edges[j_min]
    y_max_bornagain = x_edges[j_max + 1]

    z_min_uninclined = y_nexus_edges[i_min]
    z_max_uninclined = y_nexus_edges[i_max + 1]

    h_min, h_max, v_min, v_max = self.calculate_angles_from_spatial_bounds(
        sample_detector_distance,
        y_min_bornagain, y_max_bornagain,
        z_min_uninclined, z_max_uninclined
    )

    if factor != 1.0:
      h_center = 0.5 * (h_min + h_max)
      h_half = 0.5 * (h_max - h_min) * factor
      h_min, h_max = h_center - h_half, h_center + h_half

      v_center = 0.5 * (v_min + v_max)
      v_half = 0.5 * (v_max - v_min) * factor
      v_min, v_max = v_center - v_half, v_center + v_half

    return h_min, h_max, v_min, v_max

  def get_pixel_positions(self, sample_detector_distance):
    x_centers = np.linspace(self.min_edge_x_nexus + self.pixel_size_x_nexus/2, 
                            self.max_edge_x_nexus - self.pixel_size_x_nexus/2, 
                            self.pixels_x_nexus)
    y_centers = np.linspace(self.min_edge_y_nexus + self.pixel_size_y_nexus/2, 
                            self.max_edge_y_nexus - self.pixel_size_y_nexus/2, 
                            self.pixels_y_nexus)
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    num_pixels = self.pixels_x_nexus * self.pixels_y_nexus
    positions = np.zeros((num_pixels, 3))
    positions[:, 0] = X.flatten()
    positions[:, 1] = Y.flatten()
    positions[:, 2] = sample_detector_distance
    return positions