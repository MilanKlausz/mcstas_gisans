
"""
This module defines the Detector class, which facilitates the determination of
detection coordinates.
"""

import numpy as np
from .coordinates import CoordinateTransform

class Detector:
  def __init__(self, det_params, sample_inclination, sample_orientation, no_gravity):

    # Expressing detector parameters in BornAgain coordinate system (Y=horizontal, Z=vertical)
    self.size_y_bornagain = det_params['size'][0]
    self.size_z_bornagain = det_params['size'][1]
    self.direct_beam_centre_offset_y_bornagain = det_params['direct_beam_centre_offset'][0]
    self.direct_beam_centre_offset_z_bornagain = det_params['direct_beam_centre_offset'][1]
    self.pixels_y_bornagain = det_params['pixels'][0]
    self.pixels_z_bornagain = det_params['pixels'][1]
    self.resolution_y_bornagain = det_params['resolution'][0]
    self.resolution_z_bornagain = det_params['resolution'][1]

    if sample_orientation in [0, 2]: #vertical sample orientation
        #swap y and z parameters for vertical sample orientation, because the detector is rotated by 90 degrees in the BornAgain coordinate system
        self.size_y_bornagain, self.size_z_bornagain = self.size_z_bornagain, self.size_y_bornagain
        self.pixels_y_bornagain, self.pixels_z_bornagain = self.pixels_z_bornagain, self.pixels_y_bornagain
        self.resolution_y_bornagain, self.resolution_z_bornagain = self.resolution_z_bornagain, self.resolution_y_bornagain
       
        # the direct_beam_centre offset is different for the two vertical sample orientations, because the detector is rotated by +-90 degrees in the BornAgain coordinate system
        if sample_orientation == 0: #vertical sample orientation, beam from the left
            self.direct_beam_centre_offset_y_bornagain, self.direct_beam_centre_offset_z_bornagain = -self.direct_beam_centre_offset_z_bornagain, self.direct_beam_centre_offset_y_bornagain
        else: #vertical sample orientation, beam from the right
            self.direct_beam_centre_offset_y_bornagain, self.direct_beam_centre_offset_z_bornagain = self.direct_beam_centre_offset_z_bornagain, -self.direct_beam_centre_offset_y_bornagain

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
    self.coords = CoordinateTransform(sample_inclination, sample_orientation)

    self.sample_orientation = sample_orientation
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
    z_nexus, _ = self.coords.inverse_transform_inclination_plane(z, y)
    vz_nexus, _ = self.coords.inverse_transform_inclination_plane(VZ, VY)
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

  def calculate_detection_coordinate(self, xDet, yDet, zDet):
    """
    Get the coordinate of the detection event from the position where the path of
    the particle intersects the plane of the detector surface. Using the exact
    position of intersection means infinite detector resolution.
    """
    # transform to the nexus coordinate system where the detector is vertical
    z_nexus, y_nexus = self.coords.inverse_transform_inclination_plane(zDet, yDet)
    #note: z_nexus is a fixed value due to the propagation to detector surface

    # apply gaussian randomisation to mimic the detection process
    xDet_smeared, y_smeared_nexus = self.apply_position_smearing(xDet, y_nexus)

    #get the coordinates of the centre of the pixel where the particle is detected
    xDetCoord, y_pixel_centre_nexus = self.get_pixel_centre_from_position(xDet_smeared, y_smeared_nexus)

    #transform to the sample-based bornagain coordinate system
    zDetCoord, yDetCoord = self.coords.transform_inclination_plane(z_nexus, y_pixel_centre_nexus)

    return xDetCoord, yDetCoord, zDetCoord

  def calculate_angles_from_spatial_bounds(self, sample_detector_distance, x_min, x_max, y_min_nexus, y_max_nexus):
    """
    Calculate opening angles [horiz_min, horiz_max, vert_min, vert_max] in degrees
    for given horizontal (x_min, x_max) and vertical (y_min_nexus, y_max_nexus) spatial boundaries in Nexus coord system.
    """
    angle_horiz_min_deg = np.rad2deg(np.arctan2(x_min, sample_detector_distance))
    angle_horiz_max_deg = np.rad2deg(np.arctan2(x_max, sample_detector_distance))

    z_top_ba, y_top = self.coords.transform_inclination_plane(sample_detector_distance, y_max_nexus)
    z_bottom_ba, y_bottom = self.coords.transform_inclination_plane(sample_detector_distance, y_min_nexus)

    y_angle_top = np.arctan2(y_top, z_top_ba)
    y_angle_bottom = np.arctan2(y_bottom, z_bottom_ba)

    if isinstance(y_angle_top, np.ndarray):
      y_angle_top = y_angle_top[0]
    if isinstance(y_angle_bottom, np.ndarray):
      y_angle_bottom = y_angle_bottom[0]

    angle_vert_min_deg = np.rad2deg(min(y_angle_bottom, y_angle_top))
    angle_vert_max_deg = np.rad2deg(max(y_angle_bottom, y_angle_top))

    return angle_horiz_min_deg, angle_horiz_max_deg, angle_vert_min_deg, angle_vert_max_deg

  def get_detector_angle_maximum(self, sample_detector_distance):
    """Calculate the 4 opening angles [horiz_min, horiz_max, vert_min, vert_max] covered by the detector (in deg)"""
    return self.calculate_angles_from_spatial_bounds(
        sample_detector_distance,
        self.min_edge_y_bornagain, self.max_edge_y_bornagain,
        self.min_edge_z_bornagain, self.max_edge_z_bornagain
    )

  def get_masked_angle_range(self, sample_detector_distance, mask, len_y_centres, factor=1.0):
    """
    Calculate the minimum opening angles [horiz_min, horiz_max, vert_min, vert_max] in degrees
    enclosing all unmasked (True) pixels in mask. Uses exact pixel outer boundaries.
    Optional factor scales the angular span symmetrically around the center (e.g., 1.05 for 5% margin).
    """
    x_edges = np.linspace(self.min_edge_y_bornagain, self.max_edge_y_bornagain, self.pixels_y_bornagain + 1)
    y_nexus_edges = np.linspace(self.min_edge_z_bornagain, self.max_edge_z_bornagain, self.pixels_z_bornagain + 1)

    if mask.shape[0] == len_y_centres:
      j_indices = np.where(np.any(mask, axis=1))[0]
      i_indices = np.where(np.any(mask, axis=0))[0]
    else:
      i_indices = np.where(np.any(mask, axis=1))[0]
      j_indices = np.where(np.any(mask, axis=0))[0]

    if len(j_indices) == 0 or len(i_indices) == 0:
      return self.get_detector_angle_maximum(sample_detector_distance)

    j_min, j_max = int(np.min(j_indices)), int(np.max(j_indices))
    i_min, i_max = int(np.min(i_indices)), int(np.max(i_indices))

    # Outer pixel boundaries
    x_min = x_edges[j_min]
    x_max = x_edges[j_max + 1]

    y_min_nexus = y_nexus_edges[i_min]
    y_max_nexus = y_nexus_edges[i_max + 1]

    h_min, h_max, v_min, v_max = self.calculate_angles_from_spatial_bounds(
        sample_detector_distance,
        x_min, x_max,
        y_min_nexus, y_max_nexus
    )

    if factor != 1.0:
      h_center = 0.5 * (h_min + h_max)
      h_half = 0.5 * (h_max - h_min) * factor
      h_min, h_max = h_center - h_half, h_center + h_half

      v_center = 0.5 * (v_min + v_max)
      v_half = 0.5 * (v_max - v_min) * factor
      v_min, v_max = v_center - v_half, v_center + v_half

    return h_min, h_max, v_min, v_max