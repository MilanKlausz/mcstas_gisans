
"""
This module defines the Detector class, which facilitates the determination of
detection coordinates.
"""

import numpy as np

class Detector:
  def __init__(self, det_params, sample_inclination, sample_orientation, no_gravity):

    # assume horizontal sample orientation (sample_orientation=1) as default, and then apply the necessary transformations for other sample orientations
    self.size_x = det_params['size'][0]
    self.size_y = det_params['size'][1]
    self.direct_beam_centre_offset_x = det_params['direct_beam_centre_offset'][0]
    self.direct_beam_centre_offset_y = det_params['direct_beam_centre_offset'][1]
    self.pixels_x = det_params['pixels'][0]
    self.pixels_y = det_params['pixels'][1]
    self.resolution_x = det_params['resolution'][0]
    self.resolution_y = det_params['resolution'][1]

    if sample_orientation in [0, 2]: #vertical sample orientation
        #swap x and y parameters for vertical sample orientation, because the detector is rotated by 90 degrees in the BornAgain coordinate system
        self.size_x, self.size_y = self.size_y, self.size_x
        self.pixels_x, self.pixels_y = self.pixels_y, self.pixels_x
        self.resolution_x, self.resolution_y = self.resolution_y, self.resolution_x
       
        # the direct_beam_centre offset is different for the two vertical sample orientations, because the detector is rotated by +-90 degrees in the BornAgain coordinate system
        if sample_orientation == 0: #vertical sample orientation, beam from the left
            self.direct_beam_centre_offset_x, self.direct_beam_centre_offset_y = -self.direct_beam_centre_offset_y, self.direct_beam_centre_offset_x
        else: #vertical sample orientation, beam from the right
            self.direct_beam_centre_offset_x, self.direct_beam_centre_offset_y = self.direct_beam_centre_offset_y, -self.direct_beam_centre_offset_x

    #derived parameters
    self.pixel_size_x = self.size_x / self.pixels_x
    self.pixel_size_y = self.size_y / self.pixels_y
    self.min_edge_x = self.direct_beam_centre_offset_x - 0.5 * self.size_x
    self.min_edge_y = self.direct_beam_centre_offset_y - 0.5 * self.size_y
    self.max_edge_x = self.direct_beam_centre_offset_x + 0.5 * self.size_x
    self.max_edge_y = self.direct_beam_centre_offset_y + 0.5 * self.size_y
    self.sigma_x = self.resolution_x / 2.355
    self.sigma_y = self.resolution_y / 2.355

    #coordinate transformation matrices
    self.bornagain_to_nexus_rotation_matrix = np.array([[np.cos(sample_inclination), -np.sin(sample_inclination)],
                                                        [np.sin(sample_inclination), np.cos(sample_inclination)]])

    self.nexus_to_bornagain_rotation_matrix = np.array([[np.cos(-sample_inclination), -np.sin(-sample_inclination)],
                                                        [np.sin(-sample_inclination), np.cos(-sample_inclination)]])

    self.sample_orientation = sample_orientation
    self.no_gravity = no_gravity
    if not no_gravity:
      self.gravity_acceleration_vector = self.calculate_gravity_vector()

  def calculate_gravity_vector(self):
    """ Calculate the gravity vector in bornagain coord system for different sample orientations """
    gravity_acceleration = 9.80665 #m/s2
    match self.sample_orientation:
      case 0:
        gravity_vector_nexus = [gravity_acceleration, 0.0, 0.0]
      case 1:
        gravity_vector_nexus = [0.0, -gravity_acceleration, 0.0]
      case 2:
        gravity_vector_nexus = [-gravity_acceleration, 0.0, 0.0]

    y_gravity_bornagain, z_gravity_bornagain = self.transform_to_bornagain_coordinate_system(gravity_vector_nexus[1], gravity_vector_nexus[2])

    return np.array([gravity_vector_nexus[0], y_gravity_bornagain[0], z_gravity_bornagain[0]])

  def apply_position_smearing(self, x, y):
    """ Apply Gaussian smearing to coordinates. """
    x_smeared = np.random.normal(x, self.sigma_x, size=x.shape)
    y_smeared = np.random.normal(y, self.sigma_y, size=y.shape)
    return x_smeared, y_smeared

  def get_pixel_centre_from_position(self, x, y):
    """ Find the centre of the pixel corresponding to the x,y coordinates."""
    #TODO doesn't check for missing the detector
    x_pixel_centre = np.floor((x - self.min_edge_x) / self.pixel_size_x) * self.pixel_size_x + 0.5*self.pixel_size_x + self.min_edge_x
    y_pixel_centre = np.floor((y - self.min_edge_y) / self.pixel_size_y) * self.pixel_size_y + 0.5*self.pixel_size_y + self.min_edge_y
    return x_pixel_centre, y_pixel_centre

  def transform_to_bornagain_coordinate_system(self, y, z):
    """
    Coordinate transformation from the Nexus coordinate system to the
    BornAgain coordinate system.
    Nexus coordinate system: https://manual.nexusformat.org/design.html#the-nexus-coordinate-system
    BornAgain coordinate system: 1 axis normal to the sample plane, the other 2 lay on in
    """
    z_rot, y_rot = np.matmul(self.nexus_to_bornagain_rotation_matrix, np.vstack((z, y)))
    return y_rot, z_rot

  def transform_to_nexus_coordinate_system(self, y, z):
    """
    Coordinate transformation from the BornAgain coordinate system to the
    Nexus coordinate system.
    Nexus coordinate system: https://manual.nexusformat.org/design.html#the-nexus-coordinate-system
    BornAgain coordinate system: 1 axis normal to the sample plane, the other 2 lay on in
    """
    z_rot, y_rot = np.matmul(self.bornagain_to_nexus_rotation_matrix, np.vstack((z, y)))
    return y_rot, z_rot

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
    _, zRot = self.transform_to_nexus_coordinate_system(y, z)
    _ , vzRot = self.transform_to_nexus_coordinate_system(VY, VZ)
    t_propagate = (sample_detector_distance - zRot) / vzRot

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
    yDetReal, zDetReal = self.transform_to_nexus_coordinate_system(yDet, zDet)
    #note: zDetReal is a fixed value due to the propagation to detector surface

    # apply gaussian randomisation to mimic the detection process
    xDet, yDetReal = self.apply_position_smearing(xDet, yDetReal)

    #get the coordinates of the centre of the pixel where the particle is detected
    xDetCoord, yDetCoordReal = self.get_pixel_centre_from_position(xDet, yDetReal)

    #transform to the sample-based bornagain coordinate system
    yDetCoord, zDetCoord = self.transform_to_bornagain_coordinate_system(yDetCoordReal, zDetReal)

    return xDetCoord, yDetCoord, zDetCoord

  def calculate_angles_from_spatial_bounds(self, sample_detector_distance, x_min, x_max, y_min_nexus, y_max_nexus):
    """
    Calculate opening angles [horiz_min, horiz_max, vert_min, vert_max] in degrees
    for given horizontal (x_min, x_max) and vertical (y_min_nexus, y_max_nexus) spatial boundaries in Nexus coord system.
    """
    angle_horiz_min_deg = np.rad2deg(np.arctan2(x_min, sample_detector_distance))
    angle_horiz_max_deg = np.rad2deg(np.arctan2(x_max, sample_detector_distance))

    y_top, z_top = self.transform_to_bornagain_coordinate_system(y_max_nexus, sample_detector_distance)
    y_bottom, z_bottom = self.transform_to_bornagain_coordinate_system(y_min_nexus, sample_detector_distance)

    y_angle_top = np.arctan2(y_top, z_top)
    y_angle_bottom = np.arctan2(y_bottom, z_bottom)

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
        self.min_edge_x, self.max_edge_x,
        self.min_edge_y, self.max_edge_y
    )

  def get_masked_angle_range(self, sample_detector_distance, mask, len_y_centres, factor=1.0):
    """
    Calculate the minimum opening angles [horiz_min, horiz_max, vert_min, vert_max] in degrees
    enclosing all unmasked (True) pixels in mask. Uses exact pixel outer boundaries.
    Optional factor scales the angular span symmetrically around the center (e.g., 1.05 for 5% margin).
    """
    x_edges = np.linspace(self.min_edge_x, self.max_edge_x, self.pixels_x + 1)
    y_nexus_edges = np.linspace(self.min_edge_y, self.max_edge_y, self.pixels_y + 1)

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