"""
Contains detector related calculations
"""

import numpy as np

class Detector:
  def __init__(self, det_params, sample_inclination):
    self.size_x = det_params['size'][0]
    self.size_y = det_params['size'][1]
    self.centre_centre_offset_x = det_params['centre_offset'][0]
    self.centre_centre_offset_y = det_params['centre_offset'][1]
    self.pixels_x = det_params['pixels'][0]
    self.pixels_y = det_params['pixels'][1]
    self.resolution_x = det_params['resolution'][0]
    self.resolution_y = det_params['resolution'][1]

    #derived parameters
    self.pixel_size_x = self.size_x / self.pixels_x
    self.pixel_size_y = self.size_y / self.pixels_y
    self.min_edge_x = self.centre_centre_offset_x - 0.5 * self.size_x
    self.min_edge_y = self.centre_centre_offset_y - 0.5 * self.size_y
    self.max_edge_x = self.centre_centre_offset_x + 0.5 * self.size_x
    self.max_edge_y = self.centre_centre_offset_y + 0.5 * self.size_y
    self.sigma_x = self.resolution_x / 2.355
    self.sigma_y = self.resolution_y / 2.355

    #coordinate transformation matrices
    self.bornagain_to_nexus_rotation_matrix = np.array([[np.cos(sample_inclination), -np.sin(sample_inclination)],
                                                        [np.sin(sample_inclination), np.cos(sample_inclination)]])

    self.nexus_to_bornagain_rotation_matrix = np.array([[np.cos(-sample_inclination), -np.sin(-sample_inclination)],
                                                        [np.sin(-sample_inclination), np.cos(-sample_inclination)]])

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
    gravityAcceleration = 9.80665 #m/s2
    yGravityAcc, zGravityAcc = self.transform_to_bornagain_coordinate_system(-gravityAcceleration, 0)
    zGravityDrop = zGravityAcc * 0.5 * t_propagate**2
    yGravityDrop = yGravityAcc * 0.5 * t_propagate**2
    return yGravityDrop, zGravityDrop

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

    # Calculate the effect of gravity
    yGravityDrop, zGravityDrop = self.calculate_gravity_drop(t_propagate)

    return t_propagate, (VX * t_propagate + x), (VY * t_propagate + y + yGravityDrop), (VZ * t_propagate + z + zGravityDrop)

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
