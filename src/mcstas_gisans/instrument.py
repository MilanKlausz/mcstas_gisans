
"""
This module defines the Instrument class, which is mainly intended for
scattering vector (q) related calculations.
"""

import numpy as np

from .detector import Detector
from .particle_calculations import calculate_neutron_wavelength, calculate_wavenumber, calculate_neutron_velocity

class Instrument:
  def __init__(self, instr_params, alpha_inc_deg, wavelength_selected, sample_orientation, wfm=False, no_gravity=False):
    beam_declination = instr_params.get('beam_declination_angle', 0)
    sample_inclination = float(np.deg2rad(alpha_inc_deg - beam_declination))
    self.detector = Detector(instr_params['detector'], sample_inclination, sample_orientation, no_gravity)

    #TODO there should be a user warning for wft=True but no instr_params['wfm_virtual_source_distance']
    self.nominal_source_sample_distance = instr_params['nominal_source_sample_distance'] - (0 if not wfm else instr_params['wfm_virtual_source_distance'])
    self.sample_detector_distance = instr_params['sample_detector_distance']

    self.no_gravity = no_gravity
    self.alpha_inc = float(np.deg2rad(alpha_inc_deg))
    self.wavelength_selected = wavelength_selected

    self.incident_direction = self.calculate_incident_direction(wavelength_selected)

    self.is_tof_instrument = instr_params['tof_instrument']
    if not self.is_tof_instrument:
      self.wavenumber_fixed = calculate_wavenumber(wavelength_selected)

  def calculate_incident_direction(self, wavelength):
    """
    Calculate the reference incident direction, taking gravity drop into account
    from the sample to the detector surface if needed.
    """
    incident_dir_straight = np.array([0.0, -np.sin(self.alpha_inc), np.cos(self.alpha_inc)])
    if self.no_gravity or wavelength is None:
      return incident_dir_straight

    t_flight = self.sample_detector_distance / calculate_neutron_velocity(wavelength)

    drop_vector = 0.5 * self.detector.gravity_acceleration_vector * t_flight**2
    straight_pos = incident_dir_straight * self.sample_detector_distance
    dropped_pos = straight_pos + drop_vector
    return dropped_pos / np.linalg.norm(dropped_pos)

  def get_wavenumber(self, wavelength):
    """ Return the wavenumber that is fixed in case of non-TOF instrument """
    return calculate_wavenumber(wavelength) if self.is_tof_instrument else self.wavenumber_fixed

  def calculate_q(self, x, y, z, t, VX, VY, VZ):
    """
    Calculate Q values (x,y,z) from positions at the detector surface.
    All outgoing directions from the BornAgain simulation of a single particle are
    handled at the same time using operations on vectors.
    - Outgoing direction is calculated by propagating particles to the detector surface,
    and assuming that the particle is scattered at the centre of the sample (the origin).
    - Incident direction is fixed.
    - For non-TOF instruments the (2*pi/(wavelength)) factor is fixed, calculated
      from the wavelength selected by the monochromator (wavelength_selected).
      For TOF instruments the wavelength is calculated from the TOF at the
      detector surface position and the nominal distance travelled by the
      particle until that position.
    """
    sample_detector_tof, x_detector_plane, y_detector_plane, z_detector_plane = self.detector.detector_plane_intersection(x, y, z, VX, VY, VZ, self.sample_detector_distance)
    x_detection, y_detection, z_detection = self.detector.calculate_detection_coordinate(x_detector_plane, y_detector_plane, z_detector_plane)

    detection_coordinate = np.vstack((x_detection, y_detection, z_detection)).T
    sample_detector_path_length = np.linalg.norm(detection_coordinate, axis=1)
    outgoing_direction = detection_coordinate / sample_detector_path_length[:, np.newaxis]

    if self.is_tof_instrument:
      tof_total = t + sample_detector_tof
      path_length_total = self.nominal_source_sample_distance + sample_detector_path_length
      wavelength = calculate_neutron_wavelength(tof_total, path_length_total)
      wavenumber = calculate_wavenumber(wavelength)[:, np.newaxis]
    else: #not TOF instruments
      wavenumber = self.wavenumber_fixed

    return (outgoing_direction - self.incident_direction) * wavenumber

  def calculate_q_limits(self, wavelength=None):
    """
    Calculate the min and max q values for a wavelength using the xy min and
    max coordinates of the detector (it is an approximation).
    """
    # Since the Detector class constructor already swaps the active area coordinates (size_x, size_y, min_edge_x, min_edge_y)
    # in accordance with the sample_orientation, self.detector.min_edge_x/y are already in the sample frame.
    
    # 1. Start with the limits in the uninclined sample frame (where x is horizontal, y is vertical, z is distance).
    q_min_x_sample = self.detector.min_edge_x
    q_max_x_sample = self.detector.max_edge_x

    # 2. Project the vertical (y) and longitudinal (z) limits to the inclined BornAgain coordinate system.
    q_min_y_sample, q_min_z_sample = self.detector.transform_to_bornagain_coordinate_system(
        self.detector.min_edge_y, self.sample_detector_distance
    )
    q_max_y_sample, q_max_z_sample = self.detector.transform_to_bornagain_coordinate_system(
        self.detector.max_edge_y, self.sample_detector_distance
    )

    # 3. Combine into coordinate limit vectors in BornAgain space.
    q_min_coords = [q_min_x_sample, q_min_y_sample[0], q_min_z_sample[0]]
    q_max_coords = [q_max_x_sample, q_max_y_sample[0], q_max_z_sample[0]]

    # 4. Convert coordinate limits to outgoing direction unit vectors.
    outgoing_direction_q_min = q_min_coords / np.linalg.norm(q_min_coords)
    outgoing_direction_q_max = q_max_coords / np.linalg.norm(q_max_coords)

    wavenumber = self.get_wavenumber(wavelength)
    
    # 5. Compute the reference incident direction. For non-TOF instruments, this uses the pre-calculated gravity-dropped reference direction.
    if not self.is_tof_instrument:
      w = wavelength if wavelength is not None else self.wavelength_selected
      incident_direction = self.calculate_incident_direction(w)
    else:
      incident_direction = self.incident_direction

    # 6. Calculate min and max scattering vector (Q) limits.
    q_min = (outgoing_direction_q_min - incident_direction) * wavenumber
    q_max = (outgoing_direction_q_max - incident_direction) * wavenumber

    return q_min, q_max

  def get_q_pixel_limits(self, wavelength=None):
    """
    Calculate and return the Q-space bin edges (q_y, q_z) for each pixel
    boundary on the detector, relying on the detector's pixel dimensions.
    """
    q_min, q_max = self.calculate_q_limits(wavelength)
    q_y = np.linspace(q_min[0], q_max[0], num=self.detector.pixels_x + 1)
    q_z = np.linspace(q_min[1], q_max[1], num=self.detector.pixels_y + 1)
    return q_y, q_z

  def get_expected_specular_peak_q(self, wavelength=None):
    """Calculate approximate q value for the specular peak (without gravity)"""
    outgoing_direction = np.array([self.incident_direction[0], -self.incident_direction[1], self.incident_direction[2]])
    wavenumber = self.get_wavenumber(wavelength)
    specular_peak_expected_q = (outgoing_direction - self.incident_direction) * wavenumber
    print("specular_peak_expected_q", specular_peak_expected_q)

  def get_detector_angle_maximum(self):
    return self.detector.get_detector_angle_maximum(self.sample_detector_distance)