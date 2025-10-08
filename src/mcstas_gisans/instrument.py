
"""
This module defines the Instrument class, which is mainly intended for
scattering vector (q) related calculations.
"""

import numpy as np

from .detector import Detector
from .neutron_calculations import calculate_neutron_wavelength, calculate_wavenumber

class Instrument:
  def __init__(self, instr_params, alpha_inc_deg, wavelength_selected, sample_orientation, wfm=False, no_gravity=False):
    beam_declination = instr_params.get('beam_declination_angle', 0)
    sample_inclination = float(np.deg2rad(alpha_inc_deg - beam_declination))
    self.detector = Detector(instr_params['detector'], sample_inclination, sample_orientation, no_gravity)

    #TODO there should be a user warning for wft=True but no instr_params['wfm_virtual_source_distance']
    self.nominal_source_sample_distance = instr_params['nominal_source_sample_distance'] - (0 if not wfm else instr_params['wfm_virtual_source_distance'])
    self.sample_detector_distance = instr_params['sample_detector_distance']

    alpha_inc = float(np.deg2rad(alpha_inc_deg))
    self.incident_direction = np.array([0, -np.sin(alpha_inc), np.cos(alpha_inc)])

    self.is_tof_instrument = instr_params['tof_instrument']
    if not self.is_tof_instrument:
      self.wavenumber_fixed = calculate_wavenumber(wavelength_selected)

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

    q_min_coords_nexus = [self.detector.min_edge_x, self.detector.min_edge_y, self.sample_detector_distance]
    q_max_coords_nexus = [self.detector.max_edge_x, self.detector.max_edge_y, self.sample_detector_distance]

    q_min_y, q_min_z = self.detector.transform_to_bornagain_coordinate_system(q_min_coords_nexus[1], q_min_coords_nexus[2])
    q_max_y, q_max_z =self.detector.transform_to_bornagain_coordinate_system(q_max_coords_nexus[1], q_max_coords_nexus[2])

    q_min_coords = [q_min_coords_nexus[0], q_min_y[0], q_min_z[0]]
    q_max_coords = [q_max_coords_nexus[0], q_max_y[0], q_max_z[0]]

    outgoing_direction_q_min = q_min_coords / np.linalg.norm(q_min_coords)
    outgoing_direction_q_max = q_max_coords / np.linalg.norm(q_max_coords)

    wavenumber = self.get_wavenumber(wavelength)

    q_min = (outgoing_direction_q_min - self.incident_direction) * wavenumber
    q_max = (outgoing_direction_q_max - self.incident_direction) * wavenumber

    return q_min, q_max

  def get_expected_specular_peak_q(self, wavelength=None):
    """Calculate approximate q value for the specular peak (without gravity)"""
    outgoing_direction = np.array([self.incident_direction[0], -self.incident_direction[1], self.incident_direction[2]])
    wavenumber = self.get_wavenumber(wavelength)
    specular_peak_expected_q = (outgoing_direction - self.incident_direction) * wavenumber
    print("specular_peak_expected_q", specular_peak_expected_q)

  def get_detector_angle_maximum(self):
    return self.detector.get_detector_angle_maximum(self.sample_detector_distance)