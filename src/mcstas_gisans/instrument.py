
import numpy as np

from .detector import Detector
from .neutron_calculations import calculate_wavelength, calculate_wavenumber

class Instrument:
  def __init__(self, detector_params, sample_detector_distance, sample_inclination, alpha_inc, nominal_source_sample_distance, wavelength_selected):
    
    self.detector = Detector(detector_params, sample_inclination)
    self.sample_detector_distance = sample_detector_distance
    
    # q calculation related parameters
    self.nominal_source_sample_distance = nominal_source_sample_distance
    self.is_tof_instrument = wavelength_selected is None #TODO better to rely on instrument default!
    if not self.is_tof_instrument:
      self.wavenumber_fixed = calculate_wavenumber(wavelength_selected)
    self.incident_direction = np.array([0, -np.sin(alpha_inc), np.cos(alpha_inc)]) #a bit out of place but it's only used for q calculation
    
  def get_wavenumber(self, wavelength):
    if self.is_tof_instrument:
      wavenumber = calculate_wavenumber(wavelength)
    else:
      wavenumber = self.wavenumber_fixed
    return wavenumber
    
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

    # outgoing_direction_old = detection_coordinate / sample_detector_path_length[:, np.newaxis]
    outgoing_direction = self.coordinate_to_outgoing_direction(x_detection, y_detection, z_detection)

    if self.is_tof_instrument:
      wavelength = calculate_wavelength(t + sample_detector_tof, self.nominal_source_sample_distance + sample_detector_path_length)
      wavenumber = calculate_wavenumber(wavelength)[:, np.newaxis]
    else: #not TOF instruments
      wavenumber = self.wavenumber_fixed

    return (outgoing_direction - self.incident_direction) * wavenumber

  def coordinate_to_outgoing_direction(self, x, y, z): #TODO rename coordinate_to_outgoing_direction
    alpha_f = np.arctan(y/z)
    phi_f = np.arctan(x/z)
    return np.column_stack([np.cos(alpha_f) * np.sin(phi_f),
                            np.sin(alpha_f),
                            np.cos(alpha_f) * np.cos(phi_f)])

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

    outgoing_direction_q_min = self.coordinate_to_outgoing_direction(*q_min_coords)
    outgoing_direction_q_max = self.coordinate_to_outgoing_direction(*q_max_coords)

    wavenumber = self.get_wavenumber(wavelength)

    q_min = (outgoing_direction_q_min[0] - self.incident_direction) * wavenumber
    q_max = (outgoing_direction_q_max[0] - self.incident_direction) * wavenumber

    return q_min, q_max
  
  def get_expected_specular_peak_q(self, wavelength=None):
    outgoing_direction = np.array([self.incident_direction[0], -self.incident_direction[1], self.incident_direction[2]])
    wavenumber = self.get_wavenumber(wavelength)
    specular_peak_expected_q = (outgoing_direction - self.incident_direction) * wavenumber
    print("specular_peak_expected_q", specular_peak_expected_q)
      