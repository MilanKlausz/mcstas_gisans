
"""
This module defines the Instrument class, which is mainly intended for
scattering vector (q) related calculations.
"""

import numpy as np

from .detector import Detector
from .particle_calculations import calculate_neutron_wavelength, calculate_wavenumber, calculate_neutron_velocity

class Instrument:
  def __init__(self, instr_params, alpha_inc_deg, wavelength_selected, sample_orientation, wfm=False, no_gravity=False):
    beam_angle = instr_params.get('beam_angle', 0)
    self.beam_angle = beam_angle
    sample_inclination = float(np.deg2rad(alpha_inc_deg + beam_angle))
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
    incident_dir_straight = np.array([np.cos(self.alpha_inc), 0.0, -np.sin(self.alpha_inc)])
    if self.no_gravity or wavelength is None:
      return incident_dir_straight

    t_flight = self.sample_detector_distance / calculate_neutron_velocity(wavelength)

    drop_vector = 0.5 * self.detector.gravity_acceleration_vector * t_flight**2
    straight_pos = incident_dir_straight * self.sample_detector_distance
    dropped_pos = straight_pos + drop_vector
    return dropped_pos / np.linalg.norm(dropped_pos)

  def get_wavenumber(self, wavelength):
    """ Return the wavenumber that is fixed in case of non-TOF instrument """
    if wavelength is None:
        wavelength = self.wavelength_selected
    return calculate_wavenumber(wavelength) if self.is_tof_instrument else self.wavenumber_fixed

  def compute_q_scipp(self, scipp_da):
    """
    Calculate Q values from positions at the detector surface using Scipp.
    Applies small angle approximations or full vector operations including gravity drop.
    """
    import scipp as sc
    import scippneutron as scn
    import scipy.constants as const

    # m_n / h in s/m^2 (approx 252.77) -> h / m_n in m^2/s
    h_over_m = sc.scalar(const.h / const.m_n, unit='m**2/s')

    # Convert detector positions from NeXus to BornAgain coordinate system
    pos = scipp_da.coords['position']
    pos_x = pos.fields.x
    pos_y = pos.fields.y
    pos_z = pos.fields.z

    if self.detector.sample_orientation == 0:
        x_horiz = -pos_y
        y_vert = pos_x
    elif self.detector.sample_orientation == 1:
        x_horiz = pos_x
        y_vert = pos_y
    elif self.detector.sample_orientation == 2:
        x_horiz = pos_y
        y_vert = -pos_x
    else:
        raise ValueError(f"Unknown sample orientation: {self.detector.sample_orientation}")

    x_ba_uninclined = pos_z
    y_ba_uninclined = x_horiz
    z_ba_uninclined = y_vert

    alpha = self.detector.coords.sample_inclination
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # Note: the matrix is M11=cos_a, M12=sin_a, M21=-sin_a, M22=cos_a
    pos_ba_x = x_ba_uninclined * cos_a + z_ba_uninclined * sin_a
    pos_ba_y = y_ba_uninclined
    pos_ba_z = -x_ba_uninclined * sin_a + z_ba_uninclined * cos_a

    L2 = sc.sqrt(pos_ba_x**2 + pos_ba_y**2 + pos_ba_z**2)
    out_dir_x = pos_ba_x / L2
    out_dir_y = pos_ba_y / L2
    out_dir_z = pos_ba_z / L2

    if self.is_tof_instrument:
        if scipp_da.bins is not None:
            if 'wavelength' not in scipp_da.bins.coords:
                scipp_da = scn.convert(scipp_da, origin='tof', target='wavelength', scatter=True)
            wavelength = scipp_da.bins.coords['wavelength']
        else:
            if 'wavelength' not in scipp_da.coords:
                scipp_da = scn.convert(scipp_da, origin='tof', target='wavelength', scatter=True)
            wavelength = scipp_da.coords['wavelength']
    else:
        wavelength_val = self.wavelength_selected if self.wavelength_selected is not None else 0.0
        if wavelength_val == 0.0:
            wavelength = sc.scalar(1.0, unit='angstrom') # fallback
        else:
            wavelength = sc.scalar(wavelength_val, unit='angstrom')

    wavenumber = 2.0 * np.pi / wavelength

    inc_dir_straight_x = sc.scalar(np.cos(self.alpha_inc))
    inc_dir_straight_z = sc.scalar(-np.sin(self.alpha_inc))

    if self.no_gravity or getattr(self, 'wavelength_selected', None) == 0.0:
        inc_dir_x = inc_dir_straight_x
        inc_dir_y = sc.scalar(0.0)
        inc_dir_z = inc_dir_straight_z
    else:
        wavelength_m = sc.to_unit(wavelength, 'm')
        velocity = h_over_m / wavelength_m
        
        L_nom = sc.scalar(self.sample_detector_distance, unit='m')
        t_flight = L_nom / velocity
        gx, gy, gz = self.detector.gravity_acceleration_vector
        drop_x = sc.scalar(0.5 * gx, unit='m/s**2') * (t_flight ** 2)
        drop_y = sc.scalar(0.5 * gy, unit='m/s**2') * (t_flight ** 2)
        drop_z = sc.scalar(0.5 * gz, unit='m/s**2') * (t_flight ** 2)
        
        straight_pos_x = inc_dir_straight_x * L_nom
        straight_pos_y = sc.scalar(0.0, unit='m')
        straight_pos_z = inc_dir_straight_z * L_nom
        
        dropped_pos_x = straight_pos_x + drop_x
        dropped_pos_y = straight_pos_y + drop_y
        dropped_pos_z = straight_pos_z + drop_z
        
        dropped_norm = sc.sqrt(dropped_pos_x**2 + dropped_pos_y**2 + dropped_pos_z**2)
        inc_dir_x = dropped_pos_x / dropped_norm
        inc_dir_y = dropped_pos_y / dropped_norm
        inc_dir_z = dropped_pos_z / dropped_norm

    Qx = (out_dir_x - inc_dir_x) * wavenumber
    Qy = (out_dir_y - inc_dir_y) * wavenumber
    Qz = (out_dir_z - inc_dir_z) * wavenumber

    if scipp_da.bins is not None:
        scipp_da.bins.coords['Qx'] = Qx
        scipp_da.bins.coords['Qy'] = Qy
        scipp_da.bins.coords['Qz'] = Qz
    else:
        scipp_da.coords['Qx'] = Qx
        scipp_da.coords['Qy'] = Qy
        scipp_da.coords['Qz'] = Qz

    return scipp_da

  def calculate_pixel_hit(self, x, y, z, t, VX, VY, VZ):
    """
    Calculate physical detector pixel indices (idx_x_nexus, idx_y_nexus) in raw NeXus frame for all outgoing rays.
    All operations are vectorized across outgoing rays for high performance.
    x, y, z, VX, VY, VZ are in the BornAgain frame.
    Returns (idx_x_nexus, idx_y_nexus, valid_mask, sample_detector_tof).
    """
    sample_detector_tof, x_intersection, y_intersection, z_intersection = self.detector.detector_plane_intersection(x, y, z, VX, VY, VZ, self.sample_detector_distance)
    idx_x_nexus, idx_y_nexus, valid_mask = self.detector.calculate_pixel_hit(x_intersection, y_intersection, z_intersection)
    return idx_x_nexus, idx_y_nexus, valid_mask, sample_detector_tof

  def create_scipp_container(self, tof_bin_edges=None):
    """
    Creates an empty Scipp DataArray to store the simulation results.
    Follows standard NeXus/Scipp conventions: 
    - 1D flattened dimension 'detector_id'
    - 'position' coordinate with sc.vectors (3D positions)
    """
    import scipp as sc

    # Calculate pixel centers
    x_centers = np.linspace(self.detector.min_edge_x_nexus + self.detector.pixel_size_x_nexus/2, 
                            self.detector.max_edge_x_nexus - self.detector.pixel_size_x_nexus/2, 
                            self.detector.pixels_x_nexus)
    y_centers = np.linspace(self.detector.min_edge_y_nexus + self.detector.pixel_size_y_nexus/2, 
                            self.detector.max_edge_y_nexus - self.detector.pixel_size_y_nexus/2, 
                            self.detector.pixels_y_nexus)
    
    # Create 1D flattened positions array (x_index is major axis)
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    num_pixels = self.detector.pixels_x_nexus * self.detector.pixels_y_nexus
    positions = np.zeros((num_pixels, 3))
    positions[:, 0] = X.flatten()
    positions[:, 1] = Y.flatten()
    positions[:, 2] = self.sample_detector_distance

    coords = {
        'position': sc.vectors(dims=['detector_id'], values=positions, unit='m'),
        'sample_position': sc.vector(value=[0, 0, 0], unit='m'),
        'source_position': sc.vector(value=[0, 0, -self.nominal_source_sample_distance], unit='m'),
        'is_tof_instrument': sc.scalar(self.is_tof_instrument),
        'wavelength_selected': sc.scalar(self.wavelength_selected if self.wavelength_selected is not None else 0.0, unit='angstrom'),
        'alpha_inc_deg': sc.scalar(np.rad2deg(self.alpha_inc), unit='deg'),
        'instrument_name': sc.scalar('unknown'),
        'sample_orientation': sc.scalar(self.detector.sample_orientation),
        'beam_angle': sc.scalar(self.beam_angle, unit='deg'),
    }

    if not self.is_tof_instrument:
      values = np.zeros(num_pixels, dtype=np.float64)
      variances = np.zeros(num_pixels, dtype=np.float64)

      return sc.DataArray(
          data=sc.array(dims=['detector_id'], values=values, variances=variances, unit='counts'),
          coords=coords
      )
    else:
      if tof_bin_edges is None:
        raise ValueError("tof_bin_edges must be provided when creating a TOF Scipp container")
      coords['tof'] = sc.array(dims=['tof'], values=tof_bin_edges, unit='s')
      values = np.zeros((num_pixels, len(tof_bin_edges) - 1), dtype=np.float64)
      variances = np.zeros((num_pixels, len(tof_bin_edges) - 1), dtype=np.float64)

      return sc.DataArray(
          data=sc.array(dims=['detector_id', 'tof'], values=values, variances=variances, unit='counts'),
          coords=coords
      )

  def calculate_q_limits(self, wavelength=None):
    """
    Calculate the min and max q values for a wavelength using the xy min and
    max coordinates of the detector (it is an approximation).
    """
    # Since the Detector class constructor already swaps the active area coordinates (size_y_bornagain, size_z_bornagain, min_edge_y_bornagain, min_edge_z_bornagain)
    # in accordance with the sample_orientation, self.detector.min_edge_y/z_bornagain are already in the sample frame.

    # 1. Start with transverse horizontal bounds (Y_BA) which are already in BornAgain frame
    q_min_y_ba = self.detector.min_edge_y_bornagain
    q_max_y_ba = self.detector.max_edge_y_bornagain

    # 2. Project vertical height (Z_BA_uninclined) and longitudinal distance for sample inclination alpha
    q_min_x_ba, q_min_z_ba = self.detector.coords.apply_inclination_angle_transformation(
        self.sample_detector_distance, self.detector.min_edge_z_bornagain
    )
    q_max_x_ba, q_max_z_ba = self.detector.coords.apply_inclination_angle_transformation(
        self.sample_detector_distance, self.detector.max_edge_z_bornagain
    )

    # 3. Combine into coordinate limit vectors in BornAgain space [X_BA, Y_BA, Z_BA].
    q_min_coords = [q_min_x_ba, q_min_y_ba, q_min_z_ba]
    q_max_coords = [q_max_x_ba, q_max_y_ba, q_max_z_ba]

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
    q_min_raw = (outgoing_direction_q_min - incident_direction) * wavenumber
    q_max_raw = (outgoing_direction_q_max - incident_direction) * wavenumber

    # Ensure that q_min strictly contains the minimums and q_max the maximums
    q_min = np.minimum(q_min_raw, q_max_raw)
    q_max = np.maximum(q_min_raw, q_max_raw)

    return q_min, q_max

  def get_q_pixel_limits(self, wavelength=None):
    """
    Calculate and return the Q-space bin edges (q_y, q_z) for each pixel
    boundary on the detector, relying on the detector's pixel dimensions.
    """
    q_min, q_max = self.calculate_q_limits(wavelength)
    q_y = np.linspace(q_min[1], q_max[1], num=self.detector.pixels_y_bornagain + 1)
    q_z = np.linspace(q_min[2], q_max[2], num=self.detector.pixels_z_bornagain + 1)
    return q_y, q_z

  def get_expected_specular_peak_q(self, wavelength=None):
    """Calculate approximate q value for the specular peak (without gravity)"""
    # In BornAgain, specular reflection reverses the vertical component (Z axis)
    outgoing_direction = np.array([self.incident_direction[0], self.incident_direction[1], -self.incident_direction[2]])
    wavenumber = self.get_wavenumber(wavelength)
    specular_peak_expected_q = (outgoing_direction - self.incident_direction) * wavenumber
    print("specular_peak_expected_q", specular_peak_expected_q)

  def get_detector_angle_maximum(self):
    return self.detector.get_detector_angle_maximum(self.sample_detector_distance)

  def get_masked_angle_range(self, mask, factor=1.0):
    return self.detector.get_masked_angle_range(self.sample_detector_distance, mask, factor=factor)