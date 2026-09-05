"""
This module defines the Instrument class, which is mainly intended for
scattering vector (Q) related calculations and coordinate management.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import numpy.typing as npt

from .detector import Detector
from .particle_calculations import calculate_neutron_wavelength, calculate_wavenumber, calculate_neutron_velocity

class Instrument:
    """
    Instrument representation for configuring and simulating a generic neutron scattering
    instrument. Manages the geometry, wave vector calculations, pixel interactions, and
    gravity corrections.

    Attributes
    ----------
    beam_angle : float
        Beam angle in degrees.
    detector : Detector
        Detector instance used for this instrument.
    nominal_source_sample_distance : float
        Nominal distance from source to sample (m).
    sample_detector_distance : float
        Distance from sample to detector (m).
    no_gravity : bool
        Flag indicating whether gravity should be ignored.
    alpha_inc : float
        Sample inclination angle in radians.
    wavelength_selected : Optional[float]
        Selected wavelength for the instrument (Angstrom).
    incident_direction : ndarray
        The 3D vector of incident neutron direction in BornAgain frame.
    is_tof_instrument : bool
        Whether the instrument operates in Time-of-Flight mode.
    wavenumber_fixed : float
        Fixed wavenumber if not a TOF instrument.
    """

    def __init__(
        self,
        instr_params: Dict[str, Any],
        alpha_inc_deg: float,
        wavelength_selected: Optional[float],
        sample_orientation: int,
        wfm: bool = False,
        no_gravity: bool = False
    ) -> None:
        """
        Initialize the Instrument.

        Parameters
        ----------
        instr_params : dict
            Dictionary of instrument configuration parameters.
        alpha_inc_deg : float
            Inclination angle of the sample in degrees.
        wavelength_selected : Optional[float]
            The designated wavelength (Angstrom), or None if purely TOF without reference.
        sample_orientation : int
            Sample orientation identifier.
        wfm : bool, optional
            Wavelength frame multiplication (True/False). Default is False.
        no_gravity : bool, optional
            Disable gravity calculations. Default is False.
        """
        self.beam_angle = float(instr_params.get('beam_angle', 0.0))
        sample_inclination = float(np.deg2rad(alpha_inc_deg + self.beam_angle))
        
        self.detector = Detector(instr_params['detector'], sample_inclination, sample_orientation, no_gravity)

        self.nominal_source_sample_distance = float(instr_params['nominal_source_sample_distance']) - (0.0 if not wfm else float(instr_params['wfm_virtual_source_distance']))
        self.sample_detector_distance = float(instr_params['sample_detector_distance'])

        self.no_gravity = no_gravity
        self.alpha_inc = float(np.deg2rad(alpha_inc_deg))
        self.wavelength_selected = wavelength_selected

        self.incident_direction = self.calculate_incident_direction(wavelength_selected)

        self.is_tof_instrument = bool(instr_params['tof_instrument'])
        if not self.is_tof_instrument:
            if wavelength_selected is not None:
                self.wavenumber_fixed = float(calculate_wavenumber(wavelength_selected))
            else:
                self.wavenumber_fixed = 0.0

    def calculate_incident_direction(self, wavelength: Optional[float]) -> npt.NDArray[np.float64]:
        """
        Calculate the reference incident direction, taking gravity drop into account
        from the sample to the detector surface if needed.

        Parameters
        ----------
        wavelength : Optional[float]
            The wavelength of the neutron in Angstrom.

        Returns
        -------
        ndarray
            Normalized 3D vector representing the incident direction.
        """
        incident_dir_straight = np.array([np.cos(self.alpha_inc), 0.0, -np.sin(self.alpha_inc)], dtype=np.float64)
        if self.no_gravity or wavelength is None:
            return incident_dir_straight

        t_flight = self.sample_detector_distance / calculate_neutron_velocity(wavelength)
        drop_vector = 0.5 * self.detector.gravity_acceleration_vector * t_flight**2
        straight_pos = incident_dir_straight * self.sample_detector_distance
        dropped_pos = straight_pos + drop_vector
        norm = float(np.linalg.norm(dropped_pos))
        if norm == 0:
            return dropped_pos
        return dropped_pos / norm

    def get_wavenumber(self, wavelength: Optional[float]) -> float:
        """
        Return the wavenumber that is fixed in case of non-TOF instrument,
        or calculated for a specific wavelength.

        Parameters
        ----------
        wavelength : Optional[float]
            Wavelength in Angstrom.

        Returns
        -------
        float
            Wavenumber in inverse Angstrom.
        """
        if wavelength is None:
            wavelength = self.wavelength_selected

        if wavelength is None:
            if not self.is_tof_instrument:
                return self.wavenumber_fixed
            raise ValueError(
                "get_wavenumber() requires a wavelength for a TOF instrument, but none was "
                "provided and self.wavelength_selected is also None."
            )

        return calculate_wavenumber(wavelength) if self.is_tof_instrument else self.wavenumber_fixed

    def compute_q_scipp(self, scipp_da: Any) -> Any:
        """
        Calculate Q values from positions at the detector surface using Scipp.
        Applies small angle approximations or full vector operations including gravity drop.

        Parameters
        ----------
        scipp_da : Any
            Scipp DataArray containing positions and coordinates.

        Returns
        -------
        Any
            Updated Scipp DataArray with Qx, Qy, Qz coordinates attached.
        """
        import scipp as sc
        import scippneutron as scn
        import scipy.constants as const

        h_over_m = sc.scalar(const.h / const.m_n, unit='m**2/s')

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

        pos_ba_x = x_ba_uninclined * cos_a + z_ba_uninclined * sin_a
        pos_ba_y = y_ba_uninclined
        pos_ba_z = -x_ba_uninclined * sin_a + z_ba_uninclined * cos_a

        L2 = sc.sqrt(pos_ba_x**2 + pos_ba_y**2 + pos_ba_z**2)
        out_dir_x = pos_ba_x / L2
        out_dir_y = pos_ba_y / L2
        out_dir_z = pos_ba_z / L2

        if self.is_tof_instrument:
            needs_wavelength = (
                (scipp_da.bins is not None and 'wavelength' not in scipp_da.bins.coords) or
                (scipp_da.bins is None and 'wavelength' not in scipp_da.coords)
            )
            if needs_wavelength:
                # scn.convert needs sample_position/source_position to compute the
                # flight path length; these aren't part of the saved event data
                # itself (only of the separate 'instrument' metadata group), so
                # attach them here from self.nominal_source_sample_distance, which
                # is already adjusted for WFM mode (shorter virtual-source distance)
                # at construction time.
                if 'sample_position' not in scipp_da.coords:
                    scipp_da.coords['sample_position'] = sc.vector(value=[0.0, 0.0, 0.0], unit='m')
                if 'source_position' not in scipp_da.coords:
                    scipp_da.coords['source_position'] = sc.vector(value=[0.0, 0.0, -self.nominal_source_sample_distance], unit='m')

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
                wavelength = sc.scalar(1.0, unit='angstrom') 
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

    def calculate_pixel_hit(
        self,
        x: Union[float, npt.NDArray],
        y: Union[float, npt.NDArray],
        z: Union[float, npt.NDArray],
        t: Union[float, npt.NDArray],
        vx: Union[float, npt.NDArray],
        vy: Union[float, npt.NDArray],
        vz: Union[float, npt.NDArray]
    ) -> Tuple[Union[int, npt.NDArray], Union[int, npt.NDArray], Union[bool, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Calculate physical detector pixel indices in raw NeXus frame for all outgoing rays.
        All operations are vectorized. x, y, z, vx, vy, vz are in the BornAgain frame.

        Parameters
        ----------
        x, y, z : float or ndarray
            Position coordinates in BornAgain frame.
        t : float or ndarray
            Time.
        vx, vy, vz : float or ndarray
            Velocities in BornAgain frame.

        Returns
        -------
        Tuple[int or ndarray, int or ndarray, bool or ndarray, float or ndarray]
            (idx_x_nexus, idx_y_nexus, valid_mask, sample_detector_tof)
        """
        sample_detector_tof, x_int, y_int, z_int = self.detector.detector_plane_intersection(
            x, y, z, vx, vy, vz, self.sample_detector_distance
        )
        idx_x, idx_y, mask = self.detector.calculate_pixel_hit(x_int, y_int, z_int)
        return idx_x, idx_y, mask, sample_detector_tof

    def create_scipp_container(self, tof_bin_edges: Optional[npt.NDArray] = None) -> Any:
        """
        Creates an empty Scipp DataArray to store the simulation results.
        Follows standard NeXus/Scipp conventions.

        Parameters
        ----------
        tof_bin_edges : ndarray, optional
            Array of Time-of-Flight bin edges if TOF instrument.

        Returns
        -------
        scipp.DataArray
            Empty initialized dataset.
        """
        import scipp as sc

        x_centers = np.linspace(
            self.detector.min_edge_x_nexus + self.detector.pixel_size_x_nexus/2, 
            self.detector.max_edge_x_nexus - self.detector.pixel_size_x_nexus/2, 
            self.detector.pixels_x_nexus
        )
        y_centers = np.linspace(
            self.detector.min_edge_y_nexus + self.detector.pixel_size_y_nexus/2, 
            self.detector.max_edge_y_nexus - self.detector.pixel_size_y_nexus/2, 
            self.detector.pixels_y_nexus
        )
        
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
        num_pixels = self.detector.pixels_x_nexus * self.detector.pixels_y_nexus
        positions = np.zeros((num_pixels, 3))
        positions[:, 0] = X.flatten()
        positions[:, 1] = Y.flatten()
        positions[:, 2] = self.sample_detector_distance

        ws = self.wavelength_selected if self.wavelength_selected is not None else 0.0

        coords = {
            'position': sc.vectors(dims=['detector_id'], values=positions, unit='m'),
            'sample_position': sc.vector(value=[0, 0, 0], unit='m'),
            'source_position': sc.vector(value=[0, 0, -self.nominal_source_sample_distance], unit='m'),
            'is_tof_instrument': sc.scalar(self.is_tof_instrument),
            'wavelength_selected': sc.scalar(ws, unit='angstrom'),
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

    def calculate_q_limits(self, wavelength: Optional[float] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Calculate the min and max q values for a given wavelength based on the detector bounds.

        Parameters
        ----------
        wavelength : Optional[float]
            Wavelength in Angstrom.

        Returns
        -------
        Tuple[ndarray, ndarray]
            (q_min, q_max) coordinates as 3D arrays.
        """
        q_min_y_bornagain = self.detector.min_edge_y_bornagain
        q_max_y_bornagain = self.detector.max_edge_y_bornagain

        q_min_x_bornagain, q_min_z_bornagain = self.detector.coords.apply_inclination_angle_transformation(
            self.sample_detector_distance, self.detector.min_edge_z_bornagain
        )
        q_max_x_bornagain, q_max_z_bornagain = self.detector.coords.apply_inclination_angle_transformation(
            self.sample_detector_distance, self.detector.max_edge_z_bornagain
        )

        q_min_coords = np.array([q_min_x_bornagain, q_min_y_bornagain, q_min_z_bornagain], dtype=np.float64)
        q_max_coords = np.array([q_max_x_bornagain, q_max_y_bornagain, q_max_z_bornagain], dtype=np.float64)

        if not self.no_gravity:
            import scipy.constants as const
            w = wavelength if wavelength is not None else self.wavelength_selected
            if w is not None and w > 0:
                velocity = (const.h / const.m_n) / (w * 1e-10)
                t_flight_min = float(np.linalg.norm(q_min_coords) / velocity)
                t_flight_max = float(np.linalg.norm(q_max_coords) / velocity)
                
                g_vec = self.detector.gravity_acceleration_vector
                q_min_coords -= 0.5 * g_vec * t_flight_min**2
                q_max_coords -= 0.5 * g_vec * t_flight_max**2

        outgoing_direction_q_min = q_min_coords / float(np.linalg.norm(q_min_coords))
        outgoing_direction_q_max = q_max_coords / float(np.linalg.norm(q_max_coords))

        wavenumber = self.get_wavenumber(wavelength)

        if not self.is_tof_instrument:
            w = wavelength if wavelength is not None else self.wavelength_selected
            incident_direction = self.calculate_incident_direction(w)
        else:
            incident_direction = self.incident_direction

        q_min_raw = (outgoing_direction_q_min - incident_direction) * wavenumber
        q_max_raw = (outgoing_direction_q_max - incident_direction) * wavenumber

        q_min = np.minimum(q_min_raw, q_max_raw)
        q_max = np.maximum(q_min_raw, q_max_raw)

        return q_min, q_max

    def get_q_pixel_limits(self, wavelength: Optional[float] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Calculate and return the Q-space bin edges (q_y, q_z) for each pixel.

        Parameters
        ----------
        wavelength : Optional[float]
            Wavelength in Angstrom.

        Returns
        -------
        Tuple[ndarray, ndarray]
            (q_y, q_z) edge arrays.
        """
        q_min, q_max = self.calculate_q_limits(wavelength)
        q_y = np.linspace(q_min[1], q_max[1], num=self.detector.pixels_y_bornagain + 1)
        q_z = np.linspace(q_min[2], q_max[2], num=self.detector.pixels_z_bornagain + 1)
        return q_y, q_z

    def get_expected_specular_peak_q(self, wavelength: Optional[float] = None) -> npt.NDArray[np.float64]:
        """
        Calculate and return approximate q value for the specular peak (without gravity).

        Parameters
        ----------
        wavelength : Optional[float]
            Wavelength in Angstrom.

        Returns
        -------
        ndarray
            3D array representing the expected q vector of the specular peak.
        """
        outgoing_direction = np.array([self.incident_direction[0], self.incident_direction[1], -self.incident_direction[2]])
        wavenumber = self.get_wavenumber(wavelength)
        specular_peak_expected_q = (outgoing_direction - self.incident_direction) * wavenumber
        print("specular_peak_expected_q", specular_peak_expected_q)
        return specular_peak_expected_q

    def get_detector_angle_maximum(self) -> Tuple[float, float, float, float]:
        """
        Calculate maximum opening angles of the detector.

        Returns
        -------
        Tuple[float, float, float, float]
            (horiz_min, horiz_max, vert_min, vert_max)
        """
        return self.detector.get_detector_angle_maximum(self.sample_detector_distance)

    def get_masked_angle_range(self, mask: npt.NDArray[np.bool_], factor: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Calculate angular range for masked pixels.

        Parameters
        ----------
        mask : ndarray
            Boolean mask.
        factor : float, optional
            Margin factor (default 1.0).

        Returns
        -------
        Tuple[float, float, float, float]
            (horiz_min, horiz_max, vert_min, vert_max)
        """
        return self.detector.get_masked_angle_range(self.sample_detector_distance, mask, factor=factor)