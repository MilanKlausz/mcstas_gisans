"""
This module defines the Instrument class, which is mainly intended for
scattering vector (Q) related calculations and coordinate management.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import numpy.typing as npt

from .detector import Detector
from .particle_calculations import calculate_wavenumber, calculate_neutron_velocity

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

    def calculate_incident_direction(self, wavelength: Optional[float] = None) -> npt.NDArray[np.float64]:
        """
        Incident beam direction at the sample in the BornAgain frame: (cos a, 0, -sin a).

        Q convention (the Mantid Q1D/Qxy and scippneutron/esssans convention): the incident
        direction is the straight beam axis at the sample (no gravity term); gravity is
        applied once, on the outgoing side, by using the launch direction of the scattered
        neutron (see calculate_q_from_nexus_positions). The wavelength argument is ignored
        and kept for backwards compatibility.
        """
        return np.array([np.cos(self.alpha_inc), 0.0, -np.sin(self.alpha_inc)], dtype=np.float64)

    def calculate_q_from_nexus_positions(
        self, positions_nexus: npt.NDArray, wavelength: Union[float, npt.NDArray]
    ) -> npt.NDArray[np.float64]:
        """
        Scattering vectors Q [1/nm] (BornAgain frame) for detection points given in the NeXus frame.

        Uses only quantities known in a real measurement: the detection point, the wavelength
        (selected wavelength, or derived from the time of flight), the incident angle and the
        detector geometry. The scattered neutron left the sample in its launch direction,
        i.e. towards the detection point raised against gravity by the drop accumulated over
        the (straight-line) flight path:  u_out ~ P - 1/2 g t^2,  t = |P| / v(lambda).
        Q = k (u_out - u_in) with u_in the straight incident direction at the sample.
        With a detector offset defined relative to the undeflected beam axis, the unscattered
        beam therefore maps to Q = 0 at every wavelength.

        Parameters
        ----------
        positions_nexus : ndarray, shape (..., 3)
            Detection points in the NeXus frame relative to the sample [m].
        wavelength : float or ndarray broadcastable to positions_nexus[..., 0]
            Wavelength in Angstrom.
        """
        pos = np.asarray(positions_nexus, dtype=np.float64)
        x, y, z = self.detector.coords.nexus_to_bornagain(pos[..., 0], pos[..., 1], pos[..., 2])
        launch = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
        wavelength = np.asarray(wavelength, dtype=np.float64)
        if not self.no_gravity:
            t = np.linalg.norm(launch, axis=-1) / calculate_neutron_velocity(wavelength)
            launch = launch - 0.5 * self.detector.gravity_acceleration_vector * (t**2)[..., np.newaxis]
        u_out = launch / np.linalg.norm(launch, axis=-1, keepdims=True)
        k = calculate_wavenumber(wavelength)
        return np.asarray(k)[..., np.newaxis] * (u_out - self.calculate_incident_direction())

    def direct_beam_landing_point_nexus(self, wavelength: Optional[float]) -> npt.NDArray[np.float64]:
        """
        Point (NeXus frame, z = sample-detector distance) where the unscattered beam hits the
        detector plane: the undeflected beam axis plus the gravity drop over the flight path.
        """
        ux, uy, uz = self.detector.coords.bornagain_to_nexus(*self.calculate_incident_direction())
        L = self.sample_detector_distance
        point = np.array([ux, uy, uz]) * (L / uz)
        if not self.no_gravity and wavelength is not None:
            t = L / (uz * calculate_neutron_velocity(wavelength))
            point[1] -= 0.5 * 9.80665 * t**2
        return point

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

        # Pixel positions (NeXus frame) -> BornAgain frame with the shared CoordinateTransform
        pos = scipp_da.coords['position']
        pos_values = np.asarray(pos.values)
        px, py, pz = self.detector.coords.nexus_to_bornagain(pos_values[..., 0], pos_values[..., 1], pos_values[..., 2])
        P = [sc.array(dims=pos.dims, values=np.broadcast_to(c, pos.shape).copy(), unit='m') for c in (px, py, pz)]

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
            if self.wavelength_selected is None:
                raise ValueError("compute_q_scipp requires wavelength_selected for a non-TOF instrument.")
            wavelength = sc.scalar(self.wavelength_selected, unit='angstrom')

        wavelength = sc.to_unit(wavelength, 'angstrom')

        # Launch direction of the scattered neutron (see calculate_q_from_nexus_positions)
        launch = list(P)
        if not self.no_gravity:
            path = sc.sqrt(P[0]**2 + P[1]**2 + P[2]**2)
            t = path / (h_over_m / sc.to_unit(wavelength, 'm'))
            half_t2 = 0.5 * t**2
            g = self.detector.gravity_acceleration_vector
            launch = [P[i] - sc.scalar(g[i], unit='m/s**2') * half_t2 for i in range(3)]
        norm = sc.sqrt(launch[0]**2 + launch[1]**2 + launch[2]**2)

        wavenumber = 2.0 * np.pi / wavelength
        u_in = self.calculate_incident_direction()
        Qx = (launch[0] / norm - u_in[0]) * wavenumber
        Qy = (launch[1] / norm - u_in[1]) * wavenumber
        Qz = (launch[2] / norm - u_in[2]) * wavenumber

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

    def get_q_pixel_limits(self, wavelength: Optional[float] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Q-space bin edges (q_y, q_z) [1/nm] of the detector pixels in the (uninclined)
        BornAgain frame, for plotting and Q-defined masks.

        Each edge is evaluated exactly with calculate_q_from_nexus_positions, along the two
        lines through the point where the unscattered beam hits the detector (so that the
        direct beam is exactly at Q = 0). Using separable 1D edges is an approximation for
        points far from these lines (second order in the scattering angle).

        Parameters
        ----------
        wavelength : Optional[float]
            Wavelength in Angstrom (default: wavelength_selected).
        """
        w = wavelength if wavelength is not None else self.wavelength_selected
        if w is None:
            raise ValueError("get_q_pixel_limits requires a wavelength (none given and wavelength_selected is None).")
        det = self.detector
        coords = det.coords
        L = self.sample_detector_distance

        landing = self.direct_beam_landing_point_nexus(w)
        y_ref, z_ref = coords.apply_sample_orientation_transform(landing[0], landing[1])

        y_edges = np.linspace(det.min_edge_y_bornagain, det.max_edge_y_bornagain, det.pixels_y_bornagain + 1)
        z_edges = np.linspace(det.min_edge_z_bornagain, det.max_edge_z_bornagain, det.pixels_z_bornagain + 1)

        def to_nexus(y_ba, z_ba):
            x_nx, y_nx = coords._apply_inverse_sample_orientation_transform(y_ba, z_ba)
            x_nx, y_nx = np.broadcast_arrays(x_nx, y_nx)
            return np.stack([x_nx, y_nx, np.full_like(x_nx, L, dtype=np.float64)], axis=-1)

        q_y = self.calculate_q_from_nexus_positions(to_nexus(y_edges, np.full_like(y_edges, z_ref)), w)[:, 1]
        q_z = self.calculate_q_from_nexus_positions(to_nexus(np.full_like(z_edges, y_ref), z_edges), w)[:, 2]
        return q_y, q_z

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