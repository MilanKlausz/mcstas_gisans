"""
This module defines the Detector class, which facilitates the determination of
detection coordinates.
"""

import numpy as np
from typing import Dict, Tuple, Union, Any
import numpy.typing as npt
from .coordinates import CoordinateTransform

class Detector:
    """
    Represents a detector in a neutron scattering instrument.
    Handles coordinate transformations, physical boundaries, smearing, and pixel calculations.

    Attributes
    ----------
    size_x_nexus : float
        Physical width of the detector in NeXus X-axis (m).
    size_y_nexus : float
        Physical height of the detector in NeXus Y-axis (m).
    pixels_x_nexus : int
        Number of pixels along NeXus X-axis.
    pixels_y_nexus : int
        Number of pixels along NeXus Y-axis.
    pixel_size_x_nexus : float
        Size of one pixel along NeXus X-axis (m).
    pixel_size_y_nexus : float
        Size of one pixel along NeXus Y-axis (m).
    direct_beam_centre_offset_x_nexus : float
        Offset of the direct beam center along NeXus X-axis (m).
    direct_beam_centre_offset_y_nexus : float
        Offset of the direct beam center along NeXus Y-axis (m).
    min_edge_x_nexus : float
        Minimum boundary of the detector along NeXus X-axis (m).
    min_edge_y_nexus : float
        Minimum boundary of the detector along NeXus Y-axis (m).
    max_edge_x_nexus : float
        Maximum boundary of the detector along NeXus X-axis (m).
    max_edge_y_nexus : float
        Maximum boundary of the detector along NeXus Y-axis (m).
    coords : CoordinateTransform
        Instance handling coordinate conversions.
    sample_orientation : int
        Sample orientation identifier.
    size_y_bornagain : float
        Detector width in BornAgain frame (m).
    size_z_bornagain : float
        Detector height in BornAgain frame (m).
    pixels_y_bornagain : int
        Number of pixels along BornAgain Y-axis.
    pixels_z_bornagain : int
        Number of pixels along BornAgain Z-axis.
    resolution_y_bornagain : float
        Detector resolution along BornAgain Y-axis (FWHM, m).
    resolution_z_bornagain : float
        Detector resolution along BornAgain Z-axis (FWHM, m).
    direct_beam_centre_offset_y_bornagain : float
        Offset of the direct beam center along BornAgain Y-axis (m).
    direct_beam_centre_offset_z_bornagain : float
        Offset of the direct beam center along BornAgain Z-axis (m).
    pixel_size_y_bornagain : float
        Size of one pixel along BornAgain Y-axis (m).
    pixel_size_z_bornagain : float
        Size of one pixel along BornAgain Z-axis (m).
    min_edge_y_bornagain : float
        Minimum boundary of the detector along BornAgain Y-axis (m).
    min_edge_z_bornagain : float
        Minimum boundary of the detector along BornAgain Z-axis (m).
    max_edge_y_bornagain : float
        Maximum boundary of the detector along BornAgain Y-axis (m).
    max_edge_z_bornagain : float
        Maximum boundary of the detector along BornAgain Z-axis (m).
    sigma_y_bornagain : float
        Standard deviation for smearing along BornAgain Y-axis (m).
    sigma_z_bornagain : float
        Standard deviation for smearing along BornAgain Z-axis (m).
    no_gravity : bool
        Flag indicating whether gravity should be ignored.
    gravity_acceleration_vector : ndarray
        Gravity vector in BornAgain coordinates.
    """

    def __init__(
        self,
        det_params: Dict[str, Any],
        sample_inclination: float,
        sample_orientation: int,
        no_gravity: bool
    ) -> None:
        """
        Initialize the Detector.

        Parameters
        ----------
        det_params : dict
            Dictionary of detector parameters (size, pixels, resolution, direct_beam_centre_offset).
        sample_inclination : float
            Inclination angle of the sample in radians.
        sample_orientation : int
            Orientation identifier for the sample.
        no_gravity : bool
            Whether to disable gravity effects.
        """
        size_nexus = det_params['size']
        pixels_nexus = det_params['pixels']
        res_nexus = det_params['resolution']
        offset_nexus = det_params['direct_beam_centre_offset']

        self.size_x_nexus = float(size_nexus[0])
        self.size_y_nexus = float(size_nexus[1])
        self.pixels_x_nexus = int(pixels_nexus[0])
        self.pixels_y_nexus = int(pixels_nexus[1])
        self.pixel_size_x_nexus = self.size_x_nexus / self.pixels_x_nexus
        self.pixel_size_y_nexus = self.size_y_nexus / self.pixels_y_nexus
        self.direct_beam_centre_offset_x_nexus = float(offset_nexus[0])
        self.direct_beam_centre_offset_y_nexus = float(offset_nexus[1])
        self.min_edge_x_nexus = self.direct_beam_centre_offset_x_nexus - 0.5 * self.size_x_nexus
        self.min_edge_y_nexus = self.direct_beam_centre_offset_y_nexus - 0.5 * self.size_y_nexus
        self.max_edge_x_nexus = self.direct_beam_centre_offset_x_nexus + 0.5 * self.size_x_nexus
        self.max_edge_y_nexus = self.direct_beam_centre_offset_y_nexus + 0.5 * self.size_y_nexus

        self.coords = CoordinateTransform(sample_inclination, sample_orientation)
        self.sample_orientation = sample_orientation

        sy_bornagain, sz_bornagain = self.coords.apply_sample_orientation_transform(size_nexus[0], size_nexus[1])
        self.size_y_bornagain, self.size_z_bornagain = abs(float(sy_bornagain)), abs(float(sz_bornagain))

        py_bornagain, pz_bornagain = self.coords.apply_sample_orientation_transform(pixels_nexus[0], pixels_nexus[1])
        self.pixels_y_bornagain, self.pixels_z_bornagain = abs(int(py_bornagain)), abs(int(pz_bornagain))

        ry_bornagain, rz_bornagain = self.coords.apply_sample_orientation_transform(res_nexus[0], res_nexus[1])
        self.resolution_y_bornagain, self.resolution_z_bornagain = abs(float(ry_bornagain)), abs(float(rz_bornagain))

        oy_bornagain, oz_bornagain = self.coords.apply_sample_orientation_transform(offset_nexus[0], offset_nexus[1])
        self.direct_beam_centre_offset_y_bornagain, self.direct_beam_centre_offset_z_bornagain = float(oy_bornagain), float(oz_bornagain)

        self.pixel_size_y_bornagain = self.size_y_bornagain / self.pixels_y_bornagain
        self.pixel_size_z_bornagain = self.size_z_bornagain / self.pixels_z_bornagain
        self.min_edge_y_bornagain = self.direct_beam_centre_offset_y_bornagain - 0.5 * self.size_y_bornagain
        self.min_edge_z_bornagain = self.direct_beam_centre_offset_z_bornagain - 0.5 * self.size_z_bornagain
        self.max_edge_y_bornagain = self.direct_beam_centre_offset_y_bornagain + 0.5 * self.size_y_bornagain
        self.max_edge_z_bornagain = self.direct_beam_centre_offset_z_bornagain + 0.5 * self.size_z_bornagain
        self.sigma_y_bornagain = self.resolution_y_bornagain / 2.355
        self.sigma_z_bornagain = self.resolution_z_bornagain / 2.355

        self.no_gravity = no_gravity
        if not no_gravity:
            self.gravity_acceleration_vector = self.calculate_gravity_vector()
        else:
            self.gravity_acceleration_vector = np.zeros(3)

    def calculate_gravity_vector(self) -> npt.NDArray[np.float64]:
        """
        Calculate the gravity vector in BornAgain coord system for different sample orientations.

        Returns
        -------
        ndarray
            3D array representing gravity acceleration in BornAgain frame.
        """
        gravity_acceleration = 9.80665  # m/s^2
        gravity_vector_nexus = [0.0, -gravity_acceleration, 0.0]

        gx, gy, gz = self.coords.nexus_to_bornagain(
            gravity_vector_nexus[0], gravity_vector_nexus[1], gravity_vector_nexus[2]
        )

        return np.array([gx, gy, gz], dtype=np.float64)

    def apply_position_smearing(
        self, y_bornagain: Union[float, npt.NDArray], z_bornagain: Union[float, npt.NDArray]
    ) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Apply Gaussian smearing to coordinates in BornAgain frame.

        Parameters
        ----------
        y_bornagain : float or ndarray
            Y-coordinates in BornAgain frame.
        z_bornagain : float or ndarray
            Z-coordinates in BornAgain frame.

        Returns
        -------
        Tuple[float or ndarray, float or ndarray]
            Smeared (y, z) coordinates.
        """
        y_shape = np.shape(y_bornagain)
        z_shape = np.shape(z_bornagain)
        y_smeared = np.random.normal(y_bornagain, self.sigma_y_bornagain, size=y_shape)
        z_smeared = np.random.normal(z_bornagain, self.sigma_z_bornagain, size=z_shape)
        return y_smeared, z_smeared

    def get_pixel_indices_from_position(
        self, x_nexus: Union[float, npt.NDArray], y_nexus: Union[float, npt.NDArray]
    ) -> Tuple[Union[int, npt.NDArray], Union[int, npt.NDArray], Union[bool, npt.NDArray]]:
        """
        Find 0-indexed integer pixel indices (idx_x, idx_y) corresponding to positions (x, y)
        in the raw physical NeXus detector frame.

        Parameters
        ----------
        x_nexus : float or ndarray
            X positions in NeXus frame.
        y_nexus : float or ndarray
            Y positions in NeXus frame.

        Returns
        -------
        Tuple[int or ndarray, int or ndarray, bool or ndarray]
            (idx_x, idx_y, valid_mask). valid_mask is True for hits within detector bounds.
        """
        idx_x = np.floor((x_nexus - self.min_edge_x_nexus) / self.pixel_size_x_nexus).astype(int)
        idx_y = np.floor((y_nexus - self.min_edge_y_nexus) / self.pixel_size_y_nexus).astype(int)
        valid_mask = (idx_x >= 0) & (idx_x < self.pixels_x_nexus) & (idx_y >= 0) & (idx_y < self.pixels_y_nexus)
        return idx_x, idx_y, valid_mask

    def calculate_pixel_hit(
        self,
        x_intersection_bornagain: Union[float, npt.NDArray],
        y_intersection_bornagain: Union[float, npt.NDArray],
        z_intersection_bornagain: Union[float, npt.NDArray]
    ) -> Tuple[Union[int, npt.NDArray], Union[int, npt.NDArray], Union[bool, npt.NDArray]]:
        """
        Calculate physical detector pixel indices in raw NeXus frame for intersection coordinates.
        Operations are vectorized across outgoing rays.

        Parameters
        ----------
        x_intersection_bornagain : float or ndarray
            X-coordinate of intersection in BornAgain frame.
        y_intersection_bornagain : float or ndarray
            Y-coordinate of intersection in BornAgain frame.
        z_intersection_bornagain : float or ndarray
            Z-coordinate of intersection in BornAgain frame.

        Returns
        -------
        Tuple[int or ndarray, int or ndarray, bool or ndarray]
            (idx_x_nexus, idx_y_nexus, valid_mask)
        """
        _, z_int_un = self.coords.apply_inverse_inclination_angle_transformation(
            x_intersection_bornagain, z_intersection_bornagain
        )
        y_smeared, z_smeared_un = self.apply_position_smearing(y_intersection_bornagain, z_int_un)

        idx_y_bornagain = np.floor((y_smeared - self.min_edge_y_bornagain) / self.pixel_size_y_bornagain).astype(int)
        idx_z_bornagain = np.floor((z_smeared_un - self.min_edge_z_bornagain) / self.pixel_size_z_bornagain).astype(int)
        
        valid_mask = (
            (idx_y_bornagain >= 0) & (idx_y_bornagain < self.pixels_y_bornagain) &
            (idx_z_bornagain >= 0) & (idx_z_bornagain < self.pixels_z_bornagain)
        )

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
            case _:
                idx_x_nexus = idx_y_bornagain
                idx_y_nexus = idx_z_bornagain

        return idx_x_nexus, idx_y_nexus, valid_mask

    def get_pixel_centre_from_position(
        self, y_bornagain: Union[float, npt.NDArray], z_bornagain: Union[float, npt.NDArray]
    ) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Find the centre of the pixel corresponding to the y, z coordinates in BornAgain frame.

        Parameters
        ----------
        y_bornagain : float or ndarray
            Y-coordinate in BornAgain frame.
        z_bornagain : float or ndarray
            Z-coordinate in BornAgain frame.

        Returns
        -------
        Tuple[float or ndarray, float or ndarray]
            Pixel center (y, z) coordinates.
        """
        y_pixel_centre = np.floor((y_bornagain - self.min_edge_y_bornagain) / self.pixel_size_y_bornagain) * self.pixel_size_y_bornagain + 0.5 * self.pixel_size_y_bornagain + self.min_edge_y_bornagain
        z_pixel_centre = np.floor((z_bornagain - self.min_edge_z_bornagain) / self.pixel_size_z_bornagain) * self.pixel_size_z_bornagain + 0.5 * self.pixel_size_z_bornagain + self.min_edge_z_bornagain
        return y_pixel_centre, z_pixel_centre

    def calculate_gravity_drop(self, t_propagate: Union[float, npt.NDArray]) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Calculate the effect of gravity during the propagation to detector surface.

        Parameters
        ----------
        t_propagate : float or ndarray
            Time of propagation to the detector.

        Returns
        -------
        Tuple[float or ndarray, float or ndarray, float or ndarray]
            Positional drop (x, y, z) due to gravity.
        """
        if self.no_gravity:
            zero = np.zeros_like(t_propagate, dtype=np.float64)
            return zero, zero, zero

        t_sq_half = 0.5 * t_propagate**2
        x_drop = self.gravity_acceleration_vector[0] * t_sq_half
        y_drop = self.gravity_acceleration_vector[1] * t_sq_half
        z_drop = self.gravity_acceleration_vector[2] * t_sq_half
        return x_drop, y_drop, z_drop

    def detector_plane_intersection(
        self,
        x: Union[float, npt.NDArray],
        y: Union[float, npt.NDArray],
        z: Union[float, npt.NDArray],
        vx: Union[float, npt.NDArray],
        vy: Union[float, npt.NDArray],
        vz: Union[float, npt.NDArray],
        sample_detector_distance: float
    ) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray], Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Calculate x,y,z position on the detector surface and the corresponding TOF.
        Assuming the detector surface is vertical in the NeXus coord system.

        Parameters
        ----------
        x, y, z : float or ndarray
            Initial position.
        vx, vy, vz : float or ndarray
            Velocity components.
        sample_detector_distance : float
            Distance from sample to detector.

        Returns
        -------
        Tuple[float or ndarray, float or ndarray, float or ndarray, float or ndarray]
            (t_propagate, x_intersection, y_intersection, z_intersection)
        """
        z_nexus, _ = self.coords.apply_inverse_inclination_angle_transformation(x, z)
        vz_nexus, _ = self.coords.apply_inverse_inclination_angle_transformation(vx, vz)
        t_propagate = (sample_detector_distance - z_nexus) / vz_nexus

        x_int = vx * t_propagate + x
        y_int = vy * t_propagate + y
        z_int = vz * t_propagate + z
        
        if not self.no_gravity:
            x_drop, y_drop, z_drop = self.calculate_gravity_drop(t_propagate)
            x_int += x_drop
            y_int += y_drop
            z_int += z_drop

        return t_propagate, x_int, y_int, z_int

    def calculate_detection_coordinate(
        self,
        x_intersection_bornagain: Union[float, npt.NDArray],
        y_intersection_bornagain: Union[float, npt.NDArray],
        z_intersection_bornagain: Union[float, npt.NDArray]
    ) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Get the coordinate of the detection event taking into account detector resolution.

        Parameters
        ----------
        x_intersection_bornagain : float or ndarray
            X-coordinate of theoretical intersection.
        y_intersection_bornagain : float or ndarray
            Y-coordinate of theoretical intersection.
        z_intersection_bornagain : float or ndarray
            Z-coordinate of theoretical intersection.

        Returns
        -------
        Tuple[float or ndarray, float or ndarray, float or ndarray]
            Smeared detection coordinates (x, y, z).
        """
        x_int_un, z_int_un = self.coords.apply_inverse_inclination_angle_transformation(
            x_intersection_bornagain, z_intersection_bornagain
        )
        y_det, z_det_un = self.apply_position_smearing(y_intersection_bornagain, z_int_un)
        y_pix, z_pix_un = self.get_pixel_centre_from_position(y_det, z_det_un)
        x_pix, z_pix = self.coords.apply_inclination_angle_transformation(x_int_un, z_pix_un)

        return x_pix, y_pix, z_pix

    def calculate_angles_from_spatial_bounds(
        self,
        sample_detector_distance: float,
        y_min_bornagain: float,
        y_max_bornagain: float,
        z_min_bornagain_uninclined: float,
        z_max_bornagain_uninclined: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate opening angles [horiz_min, horiz_max, vert_min, vert_max] in degrees
        for given boundaries.

        Parameters
        ----------
        sample_detector_distance : float
            Distance from sample to detector.
        y_min_bornagain : float
            Minimum horizontal bound.
        y_max_bornagain : float
            Maximum horizontal bound.
        z_min_bornagain_uninclined : float
            Minimum vertical bound (uninclined).
        z_max_bornagain_uninclined : float
            Maximum vertical bound (uninclined).

        Returns
        -------
        Tuple[float, float, float, float]
            (angle_horiz_min, angle_horiz_max, angle_vert_min, angle_vert_max)
        """
        h_min = float(np.rad2deg(np.arctan2(y_min_bornagain, sample_detector_distance)))
        h_max = float(np.rad2deg(np.arctan2(y_max_bornagain, sample_detector_distance)))

        x_top, z_top = self.coords.apply_inclination_angle_transformation(
            sample_detector_distance, z_max_bornagain_uninclined
        )
        x_bot, z_bot = self.coords.apply_inclination_angle_transformation(
            sample_detector_distance, z_min_bornagain_uninclined
        )

        z_ang_top = np.arctan2(z_top, x_top)
        z_ang_bot = np.arctan2(z_bot, x_bot)

        if isinstance(z_ang_top, np.ndarray):
            z_ang_top = z_ang_top.item()
        if isinstance(z_ang_bot, np.ndarray):
            z_ang_bot = z_ang_bot.item()

        v_min = float(np.rad2deg(min(z_ang_bot, z_ang_top)))
        v_max = float(np.rad2deg(max(z_ang_bot, z_ang_top)))

        return h_min, h_max, v_min, v_max

    def get_detector_angle_maximum(self, sample_detector_distance: float) -> Tuple[float, float, float, float]:
        """
        Calculate the 4 opening angles covered by the detector (in deg).

        Parameters
        ----------
        sample_detector_distance : float
            Distance from sample to detector.

        Returns
        -------
        Tuple[float, float, float, float]
            (horiz_min, horiz_max, vert_min, vert_max)
        """
        return self.calculate_angles_from_spatial_bounds(
            sample_detector_distance,
            self.min_edge_y_bornagain, self.max_edge_y_bornagain,
            self.min_edge_z_bornagain, self.max_edge_z_bornagain
        )

    def get_masked_angle_range(
        self, sample_detector_distance: float, mask: npt.NDArray[np.bool_], factor: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """
        Calculate minimum opening angles enclosing all unmasked (True) pixels.

        Parameters
        ----------
        sample_detector_distance : float
            Distance to detector.
        mask : ndarray
            Boolean mask array.
        factor : float, optional
            Scaling factor for margin (default 1.0).

        Returns
        -------
        Tuple[float, float, float, float]
            (horiz_min, horiz_max, vert_min, vert_max)
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
        z_min_un = y_nexus_edges[i_min]
        z_max_un = y_nexus_edges[i_max + 1]

        h_min, h_max, v_min, v_max = self.calculate_angles_from_spatial_bounds(
            sample_detector_distance, y_min_bornagain, y_max_bornagain, z_min_un, z_max_un
        )

        if factor != 1.0:
            h_center = 0.5 * (h_min + h_max)
            h_half = 0.5 * (h_max - h_min) * factor
            h_min, h_max = h_center - h_half, h_center + h_half

            v_center = 0.5 * (v_min + v_max)
            v_half = 0.5 * (v_max - v_min) * factor
            v_min, v_max = v_center - v_half, v_center + v_half

        return h_min, h_max, v_min, v_max

    def get_pixel_positions(self, sample_detector_distance: float) -> npt.NDArray[np.float64]:
        """
        Get the physical coordinates (X, Y, Z) of all pixel centers in NeXus frame.

        Parameters
        ----------
        sample_detector_distance : float
            Distance from sample to detector.

        Returns
        -------
        ndarray
            Array of shape (N_pixels, 3) with positions.
        """
        x_centers = np.linspace(
            self.min_edge_x_nexus + self.pixel_size_x_nexus / 2,
            self.max_edge_x_nexus - self.pixel_size_x_nexus / 2,
            self.pixels_x_nexus
        )
        y_centers = np.linspace(
            self.min_edge_y_nexus + self.pixel_size_y_nexus / 2,
            self.max_edge_y_nexus - self.pixel_size_y_nexus / 2,
            self.pixels_y_nexus
        )
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
        num_pixels = self.pixels_x_nexus * self.pixels_y_nexus
        positions = np.zeros((num_pixels, 3))
        positions[:, 0] = X.flatten()
        positions[:, 1] = Y.flatten()
        positions[:, 2] = sample_detector_distance
        return positions