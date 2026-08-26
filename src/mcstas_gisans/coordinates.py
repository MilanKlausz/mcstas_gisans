import numpy as np
from typing import Tuple, Union, Callable
import numpy.typing as npt

class CoordinateTransform:
    """
    Handles coordinate transformations between NeXus coordinate system
    and BornAgain sample-centric coordinate system, taking into account:
    1. Sample orientation (rotation around the beam axis Z_nexus by 0, +/-90 degrees)
    2. Sample inclination angle alpha (rotation in the vertical-longitudinal plane)

    Attributes
    ----------
    sample_inclination : float
        The inclination angle of the sample in radians.
    sample_orientation : int
        The orientation of the sample:
        0 = Vertical sample, beam from left (-90 deg rotation)
        1 = Horizontal sample (no rotation)
        2 = Vertical sample, beam from right (+90 deg rotation)
    inverse_sample_inclination_rotation_matrix : ndarray
        2x2 rotation matrix for inverse sample inclination.
    sample_inclination_rotation_matrix : ndarray
        2x2 rotation matrix for forward sample inclination.
    apply_sample_orientation_transform : Callable
        Pre-bound method for applying sample orientation transformation.
    _apply_inverse_sample_orientation_transform : Callable
        Pre-bound method for applying inverse sample orientation transformation.
    """

    def __init__(self, sample_inclination: float, sample_orientation: int) -> None:
        """
        Initialize the CoordinateTransform object.

        Parameters
        ----------
        sample_inclination : float
            Sample inclination angle (alpha) in radians.
        sample_orientation : int
            Sample orientation code (0, 1, or 2).
        """
        self.sample_inclination = sample_inclination
        self.sample_orientation = sample_orientation

        # 2D Rotation matrices for sample inclination angle alpha
        self.inverse_sample_inclination_rotation_matrix = np.array([
            [np.cos(sample_inclination), -np.sin(sample_inclination)],
            [np.sin(sample_inclination), np.cos(sample_inclination)],
        ])

        self.sample_inclination_rotation_matrix = np.array([
            [np.cos(-sample_inclination), -np.sin(-sample_inclination)],
            [np.sin(-sample_inclination), np.cos(-sample_inclination)],
        ])

        # Pre-bind sample orientation dependent methods to eliminate branching
        match sample_orientation:
            case 0:
                self.apply_sample_orientation_transform = self._transform_sample_orient_0
                self._apply_inverse_sample_orientation_transform = self._inverse_sample_orient_0
            case 1:
                self.apply_sample_orientation_transform = self._transform_sample_orient_1
                self._apply_inverse_sample_orientation_transform = self._inverse_sample_orient_1
            case 2:
                self.apply_sample_orientation_transform = self._transform_sample_orient_2
                self._apply_inverse_sample_orientation_transform = self._inverse_sample_orient_2
            case _:
                raise ValueError(f"Unknown sample orientation: {sample_orientation}")

    def _transform_sample_orient_0(self, x_nexus: Union[float, npt.NDArray], y_nexus: Union[float, npt.NDArray]) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        return -y_nexus, x_nexus

    def _transform_sample_orient_1(self, x_nexus: Union[float, npt.NDArray], y_nexus: Union[float, npt.NDArray]) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        return x_nexus, y_nexus

    def _transform_sample_orient_2(self, x_nexus: Union[float, npt.NDArray], y_nexus: Union[float, npt.NDArray]) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        return y_nexus, -x_nexus

    def _inverse_sample_orient_0(self, x_uninclined: Union[float, npt.NDArray], y_uninclined: Union[float, npt.NDArray]) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        return y_uninclined, -x_uninclined

    def _inverse_sample_orient_1(self, x_uninclined: Union[float, npt.NDArray], y_uninclined: Union[float, npt.NDArray]) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        return x_uninclined, y_uninclined

    def _inverse_sample_orient_2(self, x_uninclined: Union[float, npt.NDArray], y_uninclined: Union[float, npt.NDArray]) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        return -y_uninclined, x_uninclined

    def rotate_detector_image(self, hist_nexus: npt.NDArray) -> npt.NDArray:
        """
        Rotates a 2D NeXus detector image matrix (horizontal x vertical) to match the
        (uninclined) BornAgain sample frame.

        Parameters
        ----------
        hist_nexus : ndarray
            2D histogram/image array from the detector in NeXus orientation.

        Returns
        -------
        ndarray
            Rotated image array.
        """
        match self.sample_orientation:
            case 0:
                return np.rot90(hist_nexus, -1)
            case 1:
                return hist_nexus
            case 2:
                return np.rot90(hist_nexus, 1)
            case _:
                raise ValueError(f"Unknown sample orientation: {self.sample_orientation}")

    def apply_inclination_angle_transformation(
        self, x_uninclined: Union[float, npt.NDArray], z_uninclined: Union[float, npt.NDArray]
    ) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Apply sample inclination angle rotation (alpha) to the 2D vertical-longitudinal plane
        (x = forward, z = up).

        Parameters
        ----------
        x_uninclined : float or ndarray
            The longitudinal coordinate(s) in the uninclined frame.
        z_uninclined : float or ndarray
            The vertical coordinate(s) in the uninclined frame.

        Returns
        -------
        Tuple[float or ndarray, float or ndarray]
            The (x, z) coordinate(s) transformed to the BornAgain frame.
        """
        x_arr = np.asarray(x_uninclined)
        z_arr = np.asarray(z_uninclined)

        x_flat = np.ravel(x_arr)
        z_flat = np.ravel(z_arr)

        m00 = self.sample_inclination_rotation_matrix[0, 0]
        m01 = self.sample_inclination_rotation_matrix[0, 1]
        m10 = self.sample_inclination_rotation_matrix[1, 0]
        m11 = self.sample_inclination_rotation_matrix[1, 1]

        x_bornagain = (m00 * x_flat + m01 * z_flat).reshape(x_arr.shape)
        z_bornagain = (m10 * x_flat + m11 * z_flat).reshape(z_arr.shape)

        # Return scalar if input was scalar
        if x_arr.ndim == 0 and z_arr.ndim == 0:
            return float(x_bornagain), float(z_bornagain)
        return x_bornagain, z_bornagain

    def apply_inverse_inclination_angle_transformation(
        self, x_bornagain: Union[float, npt.NDArray], z_bornagain: Union[float, npt.NDArray]
    ) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Apply inverse sample inclination angle rotation (+alpha) to the 2D vertical-longitudinal plane.

        Parameters
        ----------
        x_bornagain : float or ndarray
            The longitudinal coordinate(s) in the BornAgain frame.
        z_bornagain : float or ndarray
            The vertical coordinate(s) in the BornAgain frame.

        Returns
        -------
        Tuple[float or ndarray, float or ndarray]
            The (x, z) coordinate(s) transformed back to the uninclined frame.
        """
        x_arr = np.asarray(x_bornagain)
        z_arr = np.asarray(z_bornagain)

        x_flat = np.ravel(x_arr)
        z_flat = np.ravel(z_arr)

        m00 = self.inverse_sample_inclination_rotation_matrix[0, 0]
        m01 = self.inverse_sample_inclination_rotation_matrix[0, 1]
        m10 = self.inverse_sample_inclination_rotation_matrix[1, 0]
        m11 = self.inverse_sample_inclination_rotation_matrix[1, 1]

        x_uninclined = (m00 * x_flat + m01 * z_flat).reshape(x_arr.shape)
        z_uninclined = (m10 * x_flat + m11 * z_flat).reshape(z_arr.shape)

        if x_arr.ndim == 0 and z_arr.ndim == 0:
            return float(x_uninclined), float(z_uninclined)
        return x_uninclined, z_uninclined

    def nexus_to_bornagain(
        self, x_nexus: Union[float, npt.NDArray], y_nexus: Union[float, npt.NDArray], z_nexus: Union[float, npt.NDArray]
    ) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Transform 3D position or velocity vector from NeXus coordinate system to BornAgain coordinate system.
        
        NeXus: X=horizontal left, Y=vertical up, Z=longitudinal forward.
        BornAgain: X=longitudinal forward, Y=horizontal left, Z=vertical up.
        
        Parameters
        ----------
        x_nexus : float or ndarray
            X-coordinate in NeXus frame.
        y_nexus : float or ndarray
            Y-coordinate in NeXus frame.
        z_nexus : float or ndarray
            Z-coordinate in NeXus frame.
            
        Returns
        -------
        Tuple[float or ndarray, float or ndarray, float or ndarray]
            (x, y, z) coordinates in BornAgain frame.
        """
        x_arr = np.asarray(x_nexus)
        y_arr = np.asarray(y_nexus)
        z_arr = np.asarray(z_nexus)

        x_horiz, y_vert = self.apply_sample_orientation_transform(x_arr, y_arr)

        x_bornagain_uninclined = z_arr
        y_bornagain_uninclined = x_horiz
        z_bornagain_uninclined = y_vert

        x_bornagain, z_bornagain = self.apply_inclination_angle_transformation(x_bornagain_uninclined, z_bornagain_uninclined)
        y_bornagain = y_bornagain_uninclined

        if x_arr.ndim == 0 and y_arr.ndim == 0 and z_arr.ndim == 0:
            return float(x_bornagain), float(y_bornagain), float(z_bornagain)
        return x_bornagain, y_bornagain, z_bornagain

    def bornagain_to_nexus(
        self, x_bornagain: Union[float, npt.NDArray], y_bornagain: Union[float, npt.NDArray], z_bornagain: Union[float, npt.NDArray]
    ) -> Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Transform 3D position or velocity vector from BornAgain coordinate system to NeXus coordinate system.
        
        BornAgain: X=longitudinal forward, Y=horizontal left, Z=vertical up.
        NeXus: X=horizontal left, Y=vertical up, Z=longitudinal forward.
        
        Parameters
        ----------
        x_bornagain : float or ndarray
            X-coordinate in BornAgain frame.
        y_bornagain : float or ndarray
            Y-coordinate in BornAgain frame.
        z_bornagain : float or ndarray
            Z-coordinate in BornAgain frame.
            
        Returns
        -------
        Tuple[float or ndarray, float or ndarray, float or ndarray]
            (x, y, z) coordinates in NeXus frame.
        """
        x_arr = np.asarray(x_bornagain)
        y_arr = np.asarray(y_bornagain)
        z_arr = np.asarray(z_bornagain)

        x_uninclined, z_uninclined = self.apply_inverse_inclination_angle_transformation(x_arr, z_arr)
        y_uninclined = y_arr

        x_horiz = y_uninclined
        y_vert = z_uninclined
        z_nexus = x_uninclined

        x_nexus, y_nexus = self._apply_inverse_sample_orientation_transform(x_horiz, y_vert)

        if x_arr.ndim == 0 and y_arr.ndim == 0 and z_arr.ndim == 0:
            return float(x_nexus), float(y_nexus), float(z_nexus)
        return x_nexus, y_nexus, z_nexus
