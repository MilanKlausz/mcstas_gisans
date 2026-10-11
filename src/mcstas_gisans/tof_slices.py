"""
Wavelength slices of time-of-flight (TOF) data for mg_fit (--wavelength_slice): the measured and the simulated
counts of every detector pixel and TOF bin are converted to (Qy, Qz) in the same way and histogrammed into a fixed
Q grid, so that the existing comparisons (losses, masks, background fit, plots) work on the Q histograms.

The wavelength of a TOF bin of a pixel is lambda = h t / (m L) at the centre t of the bin, with the flight path
L = L1 + L2_pixel: L1 the nominal source-sample distance of the instrument (the McStas path length for the simulated
events; the same L1 is used for the measurement, so that both sides get the identical conversion) and L2_pixel the
distance from the sample to the pixel centre. A TOF bin of a pixel belongs to the slice [lambda_min, lambda_max) if its
centre wavelength does; its counts are put into the Q bin of Q(pixel centre, its centre wavelength) (with the
instrument geometry: incident angle, sample orientation, detector offset and gravity, as for monochromatic data).
The simulated events are histogrammed into the same TOF bins of the same pixels first (run.process_particles with
params['tof_bin_edges']), so the simulation goes through exactly the same conversion as the measurement.

Regions that follow the specular reflection (mg_fit --mask_exit_angle_band, --band_exit_angle) are selections of
detector pixels: a slice keeps the (pixel, TOF bin) of the pixels whose exit angle at the centre wavelength of the slice is
in the band (WavelengthSlice.pixel_selection), so that the specular Qz = 2 k sin(alpha), which moves with the wavelength,
is followed from slice to slice, and every TOF bin of a slice uses the same detector rows.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .particle_calculations import V2L

# Relative margin of the TOF limits of the MCPL input (mcpl_tof_limits): the MCPL file is filtered with the TOF range
# of the slices widened by this fraction on both sides, for the moderator emission time and an MCPL plane upstream of
# the sample (the exact selection of the neutrons that can reach a slice follows after the preconditioning,
# particles_reaching_slices).
MCPL_TOF_RELATIVE_MARGIN = 0.1
# Relative margin of the wavelength range that a preconditioned neutron can reach in particles_reaching_slices (for
# the curved path under gravity and the scattering geometry, which the estimate neglects)
REACH_RELATIVE_MARGIN = 0.005


def tof_to_wavelength(tof, path_length):
    """Wavelength [Angstrom] of a neutron with time of flight tof [s] over path_length [m]."""
    return V2L * np.asarray(tof, dtype=np.float64) / path_length


def wavelength_to_tof(wavelength, path_length):
    """Time of flight [s] of a neutron of wavelength [Angstrom] over path_length [m]."""
    return np.asarray(wavelength, dtype=np.float64) * path_length / V2L


@dataclass
class WavelengthSlice:
    """
    One wavelength slice: the mapping of the (pixel, TOF bin) of the common TOF bin range of all slices
    (build_wavelength_slices) to the bins of its Q grid, and, once filled by mg_fit, the measured Q histogram.

    q_index[p, j] is the flat index (iy * number of Qz bins + iz) of the Q bin of pixel p (flat detector ID
    ix * pixels_y + iy of the [ix, iy] detector image) and the TOF bin first_bin + j, or -1 if that TOF bin is not
    in the slice for the pixel or its Q is outside the grid.

    pixel_selection (boolean per pixel, None: all pixels) restricts the slice to the (pixel, TOF bin) of these pixels
    (--mask_exit_angle_band): to_q, background_unit, coverage and used_pixels use only them.
    """
    lambda_min: float
    lambda_max: float
    y_edges: np.ndarray
    z_edges: np.ndarray
    q_index: np.ndarray
    background_unit: np.ndarray
    reach_range: Tuple[float, float]
    n_tof_bins: int
    detector_shape: Tuple[int, int]
    pixel_selection: Optional[np.ndarray] = None
    # the measured data (prepare_experimental_data): masked (NaN outside the mask and outside the coverage) and raw
    # (every pixel, without masks)
    hist_nxs: Optional[np.ndarray] = None
    hist_nxs_error: Optional[np.ndarray] = None
    hist_nxs_raw: Optional[np.ndarray] = None
    hist_nxs_error_raw: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    # the 1D band of the exit angle (--band_exit_angle, set by mg_fit): its pixels, the measured Q histogram of their
    # (pixel, TOF bin) (NaN outside the mask and outside their coverage) and its background unit
    profile_pixels: Optional[np.ndarray] = None
    profile_hist_nxs: Optional[np.ndarray] = None
    profile_unit: Optional[np.ndarray] = None
    _entries: np.ndarray = field(default=None, repr=False)
    _targets: np.ndarray = field(default=None, repr=False)
    _selected: Optional[np.ndarray] = field(default=None, repr=False)
    _unit: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        flat = self.q_index.ravel()
        self._entries = np.flatnonzero(flat >= 0)
        self._targets = flat[self._entries]
        if self.pixel_selection is not None:
            self.pixel_selection = np.asarray(self.pixel_selection, dtype=bool).ravel()
        self._selected = self._entry_selection(self.pixel_selection)

    @property
    def label(self) -> str:
        return f"{self.lambda_min:g}-{self.lambda_max:g} AA"

    @property
    def q_shape(self) -> Tuple[int, int]:
        return len(self.y_edges) - 1, len(self.z_edges) - 1

    @property
    def n_pixels(self) -> int:
        return self.q_index.shape[0]

    @property
    def coverage(self) -> np.ndarray:
        """The Q bins that receive counts of at least one (pixel, TOF bin) of the slice (of its pixel selection)."""
        return self.background_unit > 0

    def _entry_selection(self, pixels: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Boolean over the mapped (pixel, TOF bin): those of the given pixels (boolean per pixel or [ix, iy] image;
        None: all)."""
        if pixels is None:
            return None
        pixels = np.asarray(pixels, dtype=bool).ravel()
        if pixels.size != self.n_pixels:
            raise ValueError(f"The pixel selection has {pixels.size} pixels, the slice {self.n_pixels}.")
        return pixels[self._entries // self.q_index.shape[1]]

    def _histogram(self, values: np.ndarray, keep: Optional[np.ndarray]) -> np.ndarray:
        """Q histogram of values given for the mapped (pixel, TOF bin) (self._entries), of those of keep (None: all)."""
        targets = self._targets
        if keep is not None:
            values, targets = values[keep], targets[keep]
        n_y, n_z = self.q_shape
        return np.bincount(targets, weights=values, minlength=n_y * n_z).reshape(n_y, n_z)

    def to_q(self, data: np.ndarray, pixels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Q histogram (Qy bins, Qz bins) of data per (pixel, TOF bin of the common range), shape q_index.shape (or an
        [ix, iy, TOF bin] array of that many elements): the sum of the data of the (pixel, TOF bin) of every Q bin, of
        the pixels of the slice's pixel_selection (all pixels without one), or of `pixels` (boolean per pixel) if given.
        """
        flat = np.asarray(data, dtype=np.float64).reshape(-1)
        keep = self._selected if pixels is None else self._entry_selection(pixels)
        entries = self._entries if keep is None else self._entries[keep]
        targets = self._targets if keep is None else self._targets[keep]
        n_y, n_z = self.q_shape
        return np.bincount(targets, weights=flat[entries], minlength=n_y * n_z).reshape(n_y, n_z)

    def set_background_unit(self, unit: np.ndarray) -> None:
        """Set the background of one count per pixel over the slice, per (pixel, TOF bin) (build_wavelength_slices), and
        background_unit, its Q histogram over the pixel selection."""
        self._unit = np.asarray(unit, dtype=np.float64).reshape(-1)[self._entries]
        self.background_unit = self._histogram(self._unit, self._selected)

    def background_unit_of(self, pixels: np.ndarray) -> np.ndarray:
        """The Q histogram of one count per pixel over the slice, of the given pixels (boolean per pixel) only."""
        return self._histogram(self._unit, self._entry_selection(pixels))

    def used_pixels(self, q_keep: np.ndarray) -> np.ndarray:
        """Boolean [ix, iy] detector image: the pixels (of the pixel selection) with a TOF bin of the slice in a kept
        (True) Q bin."""
        n_bins = self.q_index.shape[1]
        keep = np.asarray(q_keep, dtype=bool).ravel()[self._targets]
        if self._selected is not None:
            keep &= self._selected
        pixels = np.zeros(self.n_pixels, dtype=bool)
        pixels[self._entries[keep] // n_bins] = True
        return pixels.reshape(self.detector_shape)


def pixel_path_lengths(instrument, source_sample_distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-centre positions (NeXus frame, relative to the sample, flat detector-ID order) and the flight paths
    L1 + L2_pixel [m] of their neutrons."""
    positions = instrument.detector.get_pixel_positions(instrument.sample_detector_distance)
    return positions, source_sample_distance + np.linalg.norm(positions, axis=1)


def common_tof_bin_range(tof_edges: np.ndarray, path_lengths: np.ndarray,
                         wavelength_slices: Sequence[Sequence[float]]) -> Tuple[int, int]:
    """
    The range [first, last) of the TOF bins whose centre wavelength is in a slice for at least one pixel: the
    (pixel, TOF bin) arrays of the simulation and of the slices cover this range.
    """
    tof_centres = 0.5 * (tof_edges[1:] + tof_edges[:-1])
    shortest, longest = tof_to_wavelength(tof_centres, path_lengths.max()), tof_to_wavelength(tof_centres, path_lengths.min())
    candidate = np.zeros(len(tof_centres), dtype=bool)
    for lambda_min, lambda_max in wavelength_slices:
        candidate |= (longest >= lambda_min) & (shortest < lambda_max)
    if not np.any(candidate):
        ranges = ', '.join(f"{a:g}-{b:g}" for a, b in wavelength_slices)
        raise ValueError(f"No TOF bin of the measurement is in the wavelength slice(s) {ranges} AA (the TOF bins cover "
                         f"{tof_to_wavelength(tof_edges[0], path_lengths.max()):.3g}-"
                         f"{tof_to_wavelength(tof_edges[-1], path_lengths.min()):.3g} AA).")
    bins = np.flatnonzero(candidate)
    return int(bins[0]), int(bins[-1]) + 1


def tof_bin_wavelengths(instrument, tof_edges: np.ndarray, source_sample_distance: float) -> np.ndarray:
    """The wavelength [Angstrom] of every (pixel, TOF bin): h t / (m (L1 + L2_pixel)) at the bin centre t, shape
    (number of pixels, number of TOF bins) for the TOF bin edges tof_edges [s]."""
    tof_edges = np.asarray(tof_edges, dtype=np.float64)
    _, path_lengths = pixel_path_lengths(instrument, source_sample_distance)
    return tof_to_wavelength(0.5 * (tof_edges[1:] + tof_edges[:-1])[np.newaxis, :], path_lengths[:, np.newaxis])


def build_wavelength_slices(instrument, tof_edges: np.ndarray, wavelength_slices: Sequence[Sequence[float]],
                            source_sample_distance: float, exit_angle_band: Optional[Sequence[float]] = None
                            ) -> Tuple[List[WavelengthSlice], int, int]:
    """
    The WavelengthSlice of every [lambda_min, lambda_max] of wavelength_slices, for the TOF bin edges tof_edges [s]
    of the measurement, and the common TOF bin range [first, last) that their (pixel, TOF bin) arrays cover.

    The Q grid of a slice has the pixel edges of the detector at the centre wavelength of the slice
    (Instrument.get_q_pixel_limits), so the Q masks and the plots work as for monochromatic data. Its background_unit
    is the Q histogram of one count per pixel over the slice (spread over the TOF bins of the slice of the pixel in
    proportion to their widths, i.e. a background constant in time): a flat background of b counts per pixel adds
    b * background_unit to the simulated Q histogram.

    With exit_angle_band (LOW, HIGH) [deg] (--mask_exit_angle_band), a slice keeps only the pixels whose exit angle
    alpha_f at the centre wavelength of the slice has LOW <= alpha_f - alpha <= HIGH (masking.exit_angle_band_pixels;
    its pixel_selection).
    """
    from .masking import exit_angle_band_pixels
    tof_edges = np.asarray(tof_edges, dtype=np.float64)
    if np.any(np.diff(tof_edges) <= 0):
        raise ValueError("The TOF bin edges must be increasing.")
    positions, path_lengths = pixel_path_lengths(instrument, source_sample_distance)
    first, last = common_tof_bin_range(tof_edges, path_lengths, wavelength_slices)
    edges = tof_edges[first:last + 1]
    widths = np.diff(edges)
    centre_wavelengths = tof_bin_wavelengths(instrument, edges, source_sample_distance)
    detector_shape = (instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus)

    slices = []
    for lambda_min, lambda_max in wavelength_slices:
        centre = 0.5 * (lambda_min + lambda_max)
        y_edges, z_edges = instrument.get_q_pixel_limits(wavelength=centre)
        if np.any(np.diff(y_edges) <= 0) or np.any(np.diff(z_edges) <= 0):
            raise ValueError("The Q edges of the detector pixels are expected to be increasing.")
        n_y, n_z = len(y_edges) - 1, len(z_edges) - 1
        in_slice = (centre_wavelengths >= lambda_min) & (centre_wavelengths < lambda_max)
        q_index = np.full(in_slice.shape, -1, dtype=np.int64)
        for j in range(in_slice.shape[1]):
            pixels = np.flatnonzero(in_slice[:, j])
            if len(pixels) == 0:
                continue
            q = instrument.calculate_q_from_nexus_positions(positions[pixels], centre_wavelengths[pixels, j])
            iy = np.searchsorted(y_edges, q[:, 1], side='right') - 1
            iz = np.searchsorted(z_edges, q[:, 2], side='right') - 1
            inside = (iy >= 0) & (iy < n_y) & (iz >= 0) & (iz < n_z)
            q_index[pixels[inside], j] = iy[inside] * n_z + iz[inside]

        # one count per pixel over the slice, in proportion to the TOF bin widths
        slice_width = np.sum(np.where(in_slice, widths, 0.0), axis=1, keepdims=True)
        unit = np.divide(np.where(in_slice, widths, 0.0), slice_width, out=np.zeros(in_slice.shape), where=slice_width > 0)

        # the wavelength range of the events that can land in the TOF bins of the slice (their edges)
        lower = tof_to_wavelength(edges[:-1][np.newaxis, :], path_lengths[:, np.newaxis])
        upper = tof_to_wavelength(edges[1:][np.newaxis, :], path_lengths[:, np.newaxis])
        if not np.any(in_slice):
            raise ValueError(f"No TOF bin of the measurement has its centre in the wavelength slice {lambda_min:g}-{lambda_max:g} AA.")
        reach_range = (float(lower[in_slice].min()), float(upper[in_slice].max()))

        pixel_selection = None if exit_angle_band is None else exit_angle_band_pixels(instrument, centre, exit_angle_band)
        wavelength_slice = WavelengthSlice(
            lambda_min=float(lambda_min), lambda_max=float(lambda_max), y_edges=y_edges, z_edges=z_edges,
            q_index=q_index, background_unit=np.zeros((n_y, n_z)), reach_range=reach_range,
            n_tof_bins=int(np.count_nonzero(np.any(in_slice, axis=0))), detector_shape=detector_shape,
            pixel_selection=pixel_selection)
        wavelength_slice.set_background_unit(unit)
        slices.append(wavelength_slice)
    return slices, first, last


def mcpl_tof_limits(wavelength_slices: Sequence[Sequence[float]], source_sample_distance: float,
                    relative_margin: float = MCPL_TOF_RELATIVE_MARGIN) -> List[float]:
    """
    TOF limits [millisecond] for filtering the MCPL input (tof_filtering.get_tof_filtering_limits) for the wavelength
    slices: the TOF of the slices over the source-sample distance, widened by relative_margin on both sides.
    """
    lambda_min = min(s[0] for s in wavelength_slices) * (1.0 - relative_margin)
    lambda_max = max(s[1] for s in wavelength_slices) * (1.0 + relative_margin)
    return [float(wavelength_to_tof(lambda_min, source_sample_distance)) * 1e3,
            float(wavelength_to_tof(lambda_max, source_sample_distance)) * 1e3]


def particles_reaching_slices(particles: np.ndarray, instrument, slices: Sequence[WavelengthSlice],
                              source_sample_distance: float, relative_margin: float = REACH_RELATIVE_MARGIN) -> np.ndarray:
    """
    Boolean array: which of the preconditioned particles (BornAgain frame, columns p, x, y, z, vx, vy, vz, w, t, ...)
    can be detected in a TOF bin of one of the slices. A neutron at x (along the beam; 0 on the sample surface) with
    time t and speed v reaches a pixel at distance d from the sample after the path d - x, so it is detected at the
    wavelength lambda = V2L (t + (d - x) / v) / (L1 + d); d ranges from the sample-detector distance to the most
    distant pixel centre. The neutron is kept if this range, widened by relative_margin, overlaps the reach_range of a
    slice (the wavelengths of the edges of its TOF bins).
    """
    particles = np.atleast_2d(np.asarray(particles, dtype=np.float64))
    if len(particles) == 0:
        return np.zeros(0, dtype=bool)
    x, vx, vy, vz, t = particles[:, 1], particles[:, 4], particles[:, 5], particles[:, 6], particles[:, 8]
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    _, path_lengths = pixel_path_lengths(instrument, source_sample_distance)
    distances = (instrument.sample_detector_distance, float(path_lengths.max()) - source_sample_distance)
    wavelengths = np.array([V2L * (t + (d - x) / v) / (source_sample_distance + d) for d in distances])
    low = wavelengths.min(axis=0) * (1.0 - relative_margin)
    high = wavelengths.max(axis=0) * (1.0 + relative_margin)
    keep = np.zeros(len(particles), dtype=bool)
    for wavelength_slice in slices:
        reach_min, reach_max = wavelength_slice.reach_range
        keep |= (high >= reach_min) & (low <= reach_max)
    return keep


def read_intensity_factor_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    A wavelength-dependent intensity calibration (mg_fit --intensity_factor_spectrum): a text file with the columns
    wavelength [Angstrom] and factor (comments start with '#'). Returns (wavelengths, factors); the wavelengths must
    increase and the factors be positive.
    """
    try:
        table = np.loadtxt(path, comments='#', ndmin=2)
    except ValueError as err:
        raise ValueError(f"{path}: expected two columns, wavelength [Angstrom] and factor ({err}).")
    if table.shape[1] != 2 or len(table) < 2:
        raise ValueError(f"{path}: expected two columns (wavelength [Angstrom], factor) and at least two rows, got the shape {table.shape}.")
    wavelengths, factors = table[:, 0], table[:, 1]
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{path}: the values must be finite.")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError(f"{path}: the wavelengths must increase.")
    if np.any(factors <= 0):
        raise ValueError(f"{path}: the factors must be positive.")
    return wavelengths, factors


def intensity_factors_of_tof_bins(spectrum: Tuple[np.ndarray, np.ndarray], bin_wavelengths: np.ndarray,
                                  wavelength_slices: Sequence[Sequence[float]]) -> np.ndarray:
    """
    The factor of an intensity calibration spectrum (read_intensity_factor_spectrum) for every (pixel, TOF bin), linearly
    interpolated at its wavelength (bin_wavelengths, tof_bin_wavelengths). The spectrum must cover the wavelength slices
    (the TOF bins outside the slices are not compared and get the factor of the nearest end).
    """
    wavelengths, factors = spectrum
    low, high = min(s[0] for s in wavelength_slices), max(s[1] for s in wavelength_slices)
    if low < wavelengths[0] or high > wavelengths[-1]:
        raise ValueError(f"The intensity factor spectrum covers {wavelengths[0]:g}-{wavelengths[-1]:g} AA, but the wavelength "
                         f"slices {low:g}-{high:g} AA.")
    return np.interp(bin_wavelengths, wavelengths, factors)
