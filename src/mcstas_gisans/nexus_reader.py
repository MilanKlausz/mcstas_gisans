
"""
Read measured detector data from NeXus files: the 2D detector image of D22 (ILL) files (hard-coded paths), and the
time-of-flight histograms of the rear detector of raw ISIS SANS2D files.
"""

import h5py
import numpy as np


# Hard-coded HDF5 paths tried (in order) for the detector data when no
# explicit --nxs_data_path is given.
DEFAULT_NEXUS_DATA_PATHS = ('entry0/D22/Detector 1/data1', 'entry0/data1/MultiDetector1_data')

# NXentry field holding the measurement duration in seconds. 'duration' is
# the standard NeXus field name (NXentry base class); 'time' is an
# ILL-specific alias observed to hold the same value in practice.
DURATION_FIELD_CANDIDATES = ('entry0/duration', 'entry0/time', 'raw_data_1/duration')

# Raw ISIS SANS2D NeXus files (the TOF histograms of the tube detectors). The spectra of detector_1/counts
# (periods, spectra, TOF bins) are those of the detector IDs in isis_vms_compat/UDET after the first
# SANS2D_MONITOR_SPECTRA entries (the monitors). Detector ID = d g tt ppp (decimal digits): d = 1 rear / 2 front
# detector, g = board 1-5, tt = tube 0-23 on the board, ppp = pixel 0-511 along the tube (+x). The rear detector
# (Mantid IDF SANS2D_Definition_Tubes.xml) has 120 horizontal tubes at 8.1 mm vertical pitch: the even tt are the
# "left" tubes and the odd tt the "right" ones, interleaved from the bottom, so the tube row (+y) is
# 2 * ((g - 1) * 12 + tt // 2) + tt % 2 (the 3 mm horizontal stagger of the left/right tubes is ignored).
SANS2D_BEAMLINE = 'SANS2D'
SANS2D_MONITOR_SPECTRA = 8
SANS2D_REAR_DETECTOR_PIXELS = (512, 120)
NXS_FORMATS = ('auto', 'image', 'isis_sans2d')

def read_nexus_raw(filepath, data_path=None):
    """
    Read the raw 2D detector image [ix, iy] (first frame) from a NeXus file.

    Parameters
    ----------
    filepath : str
        Path to the NeXus file.
    data_path : str, optional
        Explicit HDF5 path to the detector data inside the NeXus file. If not
        given, DEFAULT_NEXUS_DATA_PATHS are tried in order.
    """
    with h5py.File(filepath, 'r') as file:
        if data_path is not None:
            if data_path not in file:
                raise KeyError(f"Could not find detector data at the given --nxs_data_path '{data_path}' in NeXus file {filepath}")
            detector_data = file[data_path][:]
        else:
            for candidate_path in DEFAULT_NEXUS_DATA_PATHS:
                if candidate_path in file:
                    detector_data = file[candidate_path][:]
                    break
            else:
                raise KeyError(f"Could not find detector data in NeXus file {filepath} at any of the default paths {DEFAULT_NEXUS_DATA_PATHS}. Use --nxs_data_path to specify the correct HDF5 path.")
    data = np.asarray(detector_data, dtype=np.float64)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if data.shape[2] != 1:
            print(f"WARNING: the detector data in {filepath} has {data.shape[2]} frames/time channels; they are summed.")
        return data.sum(axis=2)
    raise ValueError(f"Unsupported detector data shape {data.shape} in {filepath}: a 2D pixel image [x, y] "
                     f"(optionally with a third frame/time-channel axis) is expected.")


def read_nexus_data(filepath, instrument, data_path=None, scale_factor=None):
    """
    Read a measured detector image from a NeXus file, rotated into the BornAgain
    sample frame, with its Poisson errors and Q-space pixel edges.

    Parameters
    ----------
    filepath : str
        Path to the NeXus file.
    instrument : Instrument
        Instrument object used for detector-image rotation and Q-space conversion.
    data_path : str, optional
        Explicit HDF5 path to the detector data inside the NeXus file. If not
        given, DEFAULT_NEXUS_DATA_PATHS are tried in order.
    scale_factor : float, optional
        Multiplicative factor applied to the raw detector counts (and their errors).
    """
    hist = read_nexus_raw(filepath, data_path)
    expected_shape = (instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus)
    if hist.shape != expected_shape:
        raise ValueError(f"The detector image in {filepath} has shape {hist.shape}, but the instrument detector "
                         f"has {expected_shape} pixels (see --instrument_detector_pixels / --nxs_instrument_detector_pixels).")
    hist_error = np.sqrt(hist)
    if scale_factor is not None:
        hist = hist * scale_factor
        hist_error = hist_error * scale_factor

    # Rotate the NeXus detector image to match the BornAgain sample frame
    hist = instrument.detector.coords.rotate_detector_image(hist)
    hist_error = instrument.detector.coords.rotate_detector_image(hist_error)

    q_y, q_z = instrument.get_q_pixel_limits()

    return hist, hist_error, q_y, q_z

def read_nexus_duration(filepath):
    """
    Read the measurement duration in seconds from a NeXus file, if present.

    Checks DURATION_FIELD_CANDIDATES in order and returns the first match.

    Returns
    -------
    float or None
        The measurement duration in seconds, or None if no known duration
        field is present in the file.
    """
    with h5py.File(filepath, 'r') as file:
        for field in DURATION_FIELD_CANDIDATES:
            if field in file:
                return float(np.asarray(file[field][()]).flatten()[0])
    return None

def warn_if_duration_mismatch(nxs_paths, experiment_time, relative_tolerance=0.01, label="NeXus"):
    """
    Weak, non-blocking sanity check: if every given NeXus file reports a
    measurement duration, compare their sum against experiment_time and
    print a warning (without raising) if they disagree by more than
    relative_tolerance. Silently does nothing if experiment_time is None,
    or if any file is missing a duration field.

    Parameters
    ----------
    nxs_paths : list of str
        Paths to the NeXus file(s); their durations are summed before comparing.
    experiment_time : float or None
        The --experiment_time value provided by the user, in seconds.
    relative_tolerance : float, optional
        Fractional difference above which a warning is printed (default: 0.01, i.e. 1%).
    label : str, optional
        Text identifying the file(s) in the warning message.
    """
    if experiment_time is None:
        return
    durations = [read_nexus_duration(p) for p in nxs_paths]
    if any(d is None for d in durations):
        return
    total_duration = sum(durations)
    if total_duration <= 0:
        return
    relative_diff = abs(total_duration - experiment_time) / total_duration
    if relative_diff > relative_tolerance:
        print(
            f"WARNING: {label} report a total measurement duration of {total_duration:.1f}s "
            f"(from the NeXus file's duration/time field), but --experiment_time was set to "
            f"{experiment_time:.1f}s ({relative_diff*100:.1f}% difference). This does not stop "
            f"execution, but double-check --experiment_time matches the actual measurement."
        )


def _first_value(dataset):
    """The first element of an HDF5 dataset (scalar or 1-element array), bytes decoded to str."""
    value = np.asarray(dataset[()]).flatten()[0]
    return value.decode() if isinstance(value, bytes) else value


def is_isis_sans2d_raw(filepath):
    """Whether filepath is a raw ISIS SANS2D NeXus file (raw_data_1/beamline is 'SANS2D')."""
    with h5py.File(filepath, 'r') as file:
        if 'raw_data_1/beamline' not in file:
            return False
        return str(_first_value(file['raw_data_1/beamline'])).strip().upper() == SANS2D_BEAMLINE


def get_nxs_format(filepath, nxs_format='auto'):
    """
    The format of a measured NeXus file (--nxs_format): 'image' (a 2D detector image, e.g. D22) or 'isis_sans2d'
    (raw ISIS SANS2D TOF histograms); 'auto' recognises raw SANS2D files by raw_data_1/beamline.
    """
    if nxs_format not in NXS_FORMATS:
        raise ValueError(f"Unknown NeXus format '{nxs_format}', expected one of {NXS_FORMATS}.")
    if nxs_format != 'auto':
        return nxs_format
    return 'isis_sans2d' if is_isis_sans2d_raw(filepath) else 'image'


def read_sans2d_raw(filepath):
    """
    Read the rear-detector TOF histograms of a raw ISIS SANS2D NeXus file.

    Returns
    -------
    dict
        - ``counts``: int array [ix, iy, TOF bin] of shape (512, 120, number of TOF bins) (summed over the periods),
          ix along the tubes (+x), iy the tube row from the bottom (+y), the layout of the [ix, iy] detector images
          (CoordinateTransform.rotate_detector_image, Detector.get_pixel_positions).
        - ``tof_edges``: TOF bin edges [s] (raw ISIS TOF, from the proton pulse).
        - ``duration``: measurement duration [s] (None if not in the file).
        - ``proton_charge``: [uAh] (None if not in the file).
        - ``rear_detector_distance``: Rear_Det_Z (sample - rear detector distance) [m] (None if not in the file).
        - ``title``: run title (str, '' if not in the file).
    """
    with h5py.File(filepath, 'r') as file:
        if 'raw_data_1/detector_1/counts' not in file or 'raw_data_1/isis_vms_compat/UDET' not in file:
            raise KeyError(f"{filepath} is not a raw ISIS NeXus file with raw_data_1/detector_1/counts and "
                           f"raw_data_1/isis_vms_compat/UDET.")
        raw = file['raw_data_1']
        counts_dataset = raw['detector_1/counts']
        n_spectra = counts_dataset.shape[-2]
        detector_ids = np.asarray(raw['isis_vms_compat/UDET'][SANS2D_MONITOR_SPECTRA:SANS2D_MONITOR_SPECTRA + n_spectra])
        if len(detector_ids) != n_spectra:
            raise ValueError(f"{filepath}: isis_vms_compat/UDET has {len(detector_ids)} detector IDs after the "
                             f"{SANS2D_MONITOR_SPECTRA} monitors, but detector_1/counts has {n_spectra} spectra.")
        rear_spectra = np.flatnonzero(detector_ids // 1000000 == 1)
        # read the (contiguous) range of the rear-detector spectra only
        first, last = int(rear_spectra.min()), int(rear_spectra.max()) + 1
        counts = np.asarray(counts_dataset[..., first:last, :])
        counts = counts.reshape((-1,) + counts.shape[-2:]).sum(axis=0)[rear_spectra - first]
        tof_edges = np.asarray(raw['detector_1/time_of_flight'][()], dtype=np.float64) * 1e-6  # us -> s
        optional = lambda path: float(_first_value(raw[path])) if path in raw else None
        duration = optional('duration')
        proton_charge = optional('proton_charge')
        rear_detector_z = optional('selog/Rear_Det_Z/value_log/value')
        title = str(_first_value(raw['title'])) if 'title' in raw else ''

    if counts.shape[-1] != len(tof_edges) - 1:
        raise ValueError(f"{filepath}: {counts.shape[-1]} TOF bins in detector_1/counts, but "
                         f"{len(tof_edges)} TOF bin edges in detector_1/time_of_flight.")
    rear_ids = detector_ids[rear_spectra]
    board, tube, pixel = (rear_ids // 100000) % 10, (rear_ids // 1000) % 100, rear_ids % 1000
    row = 2 * ((board - 1) * 12 + tube // 2) + tube % 2
    n_x, n_y = SANS2D_REAR_DETECTOR_PIXELS
    if np.any((pixel < 0) | (pixel >= n_x) | (row < 0) | (row >= n_y)):
        raise ValueError(f"{filepath}: rear-detector IDs outside the {n_x} x {n_y} pixel tube detector.")
    image = np.zeros((n_x, n_y, counts.shape[-1]), dtype=counts.dtype)
    np.add.at(image, (pixel, row), counts)
    return {
        'counts': image,
        'tof_edges': tof_edges,
        'duration': duration,
        'proton_charge': proton_charge,
        'rear_detector_distance': rear_detector_z * 1e-3 if rear_detector_z is not None else None,  # mm -> m
        'title': title,
    }
