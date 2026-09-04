
"""
Read data from measurements at D22(ILL) from nxs files (hard-coded)
"""

import h5py
import numpy as np

from .instrument import Instrument
from .instrument_defaults import instrument_defaults

# Hard-coded HDF5 paths tried (in order) for the detector data when no
# explicit --nxs_data_path is given.
DEFAULT_NEXUS_DATA_PATHS = ('entry0/D22/Detector 1/data1', 'entry0/data1/MultiDetector1_data')

# NXentry field holding the measurement duration in seconds. 'duration' is
# the standard NeXus field name (NXentry base class); 'time' is an
# ILL-specific alias observed to hold the same value in practice.
DURATION_FIELD_CANDIDATES = ('entry0/duration', 'entry0/time')

def read_nexus_data(filepath, instrument, data_path=None, scale_factor=None):
    """
    Read data from measurements at D22(ILL) from nxs files.

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
        Multiplicative factor applied to the raw detector counts.
    """

    # Open the NeXus file
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
    hist = detector_data[:,:,0]
    if scale_factor is not None:
        hist = hist * scale_factor
    hist_error = np.sqrt(hist)

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
