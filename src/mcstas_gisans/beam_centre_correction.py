"""
Find the detector centre offset from a measured direct-beam NeXus file.

The offset is the position of the detector centre relative to the undeflected nominal
beam axis through the sample (NeXus frame, metres). It is a property of the detector
position only, so it does not depend on the sample orientation. It is computed directly
in real space: the unscattered beam lands at the point where the undeflected beam axis
(tilted by the beam angle) hits the detector plane, lowered by the gravity drop at the
given wavelength; the offset places the measured intensity centroid on that point.
Together with the Q convention of Instrument.calculate_q_from_nexus_positions this puts
the measured direct beam exactly at Q = 0.
"""

import sys
import copy
import numpy as np

from .nexus_reader import read_nexus_raw
from .instrument import Instrument
from .instrument_defaults import instrument_defaults, default_detector


def find_required_centre_offset(filepath, beam_angle=None, wavelength=6.0, sample_orientation=1, instrument_name='d22', verbose=False, nxs_data_path=None):
    """
    Detector centre offset [x, y] (m, NeXus frame) that reproduces a measured direct beam.

    Parameters
    ----------
    filepath : str
        Path to the NeXus file of the direct-beam measurement.
    beam_angle : float, optional
        Beam angle in degrees (angle of the incident beam above the nominal axis in the
        plane of incidence, towards the sample surface normal). Default: the instrument's
        configured 'beam_angle', or 0.0.
    wavelength : float, optional
        Wavelength of the direct-beam measurement in Angstrom (default: 6.0). Only enters
        through the gravity drop.
    sample_orientation : int, optional
        Sample orientation (0, 1, 2). Only defines the plane in which the beam angle acts.
    instrument_name : str, optional
        The name of the instrument key in instrument_defaults (default: 'd22').
    verbose : bool, optional
        Print the intermediate quantities.
    nxs_data_path : str, optional
        Explicit HDF5 path to the detector data inside the NeXus file.

    Returns
    -------
    centre_offset : np.ndarray
        The detector centre offset [x, y] in metres.
    """
    if instrument_name not in instrument_defaults:
        raise ValueError(f"Unknown instrument '{instrument_name}'. Available: {list(instrument_defaults)}")
    params = copy.deepcopy(instrument_defaults[instrument_name])
    params.setdefault('detector', copy.deepcopy(default_detector))
    params['detector']['direct_beam_centre_offset'] = [0.0, 0.0]
    if beam_angle is not None:
        params['beam_angle'] = beam_angle
    instrument = Instrument(params, 0.0, wavelength, sample_orientation)
    det = instrument.detector

    raw = read_nexus_raw(filepath, nxs_data_path)
    if raw.shape != (det.pixels_x_nexus, det.pixels_y_nexus):
        raise ValueError(f"Detector image shape {raw.shape} does not match the '{instrument_name}' detector "
                         f"({det.pixels_x_nexus}, {det.pixels_y_nexus}).")
    total = raw.sum()
    if total <= 0:
        raise ValueError(f"No counts in the detector image of {filepath}.")

    x_rel = (np.arange(det.pixels_x_nexus) + 0.5) * det.pixel_size_x_nexus - 0.5 * det.size_x_nexus
    y_rel = (np.arange(det.pixels_y_nexus) + 0.5) * det.pixel_size_y_nexus - 0.5 * det.size_y_nexus
    centroid = np.array([(raw.sum(axis=1) * x_rel).sum(), (raw.sum(axis=0) * y_rel).sum()]) / total

    landing = instrument.direct_beam_landing_point_nexus(wavelength)
    offset = landing[:2] - centroid

    if verbose:
        print(f"Direct-beam centroid relative to the detector centre [m]: [{centroid[0]:.6f}, {centroid[1]:.6f}]")
        print(f"Predicted landing point of the unscattered beam [m]:      [{landing[0]:.6f}, {landing[1]:.6f}]"
              f" (beam angle {instrument.beam_angle} deg, wavelength {wavelength} Å)")
    return offset


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Find required detector centre_offset for a given NeXus data file.")
    parser.add_argument('filepath', type=str, help="Path to the NeXus data file.")
    parser.add_argument('--wavelength', type=float, default=6.0, help="Wavelength of the direct-beam measurement in Angstrom; enters through the gravity drop (default: 6.0).")
    parser.add_argument('--sample_orientation', type=int, default=1, choices=[0, 1, 2], help="Sample orientation (0, 1, 2); defines the plane in which the beam angle acts (default: 1).")
    parser.add_argument('--instrument', type=str.lower, default='d22', choices=list(instrument_defaults.keys()), help="Instrument name in instrument_defaults (default: 'd22').")
    parser.add_argument('--beam_angle', type=float, default=None, help="Beam angle in degrees: angle of the incident beam above the nominal beam axis, in the plane of incidence, positive towards the sample surface normal (default: the instrument's configured value, or 0). Note: opposite sign to the former --beam_declination.")
    parser.add_argument('--beam_declination', type=float, default=None, help=argparse.SUPPRESS)  # removed, see main
    parser.add_argument('--nxs_data_path', type=str, default=None, help='Explicit HDF5 path to the detector data inside the NeXus file, e.g. "entry0/data1/MultiDetector1_data". Overrides the default paths that are otherwise tried automatically.')
    parser.add_argument('--verbose', action='store_true', help="Print the direct-beam centroid and the predicted landing point.")
    return parser

def main():
    parser = create_argparser()
    args = parser.parse_args()
    if args.beam_declination is not None:
        parser.error("--beam_declination was renamed to --beam_angle, with the OPPOSITE sign "
                     f"(positive = beam rising towards the sample normal): use --beam_angle {-args.beam_declination}")

    try:
        offset = find_required_centre_offset(
            args.filepath,
            beam_angle=args.beam_angle,
            wavelength=args.wavelength,
            sample_orientation=args.sample_orientation,
            instrument_name=args.instrument,
            verbose=args.verbose,
            nxs_data_path=args.nxs_data_path
        )
        print(f"Calculated centre_offset [m]:  [{offset[0]:.6f}, {offset[1]:.6f}]")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
