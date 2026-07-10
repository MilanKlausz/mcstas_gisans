"""
Module to calculate the required detector centre_offset for NeXus measurements
so that the beam centre aligns with (qy, qz) = (0, 0) in Q-space.
"""

import sys
import copy
import numpy as np
from scipy.optimize import root

from .read_d22 import read_nexus_data
from .instrument import Instrument
from .instrument_defaults import instrument_defaults

def find_required_centre_offset(filepath, initial_guess=None, alpha_inc_deg=None, wavelength=6.0, sample_orientation=1, instrument_name='d22', verbose=False):
  """
  Find the required centre_offset values for the detector so that the beam centre
  measured in a NeXus file is positioned at (qy, qz) = (0, 0) in Q-space.

  Parameters
  ----------
  filepath : str
      Path to the NeXus file.
  initial_guess : list or np.ndarray, optional
      Initial guess for the centre_offset [x, y] in meters.
      If None, the default centre_offset for the selected instrument is used.
  alpha_inc_deg : float, optional
      Incident angle in degrees. If None, it is automatically determined from the filepath.
  wavelength : float, optional
      Wavelength in Angstroms (default: 6.0).
  sample_orientation : int, optional
      Sample orientation: 0 (vertical, beam from left), 1 (horizontal), 2 (vertical, beam from right). (default: 1).
  instrument_name : str, optional
      The name of the instrument key in instrument_defaults (default: 'd22').
  verbose : bool, optional
      If True, print detailed optimization progress.

  Returns
  -------
  centre_offset : np.ndarray
      The calculated centre_offset [x, y] in meters.
  """
  hist, _, _, _ = read_nexus_data(filepath, sample_orientation=sample_orientation)

  # Is the any reason for alpha_inc_deg!=0 ?
  if alpha_inc_deg is None:
    if filepath.endswith("073174.nxs"):
      alpha_inc_deg = 0.24
    elif filepath.endswith("73378.nxs"):
      alpha_inc_deg = 0.35
    else:
      alpha_inc_deg = 0.0

  if initial_guess is None:
    initial_guess = instrument_defaults.get(instrument_name, {}).get('detector', {}).get('direct_beam_centre_offset', [0.0, 0.0])

  if verbose:
    print(f"\n--- Starting Beam Centre Minimisation ---")
    print(f"Filepath: {filepath}")
    print(f"Initial Guess: {initial_guess}")

  def residual(direct_beam_centre_offset):
    # Copy defaults to avoid modifying global settings in place
    params = copy.deepcopy(instrument_defaults[instrument_name])
    params['detector']['direct_beam_centre_offset'] = list(direct_beam_centre_offset)

    instrument = Instrument(params, alpha_inc_deg, wavelength, sample_orientation=sample_orientation)
    q_y, q_z = instrument.get_q_pixel_limits()

    # Calculate bin centres
    y_centres = (q_y[:-1] + q_y[1:]) / 2.0
    z_centres = (q_z[:-1] + q_z[1:]) / 2.0

    # Calculate weight distributions
    y_intensity = np.sum(hist, axis=1)
    z_intensity = np.sum(hist, axis=0)
    total_intensity = np.sum(hist)

    if total_intensity <= 0:
      raise ValueError("Total intensity of the NeXus dataset is zero or negative.")

    y_centre = np.sum(y_centres * y_intensity) / total_intensity
    z_centre = np.sum(z_centres * z_intensity) / total_intensity

    if verbose:
      print(f"  Eval offset: [{direct_beam_centre_offset[0]:.6f}, {direct_beam_centre_offset[1]:.6f}] -> Q-centre: ({y_centre:.6f}, {z_centre:.6f})")
    return np.array([y_centre, z_centre])

  res = root(residual, initial_guess)
  print(f"Optimization Success: {res.success}")
  print(f"Optimization Message: {res.message}")

  if not res.success:
    raise RuntimeError(f"Optimization failed to find required direct_beam_centre_offset: {res.message}")

  return res.x


def main():
  import argparse
  parser = argparse.ArgumentParser(description="Find required detector centre_offset for a given NeXus data file.")
  parser.add_argument('filepath', type=str, help="Path to the NeXus data file.")
  parser.add_argument('--alpha', type=float, default=None, help="Incident angle in degrees (default: auto-detected).")
  parser.add_argument('--wavelength', type=float, default=6.0, help="Wavelength in Angstroms (default: 6.0).")
  parser.add_argument('--sample_orientation', type=int, default=1, help="Sample orientation (default: 1).")
  parser.add_argument('--instrument', type=str, default='d22', help="Instrument name in instrument_defaults (default: 'd22').")
  parser.add_argument('--verbose', action='store_true', help="Print detailed optimization progress.")
  
  args = parser.parse_args()

  try:
    offset = find_required_centre_offset(
        args.filepath,
        alpha_inc_deg=args.alpha,
        wavelength=args.wavelength,
        sample_orientation=args.sample_orientation,
        instrument_name=args.instrument,
        verbose=args.verbose
    )
    print(f"Calculated centre_offset [m]:  [{offset[0]:.6f}, {offset[1]:.6f}]")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)


if __name__ == '__main__':
  main()
