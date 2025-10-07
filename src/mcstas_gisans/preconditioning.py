"""
Preconditioning the particles before the BornAgain simulation
"""

import numpy as np
from pathlib import Path

from .instruments import instrumentParameters

def transform_to_sample_system(events, alpha_inc_deg):
  """Apply coordinate transformation to express neutron parameters in a
  coordinate system with the sample in the centre and being horizontal.
  """
  alpha_inc = float(np.deg2rad(alpha_inc_deg))
  rotation_matrix = np.array([[np.cos(-alpha_inc), -np.sin(-alpha_inc)],
                              [np.sin(-alpha_inc), np.cos(-alpha_inc)]])
  p, x, y, z, vx, vy, vz, t = events.T
  zRot, yRot = np.dot(rotation_matrix, [z, y])
  vzRot, vyRot = np.dot(rotation_matrix, [vz, vy])
  return np.vstack([p, x, yRot, zRot, vx, vyRot, vzRot, t]).T

def propagate_to_sample_surface(events, sample_xwidth, sample_zheight, allow_sample_miss):
  """Propagate particles to y=0, the sample surface.
  Discard those which would miss the sample.
  """
  p, x, y, z, vx, vy, vz, t = events.T
  t_propagate = -y/vy # y+vy*t_propagate=0 (where y is the initial position)
  x += vx * t_propagate
  y += vy * t_propagate
  z += vz * t_propagate
  t += t_propagate

  # Create a boolean mask for neutrons to select those which hit the sample
  hit_sample_mask = (abs(x) < sample_xwidth*0.5) & (abs(z) < sample_zheight*0.5)
  events_on_sample_surface = np.vstack([p, x, y, z, vx, vy, vz, t]).T if allow_sample_miss else np.vstack([p, x, y, z, vx, vy, vz, t]).T[hit_sample_mask]

  event_number = len(events)
  sample_hit_event_number = np.sum(hit_sample_mask)
  if sample_hit_event_number != event_number:
    sum_weight_in = sum(p)
    sum_weight_sample_hit = sum(p[hit_sample_mask])
    print(f"    WARNING: {event_number - sample_hit_event_number} out of {event_number} incident neutrons missed the sample!({sum_weight_in-sum_weight_sample_hit} out of {sum_weight_in} in terms of sum particle weight)")
    if not allow_sample_miss:
      print(f"    WARNING: Incident neutrons missing the sample are not propagated to the detectors! This can be changed with the --allow_sample_miss option.") #TODO mention the option to allow them with the input parameter
  return events_on_sample_surface

def apply_t0_correction(events, args):
  """Apply t0 TOF correction for all neutrons. A fixed t0correction value can be
  given to be subtracted, or a McStas TOFLambda monitor result with a selected
  wavelength is used, in which case t0correction is retrieved as the mean value
  from fitting a Gaussian function to the TOF spectrum of the wavelength bin
  including the selected wavelength. The fitting is done for the full TOF range
  unless the WFM mode is used, in which case it is done within the wavelength
  dependent subpulse TOF limits. Rebinning along the wavelength axis can be
  applied beforehand to improve the reliability of the fitting.
  WARNING: the TOF axis of the monitor is assumed to have microsecond units!
  """
  if args.t0_fixed: #T0 correction with fixed input value
    t0_correction = args.t0_fixed
  else: #T0 correction based on McStas (TOFLambda) monitor
    if not args.wfm:
      tof_limits = [None, None] #Do not restrict the monitor TOF spectrum for T0 correction fitting
      t0_monitor = instrumentParameters[args.instrument]['t0_monitor_name']
    else: # Wavelength Frame Multiplication (WFM)
      from .instruments import getSubpulseTofLimits
      tof_limits = getSubpulseTofLimits(args.wavelength)
      t0_monitor = instrumentParameters[args.instrument]['wfm_t0_monitor_name']
    print(f"Applying T0 correction based on McStas monitor: {t0_monitor}")
    mcstas_directory = Path(args.filename).resolve().parent
    from .fit_monitor import fitGaussianToMcstasMonitor
    fit = fitGaussianToMcstasMonitor(mcstas_directory, t0_monitor, args.wavelength, tofLimits=tof_limits, wavelength_rebin=args.t0_wavelength_rebin)
    t0_correction = fit['mean'] * 1e-6
  print(f"T0 correction value: {t0_correction} second")

  p, x, y, z, vx, vy, vz, t = events.T
  t -= t0_correction
  events = np.vstack([p, x, y, z, vx, vy, vz, t]).T
  return events

def precondition(events, args):
  """
  Precondition particles (beam) to bridge the gap between Mcstas and BornAgain
  1) Apply coordinate transformation
  2) Propagate particles to the sample surface
  3) Optionally apply T0 (time-of-flight) correction
  """
  events = transform_to_sample_system(events, args.alpha)
  events = propagate_to_sample_surface(events, args.sample_xwidth, args.sample_zheight, args.allow_sample_miss)
  if args.no_t0_correction or not instrumentParameters[args.instrument]['tof_instrument']:
    print("No T0 correction is applied.")
  else:
    events = apply_t0_correction(events, args)

  return events
