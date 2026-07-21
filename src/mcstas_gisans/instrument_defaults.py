
"""
Instrument parameters used for data reduction
"""

# 't0_monitor_name' is required to enable t0 correction based on McStas TOFLambda monitor
# Wavelength Frame Multiplication requires the following options
#   'wfm_t0_monitor_name'
#   'wfm_virtual_source_distance'

# 'detector' property is not required, but if added, it is expected to have
# all properties listed in the default_detector object below

instrument_defaults = {
  'saga': {
    'nominal_source_sample_distance' : 55, #[m]
    'sample_detector_distance' : 10, #[m] along the y axis
    'beam_declination_angle' : 0.5, #[deg]
    'tof_instrument' : True,
    'mcpl_monitor_name' : 'Mcpl_TOF_Lambda',
    't0_monitor_name' : 'Source_TOF_Lambda',
    'wfm_t0_monitor_name' : 'toflambdawfmc',
    'wfm_virtual_source_distance': 8.2, #real source to virtual source distance for WFM mode
  },
  'loki': {
    'nominal_source_sample_distance' : 23.6,
    'sample_detector_distance' : 10, #can be 5-10m
    'tof_instrument' : True,
    'mcpl_monitor_name' : 'Mcpl_TOF_Lambda',
    't0_monitor_name' : 'Source_TOF_Lambda',
  },
  'skadi': {
    'nominal_source_sample_distance' : 38.43,
    'sample_detector_distance' : 12, #can be 4-20m
    'tof_instrument' : True,
  },
  'd22': { #ILL
    'nominal_source_sample_distance' : 61.28, #approximate value, but it is not really used
    'sample_detector_distance' : 17.6,
    'beam_declination_angle' : 0.0, #[deg]
    'tof_instrument' : False,
    't0_monitor_name' : 'Source_TOF_Lambda',
    'detector': {
      'size': [1.024, 1.024], #[m]
      'direct_beam_centre_offset': [0.0, 0.0], #[m]
      'pixels': [128, 256],
      'resolution': [0.0, 0.004] #fwhm[m]
    },
  }
}

default_detector = {
  'size': [1.024, 1.024], #[m]
  'direct_beam_centre_offset': [0.0, 0.0],
  'pixels': [256, 256],
  'resolution': [0.0, 0.0] #fwhm[m]
}

#required keys in the instrument_defaults to enable WFM(wavelength frame multiplication) mode
required_keys_for_wfm = ['wfm_t0_monitor_name', 'wfm_virtual_source_distance']

# temporary hard-coded sub-pulse tof limits for the SAGA instrument
saga_subpulse_tof_limits = [
  [10200, 12000],
  [12000, 14300],
  [14300, 16100],
  [16100, 18000]
]

def get_saga_subpulse_tof_limits(wavelength):
  """
  Get hard-coded TOF limits of a WFM sub-pulse in between the WFM choppers
  for the SAGA instrument, depending on the wavelength
  """
  if wavelength < 5.15:
    subpulse_id = 0
  elif wavelength < 6.15:
    subpulse_id = 1
  elif wavelength < 7.1:
    subpulse_id = 2
  else:
    subpulse_id = 3

  return saga_subpulse_tof_limits[subpulse_id]

import copy

_initial_instrument_defaults = copy.deepcopy(instrument_defaults)

def reset_instrument_defaults():
  """Reset instrument_defaults dictionary to original initial values."""
  global instrument_defaults
  instrument_defaults.clear()
  for k, v in copy.deepcopy(_initial_instrument_defaults).items():
    instrument_defaults[k] = v

def set_instrument_parameters(args, instrument_name=None):
  """
  Apply command line argument overrides to instrument_defaults in-place.
  Returns the updated instrument parameter dictionary for the target instrument.
  """
  instr_name = getattr(args, 'instrument', instrument_name) if args else instrument_name
  if not instr_name:
    instr_name = 'd22'
    
  if instr_name in instrument_defaults:
    instr_params = instrument_defaults[instr_name]

    if getattr(args, 'instrument_nominal_source_sample_distance', None) is not None:
      instr_params['nominal_source_sample_distance'] = args.instrument_nominal_source_sample_distance

    if getattr(args, 'instrument_sample_detector_distance', None) is not None:
      instr_params['sample_detector_distance'] = args.instrument_sample_detector_distance

    if getattr(args, 'instrument_tof_instrument', None) is not None:
      instr_params['tof_instrument'] = (args.instrument_tof_instrument == 'true')

    if getattr(args, 'instrument_t0_monitor_name', None) is not None:
      instr_params['t0_monitor_name'] = args.instrument_t0_monitor_name

    if getattr(args, 'instrument_wfm_t0_monitor_name', None) is not None:
      instr_params['wfm_t0_monitor_name'] = args.instrument_wfm_t0_monitor_name

    if getattr(args, 'instrument_wfm_virtual_source_distance', None) is not None:
      instr_params['wfm_virtual_source_distance'] = args.instrument_wfm_virtual_source_distance

    if getattr(args, 'instrument_beam_declination_angle', None) is not None:
      instr_params['beam_declination_angle'] = args.instrument_beam_declination_angle

    # Handle detector overrides
    if 'detector' not in instr_params:
      instr_params['detector'] = copy.deepcopy(default_detector)

    det_params = instr_params['detector']

    if getattr(args, 'instrument_detector_size', None) is not None:
      det_params['size'] = list(args.instrument_detector_size)

    if getattr(args, 'instrument_detector_centre_offset', None) is not None:
      det_params['direct_beam_centre_offset'] = list(args.instrument_detector_centre_offset)

    if getattr(args, 'instrument_detector_pixels', None) is not None:
      det_params['pixels'] = list(args.instrument_detector_pixels)

    if getattr(args, 'instrument_detector_resolution', None) is not None:
      det_params['resolution'] = list(args.instrument_detector_resolution)

    return instr_params
  return instrument_defaults.get(instr_name, {})
