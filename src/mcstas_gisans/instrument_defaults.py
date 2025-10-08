
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
    'beam_declination_angle' : 0.4, #[deg]
    'tof_instrument' : True,
    'mcpl_monitor_name' : 'Mcpl_TOF_Lambda',
    't0_monitor_name' : 'Source_TOF_Lambda',
    'wfm_t0_monitor_name' : 'toflambdawfmc' ,
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
    'tof_instrument' : False,
    't0_monitor_name' : 'Source_TOF_Lambda',
    'detector': {
      'size': [1.024, 1.024], #[m]
      'centre_offset': [-0.012, 0.290], #[m]
      'pixels': [256, 128],
      'resolution': [0.004, 0.0] #fwhm[m]
    },
  }
}

default_detector = {
  'size': [1.024, 1.024], #[m]
  'centre_offset': [0.0, 0.0],
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
