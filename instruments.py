
"""
Instrument parameters used for data reduction
"""

# 't0_monitor_name' is required to enable t0 correction based on McStas TOFLambda monitor
# Wavelength Frame Multiplication requires the following options
#   'wfm_t0_monitor_name'
#   'wfm_virtual_source_distance'

instrumentParameters = {
   'saga': {
      'nominal_source_sample_distance' : 55, #[m]
      'sample_detector_distance' : 10, #[m] along the y axis
      'tof instrument' : True,
      't0_monitor_name' : 'Source_TOF_Lambda',
      'wfm_t0_monitor_name' : 'toflambdawfmc' ,
      'wfm_virtual_source_distance': 8.2, #real source to virtual source distance for WFM mode
   },
   'loki': {
      'nominal_source_sample_distance' : 23.6,
      'sample_detector_distance' : 5, #can be 5-10m
      'tof instrument' : True,
      't0_monitor_name' : 'Source_TOF_Lambda',
   },
   'skadi': {
      'nominal_source_sample_distance' : 38.43,
      'sample_detector_distance' : 12, #can be 4-20m
      'tof instrument' : True,
   },
   'd22': { #ILL
      'nominal_source_sample_distance' : 61.28, #approximate value, but it is not really used
      'sample_detector_distance' : 17.6,
      'tof instrument' : False,
      't0_monitor_name' : 'Source_TOF_Lambda',
   },
   'wfm_required_keys': ['wfm_t0_monitor_name', 'wfm_virtual_source_distance']
}
