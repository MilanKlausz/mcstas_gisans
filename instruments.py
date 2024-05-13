
"""
Instrument parameters used for data reduction
"""

instrumentParameters = {
   'saga': {
      'nominal_source_sample_distance' : 55.0, #[m]
      'sample_detector_distance' : 10, #[m] along the y axis
      'tof instrument' : True
   },
   'loki': {
      'nominal_source_sample_distance' : 23.6,
      'sample_detector_distance' : 5, #can be 5-10m
      'tof instrument' : True
   },
   'skadi': {
      'nominal_source_sample_distance' : 38.43,
      'sample_detector_distance' : 12, #can be 4-20m
      'tof instrument' : True
   },
   'd22': { #ILL
      'nominal_source_sample_distance' : 61.28, #approximate value, but it is not really used
      'sample_detector_distance' : 17.6,
      'tof instrument' : False
   }
}
