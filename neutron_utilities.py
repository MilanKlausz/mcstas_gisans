"""
Neutronic constants and conversion tools
"""

import numpy as np

VS2E = 5.22703725e-6 # Conversion factor from McStas (v[m/s])**2 to E[meV])
V2L = 3956.034012 # Conversion factor velocity->wavelength [m/s·Angstrom] (Planck constant / neutron mass)

def velocityToWavelength(velocity):
  return V2L/velocity

def calcWavelength(tof, dist):
  return V2L * tof / dist # Angstrom

def qConvFactor(wavelength):
  return 2*np.pi/(wavelength*0.1) # wavelength Angstrom to nm conversion
