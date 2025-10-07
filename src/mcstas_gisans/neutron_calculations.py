"""
Neutronic constants and conversion tools
"""

import numpy as np

VS2E = 5.22703725e-6 # Conversion factor from McStas (v[m/s])**2 to E[meV])
V2L = 3956.034012 # Conversion factor velocity->wavelength [m/s·Angstrom] (Planck constant / neutron mass)

def velocityToWavelength(velocity):
  return V2L/velocity

def calculate_wavelength(tof, dist):
  return V2L * tof / dist # Angstrom

def calculate_wavenumber(wavelength):
  return 2*np.pi/(wavelength*0.1) # wavelength Angstrom to nm conversion

def get_velocity_vector(ux, uy, uz, ekin):
  """Convert normalised direction vector and kinetic energy to velocity"""
  norm = np.sqrt(ekin * 1e9 / VS2E)
  return [ux*norm, uy*norm, uz*norm]