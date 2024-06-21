"""
Neutronic constants and conversion tools
"""

VS2E = 5.22703725e-6 # Conversion factor from McStas (v[m/s])**2 to E[meV])
V2L = 3956.034012 # Conversion factor velocity->wavelength [m/s·Angstrom]

def tofToLambda(tof, dist):
  return V2L/(dist/tof) #angstrom