"""
Neutronic constants and conversion tools
"""

import numpy as np

NEUTRON_PDG_CODE = 2112
PHOTON_PDG_CODE = 22

VS2E = 5.22703725e-6 # Conversion factor from McStas (v[m/s])**2 to E[meV])
V2L = 3956.034012 # Conversion factor velocity->wavelength [m/s·Angstrom] (Planck constant / neutron mass)

VPHOTON = 299792458 # speed of light
MEV2AA = 0.0123984198433 #[AA*MeV] (6.62607015e-34[J*s] * 299792458[m/s] / 1.602176634e-19[J/eV] / 1e6[MeV/eV] * 1e10[AA/m])

def calculate_neutron_wavelength(tof, dist):
  return V2L * tof / dist # Angstrom

def calculate_wavenumber(wavelength):
  return 2*np.pi / (wavelength*0.1) # wavelength Angstrom to nm conversion

def get_neutron_velocity_vector(ux, uy, uz, ekin):
  """Convert normalised direction vector and kinetic energy to velocity"""
  norm = np.sqrt(ekin * 1e9 / VS2E)
  return [ux * norm, uy * norm, uz * norm]

def neutron_velocity_to_wavelength(velocity_vector):
  """Convert neutron velocity vector to wavelength"""
  return V2L / np.linalg.norm(velocity_vector)

def photon_direction_to_velocity(ux, uy, uz):
  """Get photon velocity vector from direction vector"""
  return [ux*VPHOTON, uy*VPHOTON, uz*VPHOTON]

def photon_energy_to_wavelength(ekin):
  """Convert photon energy to wavelength"""
  return MEV2AA / ekin

def convert_neutron_properties(particle, intensity_factor):
  """Convert neutron properties from MCPL units
  weight: scale by intensity_factor
  position: cm -> m
  direction vector -> velocity vector
  ekin: convert to wavelength
  time: ms -> s
  """
  velocity_vector = get_neutron_velocity_vector(particle.ux, particle.uy, particle.uz, particle.ekin)
  
  return (particle.weight * intensity_factor,
          particle.x/100, particle.y/100, particle.z/100, #convert cm->m
          *velocity_vector,
          neutron_velocity_to_wavelength(velocity_vector),
          particle.time*1e-3 #convert ms->s
          )
  
def convert_photon_properties(particle, intensity_factor):
  """Convert photon properties from MCPL units
  weight: scale by intensity_factor
  position: cm -> m
  direction vector -> velocity vector
  ekin: convert to wavelength
  time: ms -> s
  """
  return (particle.weight * intensity_factor,
          particle.x/100, particle.y/100, particle.z/100, #convert cm->m
          *photon_direction_to_velocity(particle.ux, particle.uy, particle.uz),
          photon_energy_to_wavelength(particle.ekin),
          particle.time*1e-3 #convert ms->s
          )

def get_particle_converter(particle_pdg_code):
  """
  Decide if particle is neutron or photon and return the correct particle
  property converter and particle type
  """
  if particle_pdg_code == NEUTRON_PDG_CODE:
    particle_type = 'neutron'
    convert_particle_properties = convert_neutron_properties
  elif particle_pdg_code == PHOTON_PDG_CODE:
    particle_type = 'photon'
    convert_particle_properties = convert_photon_properties
  else:
    import sys
    sys.exit(f"Unexpeted particle type. PDG code: {particle_pdg_code}")
  return convert_particle_properties, particle_type
