"""
Read particles from MCPL files (or .dat file)
Output Q histogram files (or raw Q list files)
Output particles into an MCPL file using np2mcpl
"""

import sys
import numpy as np
from neutron_utilities import VS2E
import mcpl

def getVelocity(ux, uy, uz, ekin):
  """Convert normalised direction vector and kinetic energy to velocity"""
  norm = np.sqrt(ekin * 1e9 / VS2E)
  return [ux*norm, uy*norm, uz*norm]

def printTofLimits(mcplTofLimits):
  """Small utility function to print out TOF limits used for filtering"""
  if mcplTofLimits == [float('-inf'), float('inf')]:
    print(f"No TOF filtering applied. Processing all neutrons from the input file.")
  else:
    print(f"Using MCPL input TOF limits: : {mcplTofLimits[0]:.3f} - {mcplTofLimits[1]:.3f} [millisecond]")

def getNeutronEvents(filename, mcplTofLimits):
  """Read neutron events from an MCPL file or a .dat file (created with the
  Virtual_output McStas component). In case of an MCPL file, convert units and
  apply filtering based on the time (TOF) field using the provided limits."""
  print(f'Reading events from {filename}...')
  printTofLimits(mcplTofLimits)
  if filename.endswith('.mcpl') or filename.endswith('.mcpl.gz'):
    with mcpl.MCPLFile(filename) as myfile:
      events = np.array(
        [(p.weight,
          p.x/100, p.y/100, p.z/100, #convert cm->m
          *getVelocity(p.ux, p.uy, p.uz, p.ekin),
          p.time*1e-3 #convert ms->s
          ) for p in myfile.particles if (p.weight>1e-5 and mcplTofLimits[0] < p.time and p.time < mcplTofLimits[1])]
        )
  elif filename.endswith('.dat'): #legacy file type
    events = np.loadtxt(filename)
  else:
    sys.exit("Wrong input file extension. Expected: '.mcpl', '.mcpl.gz' or '.dat'" )
  if len(events)==0:
    if mcplTofLimits != [float('-inf'), float('inf')]:
      sys.exit(f"No neutrons found in the input file ({filename}) within the TOF filtering limits: {mcplTofLimits[0]:.3f} - {mcplTofLimits[1]:.3f} [millisecond]!")
    else:
      sys.exit(f"No neutrons found in the input file {filename}.")
  return events

def saveQHistogramFile(savename, qHist, qHistError, edges):
  """Save the histograms are corresponding bin edges in an NPZ file"""
  np.savez_compressed(savename, hist=qHist, error=qHistError, xEdges=edges[0], yEdges=edges[1], zEdges=edges[2])
  print(f"Created {savename}.npz")

def unpackQHistogramFile(npFile):
  """Unpack the content (histograms are corresponding bin edges) of an NPZ file
  created with the saveQHistogramFile function"""
  hist = npFile['hist'] #keys are hardcoded in the saveQHistogramFile function
  histError = npFile['error']
  xEdges = npFile['xEdges']
  yEdges = npFile['yEdges']
  zEdges = npFile['zEdges']
  return hist, histError, xEdges, yEdges, zEdges

def saveRawQListFile(savename, qArray):
  """Save the list of Q events in an NPZ file"""
  np.savez_compressed(savename, qArray=qArray)
  print(f"Created {savename}.npz with raw Q events.")

def unpackRawQListFile(npFile):
  """Unpack the content (list of Q events) of an NPZ file created with the
  saveRawQListFile function"""
  npFileArrayKey = npFile.files[0] #might as well use hardcoded 'qArray' key from saveRawQListFile function
  qArray = npFile[npFileArrayKey]
  weights = qArray[:, 0]
  x = qArray[:, 1]
  y = qArray[:, 2]
  z = qArray[:, 3]
  return x, y, z, weights

### Currently unused function to write MCPL file ###
# requred np2mcpl python packages (also commented out in the conda.yml file)
# def write_events_mcpl(out_events, filename, deweight=False):
#     import np2mcpl
#     weights, x, y, z, vx, vy, vz, t = out_events.T

#     pdg_codes = np.full((len(out_events), 1), 2112) #2112 for neutrons

#     # Adjust dimensions for MCPL (m->cm, s->ms)
#     x = x * 100
#     y = y * 100
#     z = z * 100
#     t = t * 1e3

#     # Calculate the kinetic energy
#     nrm = np.sqrt(vx**2 + vy**2 + vz**2)
#     e_kin = (nrm**2) / 1e9 * VS2E
#     # Normalize the velocity vector
#     ux = vx / nrm
#     uy = vy / nrm
#     uz = vz / nrm

#     sx, sy, sz = (0, 0, 0)
#     particles = np.column_stack((pdg_codes, x, y, z, ux, uy, uz, t, e_kin, weights, sx, sy, sz))

#     if deweight:
#       # weights_above_1p5 = weights > 1.5
#       # print(f'len: {len(weights)},sum: {sum(weights)}, min: {min(weights)}, max: {max(weights)}, w>1.5: {len(weights[weights_above_1p5])} ')
#       # Process particles with weights above 1: save N full-weight(1) copies for each particle with weight N.x > 1.0 (e.g. 3 for w=3.2)
#       high_weight_mask = weights >= 1.0 # Find particles with weights above 1
#       integer_weights = np.floor(weights[high_weight_mask]).astype(int)
#       additional_surviving_particles = np.repeat(particles[high_weight_mask], integer_weights, axis=0) #weight of saved particles is handled later
#       weights[high_weight_mask] -= integer_weights # Update the weights for the original high-weight particles -> all weights are now below 1.0

#       # Determine which 'partial' (weight<1) particles survive the Russian Roulette
#       surviving_mask = np.random.rand(len(weights)) <= weights

#       # Concatenate the additional surviving particles with the surviving particles
#       deweighted_particles = np.concatenate([particles[surviving_mask], additional_surviving_particles], axis=0)
#       deweighted_particles[:, 9] = 1.0 #set all weights to 1

#       np2mcpl.save(filename, deweighted_particles)
#     else:
#       np2mcpl.save(filename+'_weighted', particles)
