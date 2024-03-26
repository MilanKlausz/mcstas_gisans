"""
Output particles into an MCPL file using np2mcpl
"""

import numpy as np
from neutron_utilities import VS2E

def write_events_mcpl(out_events, filename, deweight=False):
    import np2mcpl
    weights, x, y, z, vx, vy, vz, t, sx, sy, sz = out_events.T

    pdg_codes = np.full((len(out_events), 1), 2112) #2112 for neutrons

    # Adjust dimensions for MCPL (m->cm, s->ms)
    x = x * 100
    y = y * 100
    z = z * 100
    t = t * 1e3

    # Calculate the kinetic energy
    nrm = np.sqrt(vx**2 + vy**2 + vz**2)
    e_kin = (nrm**2) / 1e9 * VS2E
    # Normalize the velocity vector
    ux = vx / nrm
    uy = vy / nrm
    uz = vz / nrm

    particles = np.column_stack((pdg_codes, x, y, z, ux, uy, uz, t, e_kin, weights, sx, sy, sz))

    if deweight:
      # weights_above_1p5 = weights > 1.5
      # print(f'len: {len(weights)},sum: {sum(weights)}, min: {min(weights)}, max: {max(weights)}, w>1.5: {len(weights[weights_above_1p5])} ')
      # Process particles with weights above 1: save N full-weight(1) copies for each particle with weight N.x > 1.0 (e.g. 3 for w=3.2)
      high_weight_mask = weights >= 1.0 # Find particles with weights above 1
      integer_weights = np.floor(weights[high_weight_mask]).astype(int)
      additional_surviving_particles = np.repeat(particles[high_weight_mask], integer_weights, axis=0) #weight of saved particles is handled later
      weights[high_weight_mask] -= integer_weights # Update the weights for the original high-weight particles -> all weights are now below 1.0

      # Determine which 'partial' (weight<1) particles survive the Russian Roulette
      surviving_mask = np.random.rand(len(weights)) <= weights

      # Concatenate the additional surviving particles with the surviving particles
      deweighted_particles = np.concatenate([particles[surviving_mask], additional_surviving_particles], axis=0)
      deweighted_particles[:, 9] = 1.0 #set all weights to 1

      np2mcpl.save(filename, deweighted_particles)
    else:
      np2mcpl.save(filename+'_weighted', particles)
