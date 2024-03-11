"""
Load events from McStas to run a BornAgain simulation and create new neutron events from the results
to feed back to McStas.
"""

from importlib import import_module
import sys
from numpy import *

import bornagain as ba
from bornagain import deg, angstrom, nm

EFILE = "" #"GISANS_events/test_events" # event file to be used
OFILE = "test_events_scattered" # event file to be written
MFILE = "models.hexagonal_spheres"

BINS=10 # number of pixels in x and y direction of the "detector"
ANGLE_RANGE=3 # degree scattering angle covered by detector

V2L = 3956.034012 # m/s·Å
xwidth=0.05 # [m] size of sample perpendicular to beam
yheight=0.15 # [m] size of sample along the beam
deweight = True #Ensure final weight of 1 using splitting and Russian Roulette
VS2E = 5.22703725e-6 # Conversion factor from McStas (v[m/s])**2 to E[meV])

def prop0(events):
    # propagate neutron events to z=0, the sample surface
    p, x, y, z, vx, vy, vz, t, sx, sy, sz = events.T
    t0 = -z/vz
    x += vx*t0
    y += vy*t0
    z += vz*t0
    t+=t0
    return vstack([p, x, y, z, vx, vy, vz, t, sx, sy, sz]).T

def get_simulation(sample, wavelength=6.0, alpha_i=0.2, p=1.0, Ry=0., Rz=0.):
    """
    Create a simulation with BINS² pixels that cover an angular range of
    ANGLE_RANGE degrees.
    The Ry and Rz values are relative rotations of the detector within one pixel
    to finely define the outgoing direction of events.
    """
    beam = ba.Beam(p, wavelength*angstrom, alpha_i*deg)

    dRy = Ry*ANGLE_RANGE*deg/(BINS-1)
    dRz = Rz*ANGLE_RANGE*deg/(BINS-1)

    # Define detector
    detector = ba.SphericalDetector(BINS, -ANGLE_RANGE*deg+dRz, ANGLE_RANGE*deg+dRz,
                                    BINS, -ANGLE_RANGE*deg+dRy, ANGLE_RANGE*deg+dRy)

    return ba.ScatteringSimulation(beam, sample, detector)

def get_simulation_specular(sample, wavelength=6.0, alpha_i=0.2):
    scan = ba.AlphaScan(2, alpha_i*deg, alpha_i*deg+1e-6)
    scan.setWavelength(wavelength*angstrom)
    return ba.SpecularSimulation(scan, sample)


def run_events(events):
    misses = 0
    total = len(events)
    out_events = []
    for in_ID, neutron in enumerate(events):
        if in_ID%200==0:
            print(f'{in_ID:10}/{total}')
        p, x, y, z, vx, vy, vz, t, sx, sy, sz = neutron
        alpha_i = arctan(vz/vy)*180./pi  # deg
        phi_i = arctan(vx/vy)*180./pi  # deg
        v = sqrt(vx**2+vy**2+vz**2)
        wavelength = V2L/v  # Å

        if abs(x)>xwidth or abs(z)>yheight:
            # beam has not hit the sample surface
            out_events.append(neutron)
            misses += 1
        else:
            # beam has hit the sample
            sample = get_sample(phi_i)

            # Calculated reflected and transmitted (1-reflected) beams
            ssim = get_simulation_specular(sample, wavelength, alpha_i)
            res = ssim.simulate()
            pref = p*res.array()[0]
            out_events.append([pref, x, y, z, vx, vy, -vz, t, sx, sy, sz])
            ptrans = (1.0-res.array()[0])*p
            if ptrans>1e-10:
                out_events.append([ptrans, x, y, z, vx, vy, vz, t, sx, sy, sz])

            # calculate BINS² outgoing beams with a random angle within one pixel range
            Ry =  2*random.random()-1
            Rz =  2*random.random()-1
            sim = get_simulation(sample, wavelength, alpha_i, p, Ry, Rz)
            sim.options().setUseAvgMaterials(True)
            res = sim.simulate()
            # get probability (intensity) for all pixels
            pout = res.array()
            # calculate beam angle relative to coordinate system, including incident beam direction
            alpha_f = ANGLE_RANGE*(linspace(1., -1., BINS)+Ry/(BINS-1))
            phi_f = phi_i+ANGLE_RANGE*(linspace(-1., 1., BINS)+Rz/(BINS-1))
            VX, VZ= meshgrid(tan(phi_f*pi/180.)*vy, tan(alpha_f*pi/180.)*vy)
            for pouti, vxi, vzi in zip(pout.flatten(), VX.flatten(), VZ.flatten()):
                out_events.append([pouti, x, y, z, vxi, vy, vzi, t, sx, sy, sz])
    print("misses:", misses)
    return array(out_events)

def write_events(out_events):
    header = ''
    with open(EFILE+'.dat', 'r') as fh:
        line = fh.readline()
        while line.startswith('#'):
            header += line
            line = fh.readline()
    with open(OFILE+'.dat', 'w') as fh:
        fh.write(header)
        savetxt(fh, out_events)

def write_events_mcpl(out_events):
    import np2mcpl
    weights, x, y, z, vx, vy, vz, t, sx, sy, sz = out_events.T

    pdg_codes = full((len(out_events), 1), 2112) #2112 for neutrons

    # Adjust dimensions for MCPL (m->cm, s->ms)
    x = x * 100
    y = y * 100
    z = z * 100
    t = t * 1e3

    # Calculate the kinetic energy
    nrm = sqrt(vx**2 + vy**2 + vz**2)
    e_kin = (nrm**2) / 1e9 * VS2E
    # Normalize the velocity vector
    ux = vx / nrm
    uy = vy / nrm
    uz = vz / nrm

    particles = column_stack((pdg_codes, x, y, z, ux, uy, uz, t, e_kin, weights, sx, sy, sz))

    if deweight:
      # Process particles with weights above 1: save N full-weight(1) copies for each particle with weight N.x > 1.0 (e.g. 3 for w=3.2)
      high_weight_mask = weights >= 1.0 # Find particles with weights above 1
      integer_weights = floor(weights[high_weight_mask]).astype(int)
      additional_surviving_particles = repeat(particles[high_weight_mask], integer_weights, axis=0) #weight of saved particles is handled later
      weights[high_weight_mask] -= integer_weights # Update the weights for the original high-weight particles -> all weihts are now below 1.0

      # Determine which 'partial' (weight<1) particles survive the Russian Roulette
      surviving_mask = random.rand(len(weights)) <= weights

      # Concatenate the additional surviving particles with the surviving particles
      deweighted_particles = concatenate([particles[surviving_mask], additional_surviving_particles], axis=0)
      deweighted_particles[:, 9] = 1.0 #set all weights to 1

      np2mcpl.save(OFILE, deweighted_particles)
    else:
      np2mcpl.save(OFILE+'_weighted', particles)

def main():
    if len(sys.argv)>1:
      MFILE='models.'+sys.argv[1]
      EFILE=sys.argv[2] #TODO implement proper argparser?

    print(f'Reading events from {EFILE}...')
    if EFILE.endswith('.dat'):
      events = loadtxt(EFILE)
      
    elif EFILE.endswith('.mcpl') or EFILE.endswith('.mcpl.gz'):
      import mcpl
      myfile = mcpl.MCPLFile(EFILE)
      events = array([(p.weight, 
                       p.x/100, p.y/100, p.z/100, #convert cm->m
                       p.ux*sqrt(p.ekin*1e9/VS2E), p.uy*sqrt(p.ekin*1e9/VS2E), p.uz*sqrt(p.ekin*1e9/VS2E), #velocity vector from dir+ekin
                       p.time*1e-3, #convert s->ms
                       p.polx, p.poly, p.polz) for p in myfile.particles if p.weight>1e-5])
    else:
      sys.exit("Wrong input file extension. Expected: '.dat', '.mcpl', or '.mcpl.gz")

    events = prop0(events)
    print(f'Running BornAgain simulations "{MFILE}" for each event...')
    global get_sample
    sim_module=import_module(MFILE)
    get_sample=sim_module.get_sample
    out_events = run_events(events)
    print(f'Writing events to {OFILE}...')
    # write_events(out_events)
    write_events_mcpl(out_events)

if __name__=='__main__':
    main()
