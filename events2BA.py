"""
Load events from McStas to run a BornAgain simulation and create new neutron events from the results
to feed back to McStas.
"""

from importlib import import_module
import sys
from numpy import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import bornagain as ba
from bornagain import deg, angstrom, nm

from neutron_utilities import VS2E, V2L, tofToLambda

EFILE = "" #"GISANS_events/test_events" # event file to be used
OFILE = "test_events_scattered" # event file to be written
MFILE = "models.silica_100nm_air"
sim_module=import_module(MFILE)
get_sample=sim_module.get_sample

BINS=10 # number of pixels in x and y direction of the "detector"
ANGLE_RANGE=3 # degree scattering angle covered by detector

xwidth=0.01 # [m] size of sample perpendicular to beam
yheight=0.03 # [m] size of sample along the beam

alpha_inc = 0.35 *pi/180 #rad
v_in_alpha = array([0, cos(alpha_inc), sin(alpha_inc)])
#rotation matrix to compensate alpha_inc rotation from MCPL_output component in McStas model
rot_matrix = array([[cos(alpha_inc), -sin(alpha_inc)],[sin(alpha_inc), cos(alpha_inc)]]) 
rot_matrix_inverse = array([[cos(-alpha_inc), -sin(-alpha_inc)],[sin(-alpha_inc), cos(-alpha_inc)]]) 

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

def virtualPropagationToDetector(x, y, z, vx, vy, vz):
  """Calculate x,y,z position on the detector surface and the corresponding tof for the sample to detector propagation"""
  #compensate coord system rotation
  _, yRot = dot(rot_matrix, [z, y])
  _, vyRot = dot(rot_matrix, [vz, vy])

  #propagate to detector surface perpendicular to the y-axis
  t_propagate= (sample_detector_distance - yRot) / vyRot
  
  x = x + vx * t_propagate
  y = y + vy * t_propagate
  z = z + vz * t_propagate

  return x, y, z, t_propagate

def qConvFactorFromTofAtDetector(sample_detector_path_length, tDet):
    path_length = nominal_source_sample_distance + sample_detector_path_length
    return 2*pi/(tofToLambda(tDet, path_length)*0.1)

def virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ):
  """Calculate x,y,z position on the detector surface and the corresponding tof for the sample to detector propagation"""
  #Calculate time until the detector surface with coord system rotation
  _, yRot = dot(rot_matrix, [z, y])
  _, vyRot = matmul(rot_matrix, vstack((VZ, VY))) # get [vz, vy]

  #propagate to detector surface perpendicular to the y-axis
  t_propagate = (sample_detector_distance - yRot) / vyRot
  
  return t_propagate, (VX * t_propagate + x), (VY * t_propagate + y), (VZ * t_propagate + z) 

def getQsAtDetector(x, y, z, t, v_in_alpha, VX, VY, VZ): 
  sample_detector_tof, xDet, yDet, zDet = virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ)
  posDetector = vstack((xDet, yDet, zDet)).T
  sample_detector_path_length = linalg.norm(posDetector, axis=1)

  v_out_det = posDetector / sample_detector_path_length[:, newaxis]

  tDet = sample_detector_tof + t
  path_length = sample_detector_path_length + nominal_source_sample_distance

  qConvFactorFromTofAtDetector = 2*pi/(tofToLambda(tDet, path_length)*0.1)

  qArray = (v_out_det - v_in_alpha) * qConvFactorFromTofAtDetector[:, newaxis]
  return qArray, tDet

def run_events(events):
    misses = 0
    total = len(events)
    out_events = []
    q_events_real = [] #p,qx,qy,qz,t - calculated with lambda and proper incident and outgoing directions
    q_events_no_incident_info = [] #p,qx,qy,qz,t - calculated with lambda but not using incident direction

    v_in_alpha = array([0, cos(alpha_inc), sin(alpha_inc)])
    q_events_calc_sample = [] #p,qx,qy,qz,t - calculated from TOF at sample position
    q_events_calc_detector = [] #p,qx,qy,qz,t - calculated from TOF at the detector position

    for in_ID, neutron in enumerate(events):
        if in_ID%200==0:
            print(f'{in_ID:10}/{total}')
        p, x, y, z, vx, vy, vz, t, sx, sy, sz = neutron
        alpha_i = arctan(vz/vy)*180./pi  # deg
        phi_i = arctan(vx/vy)*180./pi  # deg
        v = sqrt(vx**2+vy**2+vz**2)
        wavelength = V2L/v  # Å
        qConvFactorFromLambda = 2*pi/(wavelength * 0.1)
        qConvFactorFromTof = 2*pi/(tofToLambda(t)*0.1) #for an intermediate result
        v_in = array([vx, vy, vz]) / v

        if abs(x)>xwidth or abs(y)>yheight:
            # beam has not hit the sample surface
            out_events.append(neutron)
            misses += 1
        else:
            # beam has hit the sample
            sample = get_sample(phi_i)

            # Calculated reflected and transmitted (1-reflected) beams
            ssim = get_simulation_specular(sample, wavelength, alpha_i)
            res = ssim.simulate()
            # pref = p*res.array()[0]
            # out_events.append([pref, x, y, z, vx, vy, -vz, t, sx, sy, sz])
            # v_out = array([vx, vy, -vz]) / v
            # q_events_real.append([pref, *(qConvFactorFromLambda * subtract(v_out, v_in)), t])
            # q_events_no_incident_info.append([pref, *(qConvFactorFromLambda * subtract(v_out, v_in_alpha)), t])
            # q_events_calc_sample.append([pref, *(qConvFactorFromTof * subtract(v_out, v_in_alpha)), t])

            # xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vx, vy, -vz)
            # sample_detector_path_length = linalg.norm([xDet, yDet, zDet])
            # v_out_det = [xDet, yDet, zDet] / sample_detector_path_length
            # q_events_calc_detector.append([pref, *(qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof) * subtract(v_out_det, v_in_alpha)), t])

            # ptrans = (1.0-res.array()[0])*p
            # if ptrans>1e-10:
            #     out_events.append([ptrans, x, y, z, vx, vy, vz, t, sx, sy, sz])
            #     q_events_real.append([ptrans, *(qConvFactorFromLambda * subtract(v_in, v_in)), t])
            #     q_events_no_incident_info.append([ptrans, *(qConvFactorFromLambda * subtract(v_in, v_in_alpha)), t])
            #     q_events_calc_sample.append([ptrans, *(qConvFactorFromTof * subtract(v_in, v_in_alpha)), t])
            #     xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vx, vy, vz)
            #     sample_detector_path_length = linalg.norm([xDet, yDet, zDet])
            #     v_out_det = [xDet, yDet, zDet] / sample_detector_path_length
            #     q_events_calc_detector.append([ptrans, *(qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof) * subtract(v_out_det, v_in_alpha)), t])
            # calculate BINS² outgoing beams with a random angle within one pixel range
            Ry = 2*random.random()-1
            Rz = 2*random.random()-1
            sim = get_simulation(sample, wavelength, alpha_i, p, Ry, Rz)
            sim.options().setUseAvgMaterials(True)
            res = sim.simulate()
            # get probability (intensity) for all pixels
            pout = res.array()
            # calculate beam angle relative to coordinate system, including incident beam direction
            alpha_f = ANGLE_RANGE*(linspace(1., -1., BINS)+Ry/(BINS-1))
            phi_f = phi_i+ANGLE_RANGE*(linspace(-1., 1., BINS)+Rz/(BINS-1))
            alpha_f_rad = alpha_f * pi/180.
            phi_f_rad = phi_f * pi/180.
            alpha_grid, phi_grid = meshgrid(alpha_f_rad, phi_f_rad)

            VX_grid = v * cos(alpha_grid) * sin(phi_grid)
            VY_grid = v * cos(alpha_grid) * cos(phi_grid)
            VZ_grid = -v * sin(alpha_grid)

            for pouti, vxi, vyi, vzi in zip(pout.T.flatten(), VX_grid.flatten(),  VY_grid.flatten(), VZ_grid.flatten()):
                out_events.append([pouti, x, y, z, vxi, vyi, vzi, t, sx, sy, sz])
                v_out = array([vxi, vyi, vzi]) / v
                q_events_real.append([pouti, *(qConvFactorFromLambda * subtract(v_out, v_in)), t])
                q_events_no_incident_info.append([pouti, *(qConvFactorFromLambda * subtract(v_out, v_in_alpha)), t])
                q_events_calc_sample.append([pouti, *(qConvFactorFromTof * subtract(v_out, v_in_alpha)), t])

                xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vxi, vyi, vzi)
                sample_detector_path_length = linalg.norm([xDet, yDet, zDet])
                v_out_det = array([xDet, yDet, zDet]) / sample_detector_path_length
                q_events_calc_detector.append([pouti, *(qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof) * subtract(v_out_det, v_in_alpha)), t])

    print("misses:", misses)
    return array(out_events), array(q_events_real), array(q_events_no_incident_info), array(q_events_calc_sample), array(q_events_calc_detector)

def processNeutron(neutron):
  p, x, y, z, vx, vy, vz, t, _, _, _ = neutron
  alpha_i = arctan(vz/vy)*180./pi  # deg
  phi_i = arctan(vx/vy)*180./pi  # deg
  v = sqrt(vx**2+vy**2+vz**2)
  wavelength = V2L/v  # Å

  if abs(x)>xwidth or abs(y)>yheight:
    # beam has not hit the sample surface
    return []
  else:
    # beam has hit the sample
    sample = get_sample(phi_i)

    # Calculated reflected and transmitted (1-reflected) beams
    # ssim = get_simulation_specular(sample, wavelength, alpha_i)
    # res = ssim.simulate()
    # pref = p*res.array()[0]
    # xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vx, vy, -vz)
    # sample_detector_path_length = linalg.norm([xDet, yDet, zDet])
    # v_out_det = [xDet, yDet, zDet] / sample_detector_path_length
    # q_events_calc_detector.append([pref, *(qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof) * subtract(v_out_det, v_in_alpha)), t])

    # ptrans = (1.0-res.array()[0])*p
    # if ptrans>1e-10:
    #     xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vx, vy, vz)
    #     sample_detector_path_length = linalg.norm([xDet, yDet, zDet])
    #     v_out_det = [xDet, yDet, zDet] / sample_detector_path_length
    #     q_events_calc_detector.append([ptrans, *(qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof) * subtract(v_out_det, v_in_alpha)), t])
    # calculate BINS² outgoing beams with a random angle within one pixel range
    Ry = 2*random.random()-1
    Rz = 2*random.random()-1
    sim = get_simulation(sample, wavelength, alpha_i, p, Ry, Rz)
    sim.options().setUseAvgMaterials(True)
    res = sim.simulate()
    # get probability (intensity) for all pixels
    pout = res.array()
    # calculate beam angle relative to coordinate system, including incident beam direction
    alpha_f = ANGLE_RANGE*(linspace(1., -1., BINS)+Ry/(BINS-1))
    phi_f = phi_i+ANGLE_RANGE*(linspace(-1., 1., BINS)+Rz/(BINS-1))
    alpha_f_rad = alpha_f * pi/180.
    phi_f_rad = phi_f * pi/180.
    alpha_grid, phi_grid = meshgrid(alpha_f_rad, phi_f_rad)

    VX_grid = v * cos(alpha_grid) * sin(phi_grid)
    VY_grid = v * cos(alpha_grid) * cos(phi_grid)
    VZ_grid = -v * sin(alpha_grid)

    # qArray, tDet = getQsAtDetector(x, y, z, t, v_in_alpha, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten())
    qArray, _ = getQsAtDetector(x, y, z, t, v_in_alpha, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten())

    return column_stack([pout.T.flatten(), qArray])

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


def main():
    if len(sys.argv)>1:
      # MFILE='models.'+sys.argv[1]
      EFILE=sys.argv[2] #TODO implement proper argparser?

    print(f'Reading events from {EFILE}...')
    if EFILE.endswith('.dat'):
      events = loadtxt(EFILE)

    elif EFILE.endswith('.mcpl') or EFILE.endswith('.mcpl.gz'):
      import mcpl
      myfile = mcpl.MCPLFile(EFILE)
      def velocity_from_dir(ux, uy, uz, ekin):
         norm = sqrt(ekin*1e9/VS2E)
         return [ux*norm, uy*norm, uz*norm]
      events = array([(p.weight,
                       p.x/100, p.y/100, p.z/100, #convert cm->m
                       *velocity_from_dir(p.ux, p.uy, p.uz, p.ekin),
                       p.time*1e-3, #convert s->ms
                       p.polx, p.poly, p.polz) for p in myfile.particles if p.weight>1e-5])
    else:
      sys.exit("Wrong input file extension. Expected: '.dat', '.mcpl', or '.mcpl.gz")

    events = prop0(events)
    # print(f'Running BornAgain simulations "{MFILE}" for each event...')
    # global get_sample
    # sim_module=import_module(MFILE)
    # get_sample=sim_module.get_sample

    if not 'all_q' in sys.argv:
      if 'no_parallel' in sys.argv:
        total=len(events)
        q_events_calc_detector = []
        for in_ID, neutron in enumerate(events):
          if in_ID%200==0:
            print(f'{in_ID:10}/{total}')
          tmp = processNeutron(neutron)
          q_events_calc_detector.extend(tmp)
      else:
        print('Number of events being processed: ', len(events))
        # Determine the number of CPU cores
        num_processes = cpu_count() - 2 # Leave one core free for other tasks
        print('cpu_count: ', cpu_count(), ' num_processes:',num_processes)
        with Pool(processes=num_processes) as pool:
          # Use tqdm to wrap the iterable returned by pool.imap for the progressbar
          q_events = list(tqdm(pool.imap(processNeutron, events), total=len(events)))

        q_events_calc_detector = [item for sublist in q_events for item in sublist]

      savez_compressed("q_events_calc_detector.npz", q_events_calc_detector=q_events_calc_detector)
    else:
      out_events, q_events_real, q_events_no_incident_info, q_events_calc_sample, q_events_calc_detector = run_events(events)
      savez_compressed("q_events_real.npz", q_events_real=q_events_real)
      savez_compressed("q_events_no_incident_info.npz", q_events_no_incident_info=q_events_no_incident_info)
      savez_compressed("q_events_calc_sample.npz", q_events_calc_sample=q_events_calc_sample)
      savez_compressed("q_events_calc_detector.npz", q_events_calc_detector=q_events_calc_detector)
      # print(f'Writing events to {OFILE}...')
      # write_events(out_events)
      # from output_mcpl import write_events_mcpl
      # deweight = False #Ensure final weight of 1 using splitting and Russian Roulette
      # write_events_mcpl(out_events, OFILE, deweight)

if __name__=='__main__':
    main()
