"""
Load events from McStas to run a BornAgain simulation and create new neutron events from the results
to feed back to McStas.
"""

from importlib import import_module
import sys
import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory
from tqdm import tqdm

import bornagain as ba
from bornagain import deg, angstrom

from neutron_utilities import VS2E, V2L, tofToLambda

ANGLE_RANGE=3 # degree scattering angle covered by detector

xwidth=0.06 # [m] size of sample perpendicular to beam
yheight=0.08 # [m] size of sample along the beam

sharedMemoryName = 'sharedProcessMemory'
defaultSampleModel = "models.silica_100nm_air"
sim_module=import_module(defaultSampleModel)
sharedTemplate = np.array([
   0.0,                # nominal_source_sample_distance
   0.0,                # sample_detector_distance
   defaultSampleModel, # sample model
   0,                  # BINS (~detector resolution)
   0.0                 # wavelength selected (for non-TOF instruments)
   ])

instrumentParameters = {
   'saga': {
      'nominal_source_sample_distance' : 55.0, #[m]
      'sample_detector_distance' : 10, #[m] along the y axis
      'tof instrument' : True
   },
   'loki': {
      'nominal_source_sample_distance' : 23.6,
      'sample_detector_distance' : 5,
      'tof instrument' : True
   },
   'd22': {
      'nominal_source_sample_distance' : 61.28, #approximate value, but it is not really used
      'sample_detector_distance' : 17.6,
      'tof instrument' : False
   }
}

alpha_inc = 0.24 *np.pi/180 #rad #TODO turn it into an input parameter, take care of the derived values below
v_in_alpha = np.array([0, np.cos(alpha_inc), np.sin(alpha_inc)])
#rotation matrix to compensate alpha_inc rotation from MCPL_output component in McStas model
rot_matrix = np.array([[np.cos(alpha_inc), -np.sin(alpha_inc)],[np.sin(alpha_inc), np.cos(alpha_inc)]])
rot_matrix_inverse = np.array([[np.cos(-alpha_inc), -np.sin(-alpha_inc)],[np.sin(-alpha_inc), np.cos(-alpha_inc)]])

def prop0(events):
    # propagate neutron events to z=0, the sample surface
    p, x, y, z, vx, vy, vz, t, sx, sy, sz = events.T
    t0 = -z/vz
    x += vx*t0
    y += vy*t0
    z += vz*t0
    t+=t0
    return np.vstack([p, x, y, z, vx, vy, vz, t, sx, sy, sz]).T

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
  _, yRot = np.dot(rot_matrix, [z, y])
  _, vyRot = np.dot(rot_matrix, [vz, vy])

  #propagate to detector surface perpendicular to the y-axis
  t_propagate= (sample_detector_distance - yRot) / vyRot

  x = x + vx * t_propagate
  y = y + vy * t_propagate
  z = z + vz * t_propagate

  return x, y, z, t_propagate

def qConvFactorFromTofAtDetector(sample_detector_path_length, tDet):
    path_length = nominal_source_sample_distance + sample_detector_path_length
    return 2*np.pi/(tofToLambda(tDet, path_length)*0.1)

def virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ):
  """Calculate x,y,z position on the detector surface and the corresponding tof for the sample to detector propagation"""
  #Calculate time until the detector surface with coord system rotation
  _, yRot = np.dot(rot_matrix, [z, y])
  _, vyRot = np.matmul(rot_matrix, np.vstack((VZ, VY))) # get [vz, vy]

  #propagate to detector surface perpendicular to the y-axis
  t_propagate = (sample_detector_distance - yRot) / vyRot

  return t_propagate, (VX * t_propagate + x), (VY * t_propagate + y), (VZ * t_propagate + z)

def getQsAtDetector(x, y, z, t, v_in_alpha, VX, VY, VZ):
  sample_detector_tof, xDet, yDet, zDet = virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ)
  posDetector = np.vstack((xDet, yDet, zDet)).T
  sample_detector_path_length = np.linalg.norm(posDetector, axis=1)

  v_out_det = posDetector / sample_detector_path_length[:, np.newaxis]

  tDet = sample_detector_tof + t

  if notTOFInstrument is False: #TOF instruments
    path_length = sample_detector_path_length + nominal_source_sample_distance
    qConvFactorFromTofAtDetector = 2*np.pi/(tofToLambda(tDet, path_length)*0.1)
    qArray = (v_out_det - v_in_alpha) * qConvFactorFromTofAtDetector[:, np.newaxis]
  else:
    qArray = (v_out_det - v_in_alpha) * qConvFactorFixed

  return qArray, tDet

def run_events(events):
    misses = 0
    total = len(events)
    out_events = []
    q_events_real = [] #p,qx,qy,qz,t - calculated with lambda and proper incident and outgoing directions
    q_events_no_incident_info = [] #p,qx,qy,qz,t - calculated with lambda but not using incident direction

    v_in_alpha = np.array([0, np.cos(alpha_inc), np.sin(alpha_inc)])
    q_events_calc_sample = [] #p,qx,qy,qz,t - calculated from TOF at sample position
    q_events_calc_detector = [] #p,qx,qy,qz,t - calculated from TOF at the detector position

    notTOFInstrument = wavelengthSelected is not None # just to make the code more readable later on
    qConvFactorFixed = None if notTOFInstrument is False else 2*np.pi/(wavelengthSelected*0.1)

    for in_ID, neutron in enumerate(events):
        if in_ID%200==0:
            print(f'{in_ID:10}/{total}')
        p, x, y, z, vx, vy, vz, t, sx, sy, sz = neutron
        alpha_i = np.arctan(vz/vy)*180./np.pi  # deg
        phi_i = np.arctan(vx/vy)*180./np.pi  # deg
        v = np.sqrt(vx**2+vy**2+vz**2)
        wavelength = V2L/v  # Å
        qConvFactorFromLambda = 2*np.pi/(wavelength * 0.1)
        qConvFactorFromTof = qConvFactorFixed if notTOFInstrument else 2*np.pi/(tofToLambda(t)*0.1) #for an intermediate result
        v_in = np.array([vx, vy, vz]) / v

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
            pref = p*res.array()[0]
            out_events.append([pref, x, y, z, vx, vy, -vz, t, sx, sy, sz])
            v_out = np.array([vx, vy, -vz]) / v
            q_events_real.append([pref, *(qConvFactorFromLambda * np.subtract(v_out, v_in)), t])
            q_events_no_incident_info.append([pref, *(qConvFactorFromLambda * np.subtract(v_out, v_in_alpha)), t])
            q_events_calc_sample.append([pref, *(qConvFactorFromTof * np.subtract(v_out, v_in_alpha)), t])

            xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vx, vy, -vz)
            sample_detector_path_length = np.linalg.norm([xDet, yDet, zDet])
            v_out_det = [xDet, yDet, zDet] / sample_detector_path_length
            qConvFactorFromTofAtDet = qConvFactorFixed if notTOFInstrument else qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof)
            q_events_calc_detector.append([pref, *(qConvFactorFromTofAtDet * np.subtract(v_out_det, v_in_alpha)), t])

            ptrans = (1.0-res.array()[0])*p
            if ptrans>1e-10:
                out_events.append([ptrans, x, y, z, vx, vy, vz, t, sx, sy, sz])
                q_events_real.append([ptrans, *(qConvFactorFromLambda * np.subtract(v_in, v_in)), t])
                q_events_no_incident_info.append([ptrans, *(qConvFactorFromLambda * np.subtract(v_in, v_in_alpha)), t])
                q_events_calc_sample.append([ptrans, *(qConvFactorFromTof * np.subtract(v_in, v_in_alpha)), t])
                xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vx, vy, vz)
                sample_detector_path_length = np.linalg.norm([xDet, yDet, zDet])
                v_out_det = [xDet, yDet, zDet] / sample_detector_path_length
                qConvFactorFromTofAtDet = qConvFactorFixed if notTOFInstrument else qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof)
                q_events_calc_detector.append([ptrans, *(qConvFactorFromTofAtDet * np.subtract(v_out_det, v_in_alpha)), t])
            #calculate BINS² outgoing beams with a random angle within one pixel range
            Ry = 2*np.random.random()-1
            Rz = 2*np.random.random()-1
            sim = get_simulation(sample, wavelength, alpha_i, p, Ry, Rz)
            sim.options().setUseAvgMaterials(True)
            res = sim.simulate()
            # get probability (intensity) for all pixels
            pout = res.array()
            # calculate beam angle relative to coordinate system, including incident beam direction
            alpha_f = ANGLE_RANGE*(np.linspace(1., -1., BINS)+Ry/(BINS-1))
            phi_f = phi_i+ANGLE_RANGE*(np.linspace(-1., 1., BINS)+Rz/(BINS-1))
            alpha_f_rad = alpha_f * np.pi/180.
            phi_f_rad = phi_f * np.pi/180.
            alpha_grid, phi_grid = np.meshgrid(alpha_f_rad, phi_f_rad)

            VX_grid = v * np.cos(alpha_grid) * np.sin(phi_grid)
            VY_grid = v * np.cos(alpha_grid) * np.cos(phi_grid)
            VZ_grid = -v * np.sin(alpha_grid)

            for pouti, vxi, vyi, vzi in zip(pout.T.flatten(), VX_grid.flatten(),  VY_grid.flatten(), VZ_grid.flatten()):
                out_events.append([pouti, x, y, z, vxi, vyi, vzi, t, sx, sy, sz])
                v_out = np.array([vxi, vyi, vzi]) / v
                q_events_real.append([pouti, *(qConvFactorFromLambda * np.subtract(v_out, v_in)), t])
                q_events_no_incident_info.append([pouti, *(qConvFactorFromLambda * np.subtract(v_out, v_in_alpha)), t])
                q_events_calc_sample.append([pouti, *(qConvFactorFromTof * np.subtract(v_out, v_in_alpha)), t])

                xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vxi, vyi, vzi)
                sample_detector_path_length = np.linalg.norm([xDet, yDet, zDet])
                v_out_det = np.array([xDet, yDet, zDet]) / sample_detector_path_length
                qConvFactorFromTofAtDet = qConvFactorFixed if notTOFInstrument else qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof)
                q_events_calc_detector.append([pouti, *(qConvFactorFromTofAtDet * np.subtract(v_out_det, v_in_alpha)), t])

    print("misses:", misses)
    return np.array(out_events), np.array(q_events_real), np.array(q_events_no_incident_info), np.array(q_events_calc_sample), np.array(q_events_calc_detector)

def addSharedMemoryValuesToGlobalSpace():
  global nominal_source_sample_distance
  global sample_detector_distance
  global get_sample
  global BINS
  global wavelengthSelected
  global notTOFInstrument # just to make the code more readable later on
  global qConvFactorFixed
  # Access shared values
  shm = shared_memory.SharedMemory(name=sharedMemoryName)
  mem = np.ndarray(sharedTemplate.shape, dtype=sharedTemplate.dtype, buffer=shm.buf)
  nominal_source_sample_distance = float(mem[0])
  sample_detector_distance = float(mem[1])
  sim_module_name = str(mem[2])
  sim_module=import_module(sim_module_name)
  get_sample=sim_module.get_sample
  BINS = int(mem[3])

  wavelengthSelected = None if mem[4] == 'None' else float(mem[4])
  notTOFInstrument = wavelengthSelected is not None
  qConvFactorFixed = None if wavelengthSelected is None else 2*np.pi/(wavelengthSelected*0.1)

  shm.close()

def processNeutron(neutron):
  addSharedMemoryValuesToGlobalSpace()

  p, x, y, z, vx, vy, vz, t, _, _, _ = neutron
  alpha_i = np.arctan(vz/vy)*180./np.pi  # deg
  phi_i = np.arctan(vx/vy)*180./np.pi  # deg
  v = np.sqrt(vx**2+vy**2+vz**2)
  wavelength = V2L/v  # Å

  if abs(x)>xwidth or abs(y)>yheight:
    # beam has not hit the sample surface
    return []
  else:
    # beam has hit the sample
    sample = get_sample(phi_i)

    # Calculated reflected and transmitted (1-reflected) beams
    ssim = get_simulation_specular(sample, wavelength, alpha_i)
    res = ssim.simulate()
    pref = p*res.array()[0]
    xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vx, vy, -vz)
    sample_detector_path_length = np.linalg.norm([xDet, yDet, zDet])
    v_out_det = [xDet, yDet, zDet] / sample_detector_path_length
    qConvFactorFromTofAtDet = qConvFactorFixed if notTOFInstrument else qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof)
    q_specularAndTrans = [[pref, *(qConvFactorFromTofAtDet * np.subtract(v_out_det, v_in_alpha))]] #, t]
    ptrans = (1.0-res.array()[0])*p
    if ptrans>1e-10:
      xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vx, vy, vz)
      sample_detector_path_length = np.linalg.norm([xDet, yDet, zDet])
      v_out_det = [xDet, yDet, zDet] / sample_detector_path_length
      qConvFactorFromTofAtDet = qConvFactorFixed if notTOFInstrument else qConvFactorFromTofAtDetector(sample_detector_path_length, t+sample_detector_tof)
      q_specularAndTrans.append([ptrans, *(qConvFactorFromTofAtDet * np.subtract(v_out_det, v_in_alpha))]) #, t])

    # calculate BINS² outgoing beams with a random angle within one pixel range
    Ry = 2*np.random.random()-1
    Rz = 2*np.random.random()-1
    sim = get_simulation(sample, wavelength, alpha_i, p, Ry, Rz)
    sim.options().setUseAvgMaterials(True)
    res = sim.simulate()
    # get probability (intensity) for all pixels
    pout = res.array()
    # calculate beam angle relative to coordinate system, including incident beam direction
    alpha_f = ANGLE_RANGE*(np.linspace(1., -1., BINS)+Ry/(BINS-1))
    phi_f = phi_i+ANGLE_RANGE*(np.linspace(-1., 1., BINS)+Rz/(BINS-1))
    alpha_f_rad = alpha_f * np.pi/180.
    phi_f_rad = phi_f * np.pi/180.
    alpha_grid, phi_grid = np.meshgrid(alpha_f_rad, phi_f_rad)

    VX_grid = v * np.cos(alpha_grid) * np.sin(phi_grid)
    VY_grid = v * np.cos(alpha_grid) * np.cos(phi_grid)
    VZ_grid = -v * np.sin(alpha_grid)

    # qArray, tDet = getQsAtDetector(x, y, z, t, v_in_alpha, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten())
    qArray, _ = getQsAtDetector(x, y, z, t, v_in_alpha, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten())

    # q_scattered = np.column_stack([pout.T.flatten(), qArray])
    return np.concatenate((np.column_stack([pout.T.flatten(), qArray]), np.array(q_specularAndTrans)))


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


def main(args):
    print(f'Reading events from {args.filename}...')
    if args.filename.endswith('.dat'):
      events = loadtxt(args.filename)

    elif args.filename.endswith('.mcpl') or args.filename.endswith('.mcpl.gz'):
      import mcpl
      myfile = mcpl.MCPLFile(args.filename)
      def velocity_from_dir(ux, uy, uz, ekin):
         norm = np.sqrt(ekin*1e9/VS2E)
         return [ux*norm, uy*norm, uz*norm]
      events = np.array([(p.weight,
                       p.x/100, p.y/100, p.z/100, #convert cm->m
                       *velocity_from_dir(p.ux, p.uy, p.uz, p.ekin),
                       p.time*1e-3, #convert ms->s
                       p.polx, p.poly, p.polz) for p in myfile.particles
                       if (p.weight>1e-5 and args.tof_min < p.time and p.time < args.tof_max)])
    else:
      sys.exit("Wrong input file extension. Expected: '.dat', '.mcpl', or '.mcpl.gz")

    events = prop0(events)

    savenameAddition = '' if args.savename == '' else f"_{args.savename}"
    if not args.all_q:
      if args.no_parallel:
        total=len(events)
        q_events_calc_detector = []
        for in_ID, neutron in enumerate(events):
          if in_ID%200==0:
            print(f'{in_ID:10}/{total}')
          tmp = processNeutron(neutron)
          q_events_calc_detector.extend(tmp)
      else:
        print('Number of events being processed: ', len(events))
        num_processes = args.parallel_processes if args.parallel_processes else (cpu_count() - 2)
        print(f"Number of parallel processes: {num_processes} (number of CPU cores: {cpu_count()})")
        with Pool(processes=num_processes) as pool:
          # Use tqdm to wrap the iterable returned by pool.imap for the progressbar
          q_events = list(tqdm(pool.imap(processNeutron, events), total=len(events)))

        q_events_calc_detector = [item for sublist in q_events for item in sublist]

      saveFilename = f"q_events_calc_detector{savenameAddition}_bins{BINS}.npz"
      np.savez_compressed(saveFilename, q_events_calc_detector=q_events_calc_detector)
      print(f"Created {saveFilename}")

    else:
      global get_sample
      sim_module=import_module(sim_module_name)
      get_sample=sim_module.get_sample
      out_events, q_events_real, q_events_no_incident_info, q_events_calc_sample, q_events_calc_detector = run_events(events)
      np.savez_compressed(f"q_events_real{savenameAddition}.npz", q_events_real=q_events_real)
      np.savez_compressed(f"q_events_no_incident_info{savenameAddition}.npz", q_events_no_incident_info=q_events_no_incident_info)
      np.savez_compressed(f"q_events_calc_sample{savenameAddition}.npz", q_events_calc_sample=q_events_calc_sample)
      np.savez_compressed(f"q_events_calc_detector{savenameAddition}.npz", q_events_calc_detector=q_events_calc_detector)
      # print(f'Writing events to {OFILE}...')
      # write_events(out_events)
      # from output_mcpl import write_events_mcpl
      # deweight = False #Ensure final weight of 1 using splitting and Russian Roulette
      # write_events_mcpl(out_events, OFILE, deweight)

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description = 'Execute BornAgain simulation of a GISANS sample with incident neutrons taken from an input file. The output of the script is a .npz file (or files) containing the derived Q values for each outgoing neutron. The default Q value calculated is aiming to be as close as possible to the Q value from a measurement.')
  parser.add_argument('filename',  help = 'Input filename. (Preferably MCPL file from the McStas MCPL_output component, but .dat file from McStas Virtual_output works as well)')
  parser.add_argument('-s', '--savename', default='', required=False, help = 'Optional addition to the default output filename(s).')
  parser.add_argument('--all_q', default=False, action='store_true', help = 'Calculate and save multiple Q values, each with different level of approximation (from real Q calculated from all simulation parameters to the default output value, that is Q calculated at the detector surface). This results in significantly slower simulations (especially due to the lack of parallelisation), but can shed light on the effect of e.g. divergence and TOF to lambda conversion on the derived Q value, in order to gain confidence in the results.')
  parser.add_argument('--no_parallel', default=False, action='store_true', help = 'Do not use multiprocessing. This makes the simulation significantly slower, but enables profiling, and the output of the number of neutrons missing the sample.')
  parser.add_argument('-p','--parallel_processes', required=False, type=int, help = 'Number of processes to be used for parallel processing.')
  parser.add_argument('-b','--detector_bins', default=10, type=int, help = 'Number of pixels in x and y direction of the "detector".')
  parser.add_argument('-m','--model', default=defaultSampleModel, help = 'BornAgain model to be used.')
  parser.add_argument('-i','--instrument', default='loki', choices=list(instrumentParameters.keys()), help = 'Instrument.')
  parser.add_argument('-w','--wavelengthSelected', default=6.0, type=float, help = 'Wavelength (mean) in Angstrom selected by the velocity selector. Only used for non-time-of-flight instruments.')
  parser.add_argument('--tof_min', default=0, type=float, help = 'Lower TOF limit in microseconds for selecting neutrons from the input MCPL file.')
  parser.add_argument('--tof_max', default=150, type=float, help = 'Upper TOF limit in microseconds for selecting neutrons from the input MCPL file')
  args = parser.parse_args()

  # Definitions here are global, so they will be accessible for non-parallel processing simulation
  nominal_source_sample_distance = instrumentParameters[args.instrument]['nominal_source_sample_distance']
  sample_detector_distance = instrumentParameters[args.instrument]['sample_detector_distance']
  sim_module_name = args.model
  BINS = args.detector_bins
  wavelengthSelected = None if instrumentParameters[args.instrument]['tof instrument'] else args.wavelengthSelected
  # Add globally constant parameters to a shared memory for -parallel processing simulation
  shared = np.array([
     nominal_source_sample_distance,
     sample_detector_distance,
     sim_module_name,
     BINS,
     wavelengthSelected
     ])
  shm = shared_memory.SharedMemory(create=True, size=sharedTemplate.nbytes, name=sharedMemoryName)
  mem = np.ndarray(sharedTemplate.shape, dtype=sharedTemplate.dtype, buffer=shm.buf) #Create a NumPy array backed by shared memory
  mem[:] = shared[:] # Copy to the shared memory

  try:
    main(args)
  finally:
    shm.close()
    shm.unlink()
