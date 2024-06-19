#!/usr/bin/env python3

"""
Load events from McStas to run a BornAgain simulation and create new neutron events from the results
to feed back to McStas and/or calculate and save Q values for each neutron for processing/plotting.
"""

from importlib import import_module
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import bornagain as ba
from bornagain import deg, angstrom

from neutron_utilities import V2L, tofToLambda
from instruments import instrumentParameters

from sharedMemory import createSharedMemory, getSharedMemoryValues, defaultSampleModel

ANGLE_RANGE=1.7 # degree scattering angle covered by detector

xwidth=0.06 # [m] size of sample perpendicular to beam
yheight=0.08 # [m] size of sample along the beam

alpha_inc = 0.24 *np.pi/180 #rad #TODO turn it into an input parameter, take care of the derived values below
v_in_alpha = np.array([0, np.cos(alpha_inc), np.sin(alpha_inc)])
#rotation matrix to compensate alpha_inc rotation from MCPL_output component in McStas model
rot_matrix = np.array([[np.cos(alpha_inc), -np.sin(alpha_inc)],[np.sin(alpha_inc), np.cos(alpha_inc)]])
rot_matrix_inverse = np.array([[np.cos(-alpha_inc), -np.sin(-alpha_inc)],[np.sin(-alpha_inc), np.cos(-alpha_inc)]])

def coordTransformToSampleSystem(events):
    """Apply coordinate transformation to express neutron parameters in a
    coordinate system with the sample in the centre and being horisontal"""
    p, x, y, z, vx, vy, vz, t = events.T
    zRot, yRot = np.dot(rot_matrix_inverse, [z, y])
    vzRot, vyRot = np.dot(rot_matrix_inverse, [vz, vy])
    return np.vstack([p, x, yRot, zRot, vx, vyRot, vzRot, t]).T

def propagateToSampleSurface(events):
    """Propagate neutron events to z=0, the sample surface"""
    p, x, y, z, vx, vy, vz, t= events.T
    t0 = -z/vz
    x += vx*t0
    y += vy*t0
    z += vz*t0
    t+=t0
    return np.vstack([p, x, y, z, vx, vy, vz, t]).T

def get_simulation(sample, bins, wavelength=6.0, alpha_i=0.2, p=1.0, Ry=0., Rz=0.):
    """
    Create a simulation with bins pixels that cover an angular range of
    ANGLE_RANGE degrees.
    The Ry and Rz values are relative rotations of the detector within one pixel
    to finely define the outgoing direction of events.
    """
    beam = ba.Beam(p, wavelength*angstrom, alpha_i*deg)

    dRy = Ry*ANGLE_RANGE*deg/(bins-1)
    dRz = Rz*ANGLE_RANGE*deg/(bins-1)

    # Define detector
    detector = ba.SphericalDetector(bins, -ANGLE_RANGE*deg+dRz, ANGLE_RANGE*deg+dRz,
                                    bins, -ANGLE_RANGE*deg+dRy, ANGLE_RANGE*deg+dRy)

    return ba.ScatteringSimulation(beam, sample, detector)

def qConvFactorFromTofAtDetector(sample_detector_path_length, tDet):
    path_length = nominal_source_sample_distance + sample_detector_path_length
    return 2*np.pi/(tofToLambda(tDet, path_length)*0.1)

def virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ, sample_detector_distance):
  """Calculate x,y,z position on the detector surface and the corresponding tof for the sample to detector propagation"""
  #Calculate time until the detector surface with coord system rotation
  _, yRot = np.dot(rot_matrix, [z, y])
  _, vyRot = np.matmul(rot_matrix, np.vstack((VZ, VY))) # get [vz, vy]

  #propagate to detector surface perpendicular to the y-axis
  t_propagate = (sample_detector_distance - yRot) / vyRot

  return t_propagate, (VX * t_propagate + x), (VY * t_propagate + y), (VZ * t_propagate + z)

def getQsAtDetector(x, y, z, t, v_in_alpha, VX, VY, VZ, nominal_source_sample_distance, sample_detector_distance, notTOFInstrument, qConvFactorFixed):
  sample_detector_tof, xDet, yDet, zDet = virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ, sample_detector_distance)
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

  return qArray

def processNeutrons(neutron):
  sv = getSharedMemoryValues()

  sim_module = import_module(sv['sim_module_name'])
  get_sample = sim_module.get_sample
  sample = get_sample(radius=sv['silicaRadius'])

  notTOFInstrument = sv['wavelengthSelected'] is not None
  qConvFactorFixed = None if sv['wavelengthSelected'] is None else 2*np.pi/(sv['wavelengthSelected']*0.1)

  p, x, y, z, vx, vy, vz, t = neutron
  alpha_i = np.arctan(vz/vy)*180./np.pi  # deg
  phi_i = np.arctan(vx/vy)*180./np.pi  # deg
  v = np.sqrt(vx**2+vy**2+vz**2)
  wavelength = V2L/v  # Å

  if abs(x)>xwidth or abs(y)>yheight:
    # beam has not hit the sample surface
    return []
  else:
    # beam has hit the sample

    # calculate bins outgoing beams with a random angle within one pixel range
    Ry = 2*np.random.random()-1
    Rz = 2*np.random.random()-1
    sim = get_simulation(sample, sv['bins'], wavelength, alpha_i, p, Ry, Rz)
    sim.options().setUseAvgMaterials(True)
    sim.options().setIncludeSpecular(True)

    res = sim.simulate()
    # get probability (intensity) for all pixels
    pout = res.array()
    # calculate beam angle relative to coordinate system, including incident beam direction
    alpha_f = ANGLE_RANGE*(np.linspace(1., -1., sv['bins'])+Ry/(sv['bins']-1))
    phi_f = phi_i+ANGLE_RANGE*(np.linspace(-1., 1., sv['bins'])+Rz/(sv['bins']-1))
    alpha_f_rad = alpha_f * np.pi/180.
    phi_f_rad = phi_f * np.pi/180.
    alpha_grid, phi_grid = np.meshgrid(alpha_f_rad, phi_f_rad)

    VX_grid = v * np.cos(alpha_grid) * np.sin(phi_grid)
    VY_grid = v * np.cos(alpha_grid) * np.cos(phi_grid)
    VZ_grid = -v * np.sin(alpha_grid)

    qArray = getQsAtDetector(x, y, z, t, v_in_alpha, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten(), sv['nominal_source_sample_distance'], sv['sample_detector_distance'], notTOFInstrument, qConvFactorFixed)

    return  np.column_stack([pout.T.flatten(), qArray])

def main(args):
    from inputOutput import getNeutronEvents
    events = getNeutronEvents(args.filename, args.tof_min, args.tof_max)
    events = coordTransformToSampleSystem(events)
    events = propagateToSampleSurface(events)

    savename = f"q_events_bins{bins}" if args.savename == '' else args.savename
    if not args.all_q:
      if args.no_parallel:
        total=len(events)
        q_events_calc_detector = []
        for in_ID, neutron in enumerate(events):
          if in_ID%200==0:
            print(f'{in_ID:10}/{total}')
          tmp = processNeutrons(neutron)
          q_events_calc_detector.extend(tmp)
      else:
        print('Number of events being processed: ', len(events))
        num_processes = args.parallel_processes if args.parallel_processes else (cpu_count() - 2)
        print(f"Number of parallel processes: {num_processes} (number of CPU cores: {cpu_count()})")
        with Pool(processes=num_processes) as pool:
          # Use tqdm to wrap the iterable returned by pool.imap for the progressbar
          q_events = list(tqdm(pool.imap(processNeutrons, events), total=len(events)))

        q_events_calc_detector = [item for sublist in q_events for item in sublist]

      np.savez_compressed(savename, q_events_calc_detector=q_events_calc_detector)
      print(f"Created {savename}.npz")

    else: #old, non-vectorised processing, resulting in multiple q values with different definitions
      from oldProcessing import processNeutronsNonVectorised
      out_events, q_events_real, q_events_no_incident_info, q_events_calc_sample, q_events_calc_detector = processNeutronsNonVectorised(events, alpha_inc, xwidth, yheight, ANGLE_RANGE, bins, wavelengthSelected, silicaRadius, get_simulation, sample_detector_distance, qConvFactorFromTofAtDetector, sim_module_name)
      np.savez_compressed(f"{savename}_q_events_real", q_events_real=q_events_real)
      np.savez_compressed(f"{savename}_q_events_no_incident_info", q_events_no_incident_info=q_events_no_incident_info)
      np.savez_compressed(f"{savename}_q_events_calc_sample", q_events_calc_sample=q_events_calc_sample)
      np.savez_compressed(f"{savename}_q_events_calc_detector", q_events_calc_detector=q_events_calc_detector)
      saveOutgoingEvents = False
      if saveOutgoingEvents:
        from inputOutput import write_events_mcpl
        deweight = False #Ensure final weight of 1 using splitting and Russian Roulette
        write_events_mcpl(out_events, savename, deweight)

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description = 'Execute BornAgain simulation of a GISANS sample with incident neutrons taken from an input file. The output of the script is a .npz file (or files) containing the derived Q values for each outgoing neutron. The default Q value calculated is aiming to be as close as possible to the Q value from a measurement.')
  parser.add_argument('filename',  help = 'Input filename. (Preferably MCPL file from the McStas MCPL_output component, but .dat file from McStas Virtual_output works as well)')
  parser.add_argument('-s', '--savename', default='', required=False, help = 'Output filename (can be full path).')
  parser.add_argument('--all_q', default=False, action='store_true', help = 'Calculate and save multiple Q values, each with different level of approximation (from real Q calculated from all simulation parameters to the default output value, that is Q calculated at the detector surface). This results in significantly slower simulations (especially due to the lack of parallelisation), but can shed light on the effect of e.g. divergence and TOF to lambda conversion on the derived Q value, in order to gain confidence in the results.')
  parser.add_argument('--no_parallel', default=False, action='store_true', help = 'Do not use multiprocessing. This makes the simulation significantly slower, but enables profiling, and the output of the number of neutrons missing the sample.')
  parser.add_argument('-p','--parallel_processes', required=False, type=int, help = 'Number of processes to be used for parallel processing.')
  parser.add_argument('-b','--detector_bins', default=10, type=int, help = 'Number of pixels in x and y direction of the "detector".')
  parser.add_argument('-m','--model', default=defaultSampleModel, help = 'BornAgain model to be used.')
  parser.add_argument('-r', '--silicaRadius', default=53, type=float, help = 'Silica particle radius for the "Silica particles on Silicon measured in air" sample model.')
  parser.add_argument('-i','--instrument', required=True, choices=list(instrumentParameters.keys()), help = 'Instrument.')
  parser.add_argument('-w','--wavelengthSelected', default=6.0, type=float, help = 'Wavelength (mean) in Angstrom selected by the velocity selector. Only used for non-time-of-flight instruments.')
  parser.add_argument('--tof_min', default=0, type=float, help = 'Lower TOF limit in microseconds for selecting neutrons from the input MCPL file.')
  parser.add_argument('--tof_max', default=150, type=float, help = 'Upper TOF limit in microseconds for selecting neutrons from the input MCPL file')
  parser.add_argument('-a', '--alpha', default=0.24, type=float, help = 'Incident angle on the sample. (Could be thought of as a sample rotation, but it is actually achieved by an an incident beam coordinate transformations.)')
  args = parser.parse_args()

  # Definitions here are global, so they will be accessible for non-parallel processing simulation
  nominal_source_sample_distance = instrumentParameters[args.instrument]['nominal_source_sample_distance']
  sample_detector_distance = instrumentParameters[args.instrument]['sample_detector_distance']
  sim_module_name = args.model
  silicaRadius = args.silicaRadius
  bins = args.detector_bins
  wavelengthSelected = None if instrumentParameters[args.instrument]['tof instrument'] else args.wavelengthSelected
  
  shm = createSharedMemory(nominal_source_sample_distance, sample_detector_distance, sim_module_name, silicaRadius, bins, wavelengthSelected)
  try:
    main(args)
  finally:
    shm.close()
    shm.unlink()
