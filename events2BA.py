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

def coordTransformToSampleSystem(events, alpha_inc):
  """Apply coordinate transformation to express neutron parameters in a
  coordinate system with the sample in the centre and being horisontal"""
  rot_matrix_inverse = np.array([[np.cos(-alpha_inc), -np.sin(-alpha_inc)],[np.sin(-alpha_inc), np.cos(-alpha_inc)]])
  p, x, y, z, vx, vy, vz, t = events.T
  zRot, yRot = np.dot(rot_matrix_inverse, [z, y])
  vzRot, vyRot = np.dot(rot_matrix_inverse, [vz, vy])
  return np.vstack([p, x, yRot, zRot, vx, vyRot, vzRot, t]).T

def propagateToSampleSurface(events, sample_xwidth, sample_yheight):
  """
  Propagate neutron events to z=0, the sample surface.
  Discard those which avoid the sample.
  """
  p, x, y, z, vx, vy, vz, t = events.T
  t_propagate = -z/vz
  x += vx * t_propagate
  y += vy * t_propagate
  z += vz * t_propagate
  t += t_propagate

  # Create a boolean mask for neutrons to select those which hit the sample
  hitSampleMask = (abs(x) < sample_xwidth) & (abs(y) < sample_yheight)
  sampleHitEvents = np.vstack([p, x, y, z, vx, vy, vz, t]).T[hitSampleMask]

  eventNr = len(events)
  sampleHitEventNr = len(sampleHitEvents)
  if sampleHitEventNr != eventNr:
    print(f"    WARNING: {eventNr - sampleHitEventNr} out of {eventNr} neutrons avoided the sample!")
  return sampleHitEvents

def get_simulation(sample, pixelNr, angle_range, wavelength=6.0, alpha_i=0.2, p=1.0, Ry=0., Rz=0.):
  """
  Create a simulation with pixelNr pixels that cover an angular range of angle_range degrees.
  The Ry and Rz values are relative rotations of the detector within one pixel
  to finely define the outgoing direction of events.
  """
  beam = ba.Beam(p, wavelength*angstrom, alpha_i*deg)

  dRy = Ry*angle_range*deg/(pixelNr-1)
  dRz = Rz*angle_range*deg/(pixelNr-1)

  # Define detector
  detector = ba.SphericalDetector(pixelNr, -angle_range*deg+dRz, angle_range*deg+dRz,
                                  pixelNr, -angle_range*deg+dRy, angle_range*deg+dRy)

  return ba.ScatteringSimulation(beam, sample, detector)

def virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ, sample_detector_distance, alpha_inc):
  """Calculate x,y,z position on the detector surface and the corresponding tof for the sample to detector propagation"""
  rot_matrix = np.array([[np.cos(alpha_inc), -np.sin(alpha_inc)],[np.sin(alpha_inc), np.cos(alpha_inc)]])
  #Calculate time until the detector surface with coord system rotation
  _, yRot = np.dot(rot_matrix, [z, y])
  _, vyRot = np.matmul(rot_matrix, np.vstack((VZ, VY))) # get [vz, vy]

  #propagate to detector surface perpendicular to the y-axis
  t_propagate = (sample_detector_distance - yRot) / vyRot

  return t_propagate, (VX * t_propagate + x), (VY * t_propagate + y), (VZ * t_propagate + z)

def getQsAtDetector(x, y, z, t, alpha_inc, VX, VY, VZ, nominal_source_sample_distance, sample_detector_distance, notTOFInstrument, qConvFactorFixed):
  sample_detector_tof, xDet, yDet, zDet = virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ, sample_detector_distance, alpha_inc)
  posDetector = np.vstack((xDet, yDet, zDet)).T
  sample_detector_path_length = np.linalg.norm(posDetector, axis=1)

  v_out_det = posDetector / sample_detector_path_length[:, np.newaxis]

  tDet = sample_detector_tof + t
  v_in_alpha = np.array([0, np.cos(alpha_inc), np.sin(alpha_inc)])
  if notTOFInstrument is False: #TOF instruments
    path_length = sample_detector_path_length + nominal_source_sample_distance
    qConvFactorFromTofAtDetector = 2*np.pi/(tofToLambda(tDet, path_length)*0.1)
    qArray = (v_out_det - v_in_alpha) * qConvFactorFromTofAtDetector[:, np.newaxis]
  else:
    qArray = (v_out_det - v_in_alpha) * qConvFactorFixed

  return qArray

def processNeutrons(neutron, sc=None):
  if sc is None:
    sc = getSharedMemoryValues() #get shared constants from shared memory

  sim_module = import_module(sc['sim_module_name'])
  get_sample = sim_module.get_sample
  sample = get_sample(radius=sc['silicaRadius'])

  notTOFInstrument = sc['wavelengthSelected'] is not None
  qConvFactorFixed = None if sc['wavelengthSelected'] is None else 2*np.pi/(sc['wavelengthSelected']*0.1)

  p, x, y, z, vx, vy, vz, t = neutron
  alpha_i = np.arctan(vz/vy)*180./np.pi  # deg
  phi_i = np.arctan(vx/vy)*180./np.pi  # deg
  v = np.sqrt(vx**2+vy**2+vz**2)
  wavelength = V2L/v  # Å


  # calculate pixelNr outgoing beams with a random angle within one pixel range
  Ry = 2*np.random.random()-1
  Rz = 2*np.random.random()-1
  sim = get_simulation(sample, sc['pixelNr'], sc['angle_range'], wavelength, alpha_i, p, Ry, Rz)
  sim.options().setUseAvgMaterials(True)
  sim.options().setIncludeSpecular(True)

  res = sim.simulate()
  # get probability (intensity) for all pixels
  pout = res.array()
  # calculate beam angle relative to coordinate system, including incident beam direction
  alpha_f = sc['angle_range']*(np.linspace(1., -1., sc['pixelNr'])+Ry/(sc['pixelNr']-1))
  phi_f = phi_i+sc['angle_range']*(np.linspace(-1., 1., sc['pixelNr'])+Rz/(sc['pixelNr']-1))
  alpha_f_rad = alpha_f * np.pi/180.
  phi_f_rad = phi_f * np.pi/180.
  alpha_grid, phi_grid = np.meshgrid(alpha_f_rad, phi_f_rad)

  VX_grid = v * np.cos(alpha_grid) * np.sin(phi_grid)
  VY_grid = v * np.cos(alpha_grid) * np.cos(phi_grid)
  VZ_grid = -v * np.sin(alpha_grid)

  qArray = getQsAtDetector(x, y, z, t, sc['alpha_inc'], VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten(), sc['nominal_source_sample_distance'], sc['sample_detector_distance'], notTOFInstrument, qConvFactorFixed)

  return  np.column_stack([pout.T.flatten(), qArray])

def main(args):
  #Constant values necessary for neutron processing, that are stored in shared memory if parallel processing is used
  sharedConstants = {
    'nominal_source_sample_distance': instrumentParameters[args.instrument]['nominal_source_sample_distance'],
    'sample_detector_distance': instrumentParameters[args.instrument]['sample_detector_distance'],
    'sim_module_name': args.model,
    'silicaRadius': args.silicaRadius,
    'pixelNr': args.pixel_number,
    'wavelengthSelected':  None if instrumentParameters[args.instrument]['tof instrument'] else args.wavelengthSelected,
    'alpha_inc': args.alpha *np.pi/180,
    'angle_range': args.angle_range
  }

  from inputOutput import getNeutronEvents
  events = getNeutronEvents(args.filename, args.tof_min, args.tof_max)
  events = coordTransformToSampleSystem(events, sharedConstants['alpha_inc'])
  events = propagateToSampleSurface(events, args.sample_xwidth, args.sample_yheight)

  savename = f"q_events_pix{sharedConstants['pixelNr']}" if args.savename == '' else args.savename
  if not args.all_q:
    if args.no_parallel: #not using parallel processing, iterating over each neutron sequentially
      total=len(events)
      q_events_calc_detector = []
      for in_ID, neutron in enumerate(events):
        if in_ID%200==0:
          print(f'{in_ID:10}/{total}')
        tmp = processNeutrons(neutron, sharedConstants)
        q_events_calc_detector.extend(tmp)
    else:
      print('Number of events being processed: ', len(events))
      num_processes = args.parallel_processes if args.parallel_processes else (cpu_count() - 2)
      print(f"Number of parallel processes: {num_processes} (number of CPU cores: {cpu_count()})")

      try:
        shm = createSharedMemory(sharedConstants) #using shared memory to pass in constants for the parallel processes
        with Pool(processes=num_processes) as pool:
          # Use tqdm to wrap the iterable returned by pool.imap for the progressbar
          q_events = list(tqdm(pool.imap(processNeutrons, events), total=len(events)))
      finally:
        shm.close()
        shm.unlink()

      q_events_calc_detector = [item for sublist in q_events for item in sublist]

    np.savez_compressed(savename, q_events_calc_detector=q_events_calc_detector)
    print(f"Created {savename}.npz")

  else: #old, non-vectorised, non-parallel processing, resulting in multiple q values with different definitions
    from oldProcessing import processNeutronsNonVectorised
    out_events, q_events_real, q_events_no_incident_info, q_events_calc_sample, q_events_calc_detector = processNeutronsNonVectorised(events, get_simulation, sharedConstants)
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
  parser.add_argument('-n','--pixel_number', default=10, type=int, help = 'Number of pixels in x and y direction of the "detector".')
  parser.add_argument('-m','--model', default=defaultSampleModel, help = 'BornAgain model to be used.')
  parser.add_argument('-r', '--silicaRadius', default=53, type=float, help = 'Silica particle radius for the "Silica particles on Silicon measured in air" sample model.')
  parser.add_argument('-i','--instrument', required=True, choices=list(instrumentParameters.keys()), help = 'Instrument.')
  parser.add_argument('-w','--wavelengthSelected', default=6.0, type=float, help = 'Wavelength (mean) in Angstrom selected by the velocity selector. Only used for non-time-of-flight instruments.')
  parser.add_argument('--tof_min', default=0, type=float, help = 'Lower TOF limit in microseconds for selecting neutrons from the input MCPL file.')
  parser.add_argument('--tof_max', default=150, type=float, help = 'Upper TOF limit in microseconds for selecting neutrons from the input MCPL file')
  parser.add_argument('-a', '--alpha', default=0.24, type=float, help = 'Incident angle on the sample. [deg] (Could be thought of as a sample rotation, but it is actually achieved by an an incident beam coordinate transformations.)')
  parser.add_argument('--angle_range', default=1.7, type=float, help = 'Scattering angle covered by the simulation. [deg]')
  parser.add_argument('--sample_xwidth', default=0.06, type=float, help = 'Size of sample perpendicular to beam. [m]')
  parser.add_argument('--sample_yheight', default=0.08, type=float, help = 'Size of sample along the beam. [m]')

  args = parser.parse_args()

  main(args)

