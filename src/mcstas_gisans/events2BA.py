#!/usr/bin/env python3

"""
Handles the simulation and processing of neutron scattering experiments.
It takes neutron events from an MCPL file, applies transformations, and
processes them through BornAgain simulations to generate Q values for each
neutron, and saves the result for further analysis or plotting.
"""

import numpy as np
from multiprocessing import cpu_count, Queue
import multiprocessing
from pathlib import Path

import bornagain as ba
from bornagain import deg, angstrom

from .neutron_utilities import velocityToWavelength, calcWavelength, qConvFactor
from .instruments import instrumentParameters, wfmRequiredKeys
from .input_output_utilities import getNeutronEvents, saveQHistogramFile, saveRawQListFile, printTofLimits
from .get_models import getSampleModels, getSampleModule 

def getTofFilteringLimits(args, pars):
  """
  Get TOF (time-of-flight) limits that can be used for filtering neutrons from
  the MCPL input file. The options are:
    1) No filtering ([-inf, inf])
    2) Using input values (args.input_tof_limits)
    3) Derive limits from a 1D TOF spectrum corresponding to a selected
       wavelength that is retrieved from a 2D McStas TOFLambda_monitor spectrum
       (that is assumed to represent the MCPL file content). The limits are
       defined by fitting a Gaussian function and getting a single FWHM range
       centred around the mean TOF value.
  """
  tofLimits = [float('-inf'), float('inf')]
  if pars['tof_instrument'] and not args.no_mcpl_filtering and (args.input_tof_limits or args.wavelength):
    if args.input_tof_limits:
      tofLimits = args.input_tof_limits
    else:
      if args.figure_output == 'png' or args.figure_output == 'pdf':
        figureOutput = f"{args.savename}.{args.figure_output}"
      else:
        figureOutput = args.figure_output # None or 'show'
      mcstasDir = Path(args.filename).resolve().parent
      from .mcstasMonitorFitting import fitGaussianToMcstasMonitor
      fit = fitGaussianToMcstasMonitor(dirname=mcstasDir, monitor=pars['mcpl_monitor_name'], wavelength=args.wavelength, wavelength_rebin=args.input_wavelength_rebin, figureOutput=figureOutput, tofRangeFactor=args.input_tof_range_factor)
      tofLimits[0] = (fit['mean'] - fit['fwhm'] * 0.5 * args.input_tof_range_factor) * 1e-3
      tofLimits[1] = (fit['mean'] + fit['fwhm'] * 0.5 * args.input_tof_range_factor) * 1e-3
      if args.figure_output is not None:
      # Terminate the script execution because an 'only plotting' has been selected by the user
        import sys
        printTofLimits(tofLimits)
        sys.exit()
  return tofLimits

def coordTransformToSampleSystem(events, alpha_inc):
  """Apply coordinate transformation to express neutron parameters in a
  coordinate system with the sample in the centre and being horizontal.
  """
  rot_matrix_inverse = np.array([[np.cos(-alpha_inc), -np.sin(-alpha_inc)],[np.sin(-alpha_inc), np.cos(-alpha_inc)]])
  p, x, y, z, vx, vy, vz, t = events.T
  zRot, yRot = np.dot(rot_matrix_inverse, [z, y])
  vzRot, vyRot = np.dot(rot_matrix_inverse, [vz, vy])
  return np.vstack([p, x, yRot, zRot, vx, vyRot, vzRot, t]).T

def propagateToSampleSurface(events, sample_xwidth, sample_zheight, allow_sample_miss):
  """Propagate neutron events to y=0, the sample surface.
  Discard those neutrons which would miss the sample.
  """
  p, x, y, z, vx, vy, vz, t = events.T
  t_propagate = -y/vy # y+vy*t_propagate=0 (where y is the initial position)
  x += vx * t_propagate
  y += vy * t_propagate
  z += vz * t_propagate
  t += t_propagate

  # Create a boolean mask for neutrons to select those which hit the sample
  hitSampleMask = (abs(x) < sample_xwidth*0.5) & (abs(z) < sample_zheight*0.5)
  eventsOnSampleSurface = np.vstack([p, x, y, z, vx, vy, vz, t]).T if allow_sample_miss else np.vstack([p, x, y, z, vx, vy, vz, t]).T[hitSampleMask]

  eventNr = len(events)
  sampleHitEventNr = np.sum(hitSampleMask)
  if sampleHitEventNr != eventNr:
    sum_weight_in = sum(p)
    sum_weight_sample_hit = sum(p[hitSampleMask])
    print(f"    WARNING: {eventNr - sampleHitEventNr} out of {eventNr} incident neutrons missed the sample!({sum_weight_in-sum_weight_sample_hit} out of {sum_weight_in} in terms of sum particle weight)")
    if not allow_sample_miss:
      print(f"    WARNING: Incident neutrons missing the sample are not propagated to the detectors! This can be changed with the --allow_sample_miss option.") #TODO mention the option to allow them with the input parameter
  return eventsOnSampleSurface

def applyT0Correction(events, args):
  """Apply t0 TOF correction for all neutrons. A fixed t0correction value can be
  given to be subtracted, or a McStas TOFLambda monitor result with a selected
  wavelength is used, in which case t0correction is retrieved as the mean value
  from fitting a Gaussian function to the TOF spectrum of the wavelength bin
  including the selected wavelength. The fitting is done for the full TOF range
  unless the WFM mode is used, in which case it is done within the wavelength
  dependent subpulse TOF limits. Rebinning along the wavelength axis can be
  applied beforehand to improve the reliability of the fitting.
  WARNING: the TOF axis of the monitor is assumed to have microsecond units!
  """
  if args.t0_fixed: #T0 correction with fixed input value
    t0correction = args.t0_fixed
  else: #T0 correction based on McStas (TOFLambda) monitor
    if not args.wfm:
      tofLimits = [None, None] #Do not restrict the monitor TOF spectrum for T0 correction fitting
      t0monitor = instrumentParameters[args.instrument]['t0_monitor_name']
    else: # Wavelength Frame Multiplication (WFM)
      from .instruments import getSubpulseTofLimits
      tofLimits = getSubpulseTofLimits(args.wavelength)
      t0monitor = instrumentParameters[args.instrument]['wfm_t0_monitor_name']
    print(f"Applying T0 correction based on McStas monitor: {t0monitor}")
    mcstasDir = Path(args.filename).resolve().parent
    from .mcstasMonitorFitting import fitGaussianToMcstasMonitor
    fit = fitGaussianToMcstasMonitor(mcstasDir, t0monitor, args.wavelength, tofLimits=tofLimits, wavelength_rebin=args.t0_wavelength_rebin)
    t0correction = fit['mean'] * 1e-6
  print(f"T0 correction value: {t0correction} second")

  p, x, y, z, vx, vy, vz, t = events.T
  t -= t0correction
  events = np.vstack([p, x, y, z, vx, vy, vz, t]).T
  return events

def get_simulation(sample, pixelNr, angle_range, wavelength, alpha_i, p, Ry, Rz):
  """Create a simulation with pixelNr pixels that cover an angular range of angle_range degrees.
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

def virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ, sample_detector_distance, sampleToRealCoordRotMatrix, realToSampleCoordRotMatrix):
  """Calculate x,y,z position on the detector surface and the corresponding TOF for the sample to detector propagation"""
  #Calculate time until the detector surface with coord system rotation
  # NOTE: under the assumption that the detector surface is vertical in the real coord system
  zRot, _ = np.dot(sampleToRealCoordRotMatrix, [z, y])
  vzRot, _ = np.matmul(sampleToRealCoordRotMatrix, np.vstack((VZ, VY))) # get [vz, vy]

  #propagate to detector surface perpendicular to the y-axis
  t_propagate = (sample_detector_distance - zRot) / vzRot

  #Calculate the effect of gravity
  gravityAcceleration = 9.80665 #m/s2
  zGravityAcc, yGravityAcc = np.dot(realToSampleCoordRotMatrix, [0, -gravityAcceleration]) #TODO should not happen here
  zGravityDrop = zGravityAcc * 0.5 * t_propagate**2
  yGravityDrop = yGravityAcc * 0.5 * t_propagate**2

  return t_propagate, (VX * t_propagate + x), (VY * t_propagate + y + yGravityDrop), (VZ * t_propagate + z + zGravityDrop)

def getDetectionCoordinate(xDet, yDet, zDet, sampleToRealCoordRotMatrix, realToSampleCoordRotMatrix):
  """Get the coordinate of the detection event from the position where the path
  of the neutron intersects the plane of the detector surface.
  Using the exact position of intersection means infinite detector resolution.
  """
  #TODO generalise:
  #  pixel_size_x, pixel_size_y, sigma_x, sigma_y should be input

  detector_size_x = 1.024 #[m]
  detector_size_y = 1.024 #[m]
  detector_pixel_number_x = 256
  detector_pixel_number_y = 128
  pixel_size_x = detector_size_x / detector_pixel_number_x
  pixel_size_y = detector_size_y / detector_pixel_number_y

  # transform from the sample-based to the real coordinate system
  zDetReal, yDetReal = np.matmul(sampleToRealCoordRotMatrix, np.vstack((zDet, yDet)))
  #note: zDetReal is a fixed value due to the propagation to detector surface

  # apply gaussian randomisation
  sigma_x = 0.004/2.355 #FWHM=2.5-5[mm]
  sigma_y = 0 #0.005/2.355 #FWHM=2.5-5[mm]
  xDet = np.random.normal(xDet, sigma_x, size=xDet.shape)
  yDetReal = np.random.normal(yDetReal, sigma_y, size=yDetReal.shape)

  #get the coordinates of the centre of the pixel where the neutron is detected
  xDetCoord = np.floor(xDet / pixel_size_x) * pixel_size_x + 0.5*pixel_size_x
  yDetCoordReal = np.floor(yDetReal / pixel_size_y) * pixel_size_y + 0.5*pixel_size_y

  #transform from the real to the sample-based coordinate system
  zDetCoord, yDetCoord = np.matmul(realToSampleCoordRotMatrix, np.vstack((zDetReal, yDetCoordReal)))

  return xDetCoord, yDetCoord, zDetCoord


def getQsAtDetector(x, y, z, t, VX, VY, VZ, nominal_source_sample_distance, sample_detector_distance, notTOFInstrument, qConvFactorFixed, sampleToRealCoordRotMatrix, incidentDirection, realToSampleCoordRotMatrix):
  """Calculate Q values (x,y,z) from positions at the detector surface.
  All outgoing directions from the BornAgain simulation of a single neutron are
  handled at the same time using operations on vectors.
  - Outgoing direction is calculated by propagating neutrons to the detector surface,
  and assuming that the neutron is scattered at the centre of the sample (the origin).
  - Incident direction is an input value.
  - For non-TOF instruments the (2*pi/(wavelength)) factor (qConvFactorFixed) is an input value.
    For TOF instruments this factor is calculated from the TOF at the detector surface position
    and the nominal distance travelled by the neutron until that position.
  """
  sample_detector_tof, xDet, yDet, zDet = virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ, sample_detector_distance, sampleToRealCoordRotMatrix, realToSampleCoordRotMatrix)
  xDetCoord, yDetCoord, zDetCoord = getDetectionCoordinate(xDet, yDet, zDet, sampleToRealCoordRotMatrix, realToSampleCoordRotMatrix)
  posDetector = np.vstack((xDetCoord, yDetCoord, zDetCoord)).T
  sample_detector_path_length = np.linalg.norm(posDetector, axis=1)

  outgoingDirection = posDetector / sample_detector_path_length[:, np.newaxis]

  if notTOFInstrument is False: #TOF instruments
    wavelengthDet = calcWavelength(t+sample_detector_tof, nominal_source_sample_distance+sample_detector_path_length)
    qFactor = qConvFactor(wavelengthDet)[:, np.newaxis]
  else: #not TOF instruments
    qFactor = qConvFactorFixed

  return (outgoingDirection - incidentDirection) * qFactor

def processNeutrons(neutrons, params, queue=None):
  """Carry out the BornAgain simulation and subsequent calculations of each
  incident neutron (separately) in the input array.
  1) The BornAgain simulation for a certain sample model is set up with an array
     out outgoing directions
  2) The BornAgain simulation is performed, resulting in an array of outgoing
     beams with weights (outgoing probabilities)
  3) The Q values are calculated after a virtual propagation to the detector
     surface.
  4) Depending on the input options, the list of Q events (weight,qx,qy,qz) are
     either returned (old raw format), or histogrammed and added to a cumulative
     histogram where all other neutrons result are added.
  """
  sim_module = getSampleModule(params["sim_module_name"])
  sample = sim_module.get_sample(**params['sample_kwargs'])

  notTOFInstrument = params['wavelengthSelected'] is not None
  qConvFactorFixed = None if params['wavelengthSelected'] is None else qConvFactor(params['wavelengthSelected'])

  if params['raw_output']:
    q_events = [] #p, Qx, Qy, Qz, t
  else:
    qHist = np.zeros(tuple(params['bins']))
    qHistWeightsSquared = np.zeros(tuple(params['bins']))

  incidentDirection = np.array([0, -np.sin(params['alpha_inc']), np.cos(params['alpha_inc'])]) #for Q calculation

  ## Carry out BornAgain simulation for all incident neutron one-by-one
  for id, neutron in enumerate(neutrons):
    if id%200==0:
      print(f'{id:10}/{len(neutrons)}') #print output to indicate progress
    # Neutron positions, velocities and corresponding calculations are expressed
    # in the McStas coord system (X - left; Y - up; Z - forward 'along the beam')
    # not in the BornAgain coord system (X - forward, Y - left, Z - up),
    # but with the SphericalDetector, BornAgain only deals with alpha_i (input),
    # alpha_f and phi_f (output), which are the same if calculated correctly
    p, x, y, z, vx, vy, vz, t = neutron
    alpha_i = np.rad2deg(np.arctan(-vy/vz)) #[deg]
    phi_i = np.rad2deg(np.arctan(vx/vz)) #[deg], not used in sim, added to phi_f
    v = np.sqrt(vx**2+vy**2+vz**2)
    wavelength = velocityToWavelength(v) #angstrom

    if (abs(x) > params['sample_xwidth']*0.5) or (abs(z) > params['sample_zheight']*0.5):
      #direct propagation of neutrons missing the sample
      qArray = getQsAtDetector(x, y, z, t, [vx], [vy], [vz], params['nominal_source_sample_distance'], params['sample_detector_distance'], notTOFInstrument, qConvFactorFixed, params['sampleToRealCoordRotMatrix'], incidentDirection, params['realToSampleCoordRotMatrix'])
      weights = np.array([p])
    else:
      # calculate (pixelNr)^2 outgoing beams with a random angle within one pixel range
      Ry = 2*np.random.random()-1
      Rz = 2*np.random.random()-1
      sim = get_simulation(sample, params['pixelNr'], params['angle_range'], wavelength, alpha_i, p, Ry, Rz)
      sim.options().setUseAvgMaterials(True)
      sim.options().setIncludeSpecular(True)
      # sim.options().setNumberOfThreads(n) ##Experiment with this? If not parallel processing?

      res = sim.simulate()
      # get probability (intensity) for all pixels
      pout = res.array()
      # calculate beam angle relative to coordinate system, including incident beam direction
      alpha_f = params['angle_range']*(np.linspace(1., -1., params['pixelNr'])+Ry/(params['pixelNr']-1))
      phi_f = phi_i+params['angle_range']*(np.linspace(-1., 1., params['pixelNr'])+Rz/(params['pixelNr']-1))
      alpha_grid, phi_grid = np.meshgrid(np.deg2rad(alpha_f), np.deg2rad(phi_f))

      VX_grid = v * np.cos(alpha_grid) * np.sin(phi_grid) #this is Y in BA coord system) (horizontal - to the left)
      VY_grid = v * np.sin(alpha_grid)                    #this is Z in BA coord system) (horizontal - up)
      VZ_grid = v * np.cos(alpha_grid) * np.cos(phi_grid) #this is X in BA coord system) (horizontal - forward)

      qArray = getQsAtDetector(x, y, z, t, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten(), params['nominal_source_sample_distance'], params['sample_detector_distance'], notTOFInstrument, qConvFactorFixed, params['sampleToRealCoordRotMatrix'], incidentDirection, params['realToSampleCoordRotMatrix'])
      weights = pout.T.flatten()
    if params['raw_output']:
      q_events.append(np.column_stack([weights, qArray]))
    else: #histogrammed output format
      qHistOfNeutron, _ = np.histogramdd(qArray, weights=weights, bins=params['bins'], range=params['histRanges'])
      qHistWeightsSquaredOfNeutron, _ = np.histogramdd(qArray, weights=weights**2, bins=params['bins'], range=params['histRanges'])
      qHist += qHistOfNeutron
      qHistWeightsSquared += qHistWeightsSquaredOfNeutron

  if params['raw_output']:
    result = [item for sublist in q_events for item in sublist] #flatten sublists
  else:
    result = {'qHist': qHist, 'qHistWeightsSquared': qHistWeightsSquared}

  if queue: #return result from multiprocessing process
    queue.put(result)
  else:
    return result

def processNeutronsInParallel(events, params, processNumber):
  """Spawn parallel processes to carry out the BornAgain simulation and
  subsequent calculation of the neutrons.
  """
  print(f"Number of parallel processes: {processNumber} (number of CPU cores: {cpu_count()})")

  processes = []
  results = []
  queue = Queue() #a queue to get results from each process

  eventNumber = len(events)
  chunkSize = eventNumber // processNumber
  def getEventsChunk(processIndex):
    """Distribute the events array among the processes as evenly as possible"""
    start = processIndex * chunkSize
    end = (processIndex + 1) * chunkSize if processIndex < processNumber - 1 else eventNumber
    return events[start:end]

  for i in range(processNumber):
    p = multiprocessing.Process(target=processNeutrons, args=(getEventsChunk(i), params, queue,))
    processes.append(p)
    p.start()

  for p in processes: # get the results from each process
    results.append(queue.get())
  for p in processes: # Wait for all processes to finish
    p.join()

  if len(results) != len(processes):
      print(f"Warning: Expected {len(processes)} results, but received {len(results)}. Some processes may not have completed.")

  if params['raw_output']: #merge lists of raw Q events of the processes
    result = [item for sublist in results for item in sublist]
  else: #merge the histogram results of the processes
    qHist = np.zeros(tuple(params['bins']))
    qHistWeightsSquared = np.zeros(tuple(params['bins']))
    for processResult in results:
      qHist += processResult['qHist']
      qHistWeightsSquared += processResult['qHistWeightsSquared']
    result = {'qHist': qHist, 'qHistWeightsSquared': qHistWeightsSquared}
  return result

def parse_sample_arguments(args):
  """Parse the --sample_arguments string into keyword arguments."""
  if not args.sample_arguments:
    return {}

  def convert_numbers(value):
    """Attempt to convert strings to integers or floats"""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

  kwargs = {}
  pairs = args.sample_arguments.split(';')
  for pair in pairs:
    key, value = pair.split('=')
    kwargs[key.strip()] = convert_numbers(value.strip())
  return kwargs

def createParamsDict(args, instParams):
  """Pack parameters necessary for processing in a single dictionary"""
  beamDeclination = 0 if not 'beam_declination_angle' in instParams else instParams['beam_declination_angle']
  sampleInclination = float(np.deg2rad(args.alpha - beamDeclination))
  return {
    'nominal_source_sample_distance': instParams['nominal_source_sample_distance'] - (0 if not args.wfm else instParams['wfm_virtual_source_distance']),
    'sample_detector_distance': instParams['sample_detector_distance'],
    'sampleToRealCoordRotMatrix' : np.array([[np.cos(sampleInclination), -np.sin(sampleInclination)],
                                             [np.sin(sampleInclination), np.cos(sampleInclination)]]),
    'realToSampleCoordRotMatrix' : np.array([[np.cos(-sampleInclination), -np.sin(-sampleInclination)],
                                             [np.sin(-sampleInclination), np.cos(-sampleInclination)]]),
    'sim_module_name': args.model,
    'pixelNr': args.pixel_number,
    'wavelengthSelected':  None if instParams['tof_instrument'] else args.wavelengthSelected,
    'alpha_inc': float(np.deg2rad(args.alpha)),
    'angle_range': args.angle_range,
    'raw_output': args.raw_output,
    'bins': args.bins,
    'histRanges': [args.x_range, args.y_range, args.z_range],
    'sample_xwidth': args.sample_xwidth,
    'sample_zheight': args.sample_zheight,
    'sample_kwargs': parse_sample_arguments(args),
  }

def main():
  import argparse
  parser = argparse.ArgumentParser(description = 'Execute BornAgain simulation of a GISANS sample with incident neutrons taken from an input file. The output of the script is a .npz file (or files) containing the derived Q values for each outgoing neutron. The default Q value calculated is aiming to be as close as possible to the Q value from a measurement.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('filename',  help = 'Input filename. (Preferably MCPL file from the McStas MCPL_output component, but .dat file from McStas Virtual_output works as well)')
  parser.add_argument('--intensity_factor', default=1.0, type=float, help = 'A multiplication factor to modify the beam intensity. (Applied to the Monte Carlo weight of each particle in the input file.)')
  parser.add_argument('-i','--instrument', required=True, choices=list(instrumentParameters.keys()), help = 'Instrument (from instruments.py).')
  parser.add_argument('-p','--parallel_processes', required=False, type=int, help = 'Number of processes to be used for parallel processing.')
  parser.add_argument('--no_parallel', default=False, action='store_true', help = 'Do not use multiprocessing. This makes the simulation significantly slower, but enables profiling. Uses --raw_output implicitly.')
  parser.add_argument('-n','--pixel_number', default=10, type=int, help = 'Number of pixels in x and y direction of the "detector".')
  parser.add_argument('--wavelengthSelected', default=6.0, type=float, help = 'Wavelength (mean) in Angstrom selected by the velocity selector. Only used for non-time-of-flight instruments.')
  parser.add_argument('--angle_range', default=1.7, type=float, help = 'Scattering angle covered by the simulation. [deg]')

  outputGroup = parser.add_argument_group('Output', 'Control the generated outputs. By default a histogram (and corresponding uncertainty) is generated as an output, saved in a npz file, loadable with the plotQ script.')
  outputGroup.add_argument('-s', '--savename', default='', required=False, help = 'Output filename (can be full path).')
  outputGroup.add_argument('--raw_output', default=False, action='store_true', help = 'Create a raw list of Q events as output instead of the default histogrammed data. Warning: this option may require too much memory for high incident event and pixel numbers.')
  outputGroup.add_argument('--bins', nargs=3, type=int, default=[256, 128, 1], help='Number of histogram bins in x,y,z directions.')
  outputGroup.add_argument('--x_range', nargs=2, type=float, default=[-0.55, 0.55], help='Qx range of the histogram. (In horizontal plane right to left)')
  outputGroup.add_argument('--y_range', nargs=2, type=float, default=[-0.5, 0.6], help='Qy range of the histogram. (In vertical plane bottom to top)')
  outputGroup.add_argument('--z_range', nargs=2, type=float, default=[-1000, 1000], help='Qz range of the histogram. (In horizontal plane back to forward)')
  outputGroup.add_argument('--quick_plot', default=False, action='store_true', help='Show a quick Qx-Qz plot from the histogram result.')
  outputGroup.add_argument('--all_q', default=False, action='store_true', help = 'Calculate and save multiple Q values, each with different levels of approximation (from real Q calculated from all simulation parameters to the default output value, that is Q calculated at the detector surface). This results in significantly slower simulations (especially due to the lack of parallelisation), but can shed light on the effect of e.g. divergence and TOF to lambda conversion on the derived Q value, in order to gain confidence in the results.')

  sampleGroup = parser.add_argument_group('Sample', 'Sample related parameters and options.')
  sampleGroup.add_argument('-a', '--alpha', default=0.24, type=float, help = 'Incident angle on the sample. [deg] (Could be thought of as a sample rotation, but it is actually achieved by an incident beam coordinate transformation.)')
  sampleGroup.add_argument('--model', default="silica_100nm_air", choices=getSampleModels(), help = 'BornAgain model to be used.')
  sampleGroup.add_argument('--sample_arguments', help = 'Input arguments of the sample model in format: "arg1=value1;arg2=value2"')
  sampleGroup.add_argument('--sample_xwidth', default=0.06, type=float, help = 'Size of sample perpendicular to beam. [m]')
  sampleGroup.add_argument('--sample_zheight', default=0.08, type=float, help = 'Size of sample along the beam. [m]')
  sampleGroup.add_argument('--allow_sample_miss', default=False, action='store_true', help = 'Allow incident neutrons to miss the sample, and being directly propagated to the detector surface. This option can be used to simulate overillumination, or direct beam simulation by also setting one of the sample sizes to zero.')

  mcplFilteringGroup = parser.add_argument_group('MCPL filtering', 'Parameters and options to control which neutrons are used from the MCPL input file. By default no filtering is applied, but if a (central) wavelength is provided, an accepted TOF range is defined based on a McStas TOFLambda monitor (defined as mcpl_monitor_name for each instrument in instruments.py) that is assumed to correspond to the input MCPL file. The McStas monitor is looked for in the directory of the MCPL input file, and after fitting a Gaussian function, neutrons within a single FWHM range centred around the selected wavelength are used for the BornAgain simulation.')
  mcplFilteringGroup.add_argument('-w', '--wavelength', type=float, default=None, help = 'Central wavelength used for filtering based on the McStas TOFLambda monitor. (Also used for t0 correction.)')
  mcplFilteringGroup.add_argument('--input_tof_range_factor', default=1.0, type=float, help = 'Modify the accepted TOF range of neutrons by this multiplication factor.')
  mcplFilteringGroup.add_argument('--input_wavelength_rebin', default=1, type=int, help = 'Rebin the TOFLambda monitor along the wavelength axis by the provided factor (only if no extrapolation is needed).')
  mcplFilteringGroup.add_argument('--input_tof_limits', nargs=2, type=float, help = 'TOF limits for selecting neutrons from the MCPL file [millisecond]. When provided, fitting to the McStas monitor is not attempted.')
  mcplFilteringGroup.add_argument('--no_mcpl_filtering', action='store_true', help = 'Disable MCPL TOF filtering. Use all neutrons from the MCPL input file.')
  mcplFilteringGroup.add_argument('--figure_output', default=None, choices=['show', 'png', 'pdf'], help = 'Show or save the figure of the selected input TOF range and exit without doing the simulation. Only works with McStas monitor fitting.')

  t0correctionGroup = parser.add_argument_group('T0 correction', 'Parameters and options to control t0 TOF correction. Currently only works if the wavelength parameter in the MCPL filtering is provided.')
  t0correctionGroup.add_argument('--t0_fixed', default=None, type=float, help = 'Fix t0 correction value that is subtracted from the neutron TOFs. [s]')
  t0correctionGroup.add_argument('--t0_wavelength_rebin', default=None, type=int, help = 'Rebinning factor for the McStas TOFLambda monitor based t0 correction. Rebinning is applied along the wavelength axis. Only integer divisors are allowed.')
  t0correctionGroup.add_argument('--wfm', default=False, action='store_true', help = 'Wavelength Frame Multiplication (WFM) mode.')
  t0correctionGroup.add_argument('--no_t0_correction', action='store_true', help = 'Disable t0 correction. (Allows using McStas simulations which lack the supported monitors.)')

  args = parser.parse_args()

  if args.wfm and any(key not in instrumentParameters[args.instrument] for key in wfmRequiredKeys):
    parser.error(f"wfm option is not enabled for the {args.instrument} instrument! Set the required instrument parameters in instruments.py!")

  if args.figure_output:
    if not args.wavelength:
      parser.error(f"The --figure_output option can only be used if a central wavelength (--wavelength) for fitting is provided.")
    if args.input_tof_limits:
      parser.error(f"The --figure_output option can not be used when the TOF range is provided with --input_tof_limits.")
    if args.no_mcpl_filtering:
      parser.error(f"The --figure_output option can not be used when no TOF filtering is selected with --no_mcpl_filtering.")

  if args.no_t0_correction:
    if args.t0_fixed:
      parser.error(f"The --no_t0_correction option can not be used together with --t0_fixed ")
    if args.t0_wavelength_rebin:
      parser.error(f"The --no_t0_correction option can not be used together with --t0_wavelength_rebin ")
    if args.wfm:
      parser.error(f"The --no_t0_correction option can not be used together with --wfm ")
  else:
    if not args.wavelength:
      parser.error(f"The --wavelength must be provided for T0 correction. Alternatively, the --no_t0_correction option can be used to skip T0 correction.")

  if not args.no_mcpl_filtering:
    if not args.wavelength:
      parser.error(f"The --wavelength must be provided for MCPL TOF filtering. Alternatively, the --no_mcpl_filtering option can be used to skip TOF filtering")

  if args.t0_fixed:
    if args.t0_wavelength_rebin:
      parser.error(f"The --t0_fixed option can not be used together with --t0_wavelength_rebin ")

  if args.all_q and args.wfm:
    parser.error(f"The --all_q option can not be used together with --wfm. (Simply because --all_q is not well maintained, but it can be changed on demand.)")

  if (args.sample_xwidth == 0 or args.sample_zheight == 0) and not args.allow_sample_miss:
    parser.error(f"One of the sample sizes is zero. Direct beam simulation also requires the --allow_sample_miss option to be set True.")
  if (args.sample_xwidth < 0 or args.sample_zheight < 0):
    parser.error(f"The sample sizes can not be negative. (For direct beam simulation, set either of the sample sizes to zero.)")

  if args.sample_arguments:
    pairs = args.sample_arguments.split(';')
    for pair in pairs:
        if '=' not in pair:
            parser.error(f"Invalid argument format for --sample_arguments: {pair}. Should be arg=value")

  if args.intensity_factor <= 0.0:
    parser.error(f"The intensity multiplication factor (--intensity_factor) must have a positive value.")


  """The actual script execution after the input arguments are parsed."""
  #parameters necessary for neutron processing
  params = createParamsDict(args, instrumentParameters[args.instrument])

  ### Getting neutron events from the MCPL file ###
  mcplTofLimits = getTofFilteringLimits(args, instrumentParameters[args.instrument])
  events = getNeutronEvents(args.filename, mcplTofLimits, args.intensity_factor)

  ### Preconditioning ###
  events = coordTransformToSampleSystem(events, params['alpha_inc'])
  events = propagateToSampleSurface(events, args.sample_xwidth, args.sample_zheight, args.allow_sample_miss)
  if args.no_t0_correction or not instrumentParameters[args.instrument]['tof_instrument']:
    print("No T0 correction is applied.")
  else:
    events = applyT0Correction(events, args)

  ### BornAgain simulation ###
  savename = f"q_events_pix{params['pixelNr']}" if args.savename == '' else args.savename
  print('Number of events being processed: ', len(events))

  if args.all_q: #old, non-vectorised, non-parallel processing, resulting in multiple q values with different definitions
    from .old_processing import processNeutronsNonVectorised
    processNeutronsNonVectorised(events, get_simulation, params, savename)
    return #early return

  if args.no_parallel: #not using parallel processing, iterating over each neutron sequentially, mainly intended for profiling
    result = processNeutrons(events, params)
  else:
    processNumber = args.parallel_processes if args.parallel_processes else (cpu_count() - 2)
    result = processNeutronsInParallel(events, params, processNumber)

  ### Create Output ###
  if args.raw_output: #raw list of Q events (old output)
    qArray = result
    saveRawQListFile(savename, qArray)
    return # no further processing, early return

  ## Create Q histogram with corresponding uncertainty array (new output format)
  qHist = result['qHist']
  qHistWeightsSquared = result['qHistWeightsSquared']
  qHistError = np.sqrt(qHistWeightsSquared)

  #Get the bin edges of the histograms
  edges = [np.array(np.histogram_bin_edges(None, bins=b, range=r), dtype=np.float64)
               for b, r in zip(params['bins'], params['histRanges'])]

  saveQHistogramFile(savename, qHist, qHistError, edges)

  if args.quick_plot:
    hist2D = np.sum(qHist, axis=2)
    from .plotting_utilities import logPlot2d
    logPlot2d(hist2D, edges[0], edges[1], xRange=params['histRanges'][0], yRange=params['histRanges'][1], output='show')

if __name__=='__main__':
  main()
