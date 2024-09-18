#!/usr/bin/env python3

"""
Handles the simulation and processing of neutron scattering experiments.
It takes neutron events from an MCPL file, applies transformations, and
processes them through BornAgain simulations to generate Q values for each
neutron, and saves the result for further analysis or plotting.
"""

from importlib import import_module
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path

import bornagain as ba
from bornagain import deg, angstrom

from neutron_utilities import velocityToWavelength, calcWavelength, qConvFactor
from instruments import instrumentParameters, wfmRequiredKeys
from sharedMemory import sharedMemoryHandler
from mcstasMonitorFitting import fitGaussianToMcstasMonitor

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
      fit = fitGaussianToMcstasMonitor(dirname=mcstasDir, monitor=pars['mcpl_monitor_name'], wavelength=args.wavelength, wavelength_rebin=args.input_wavelength_rebin, figureOutput=figureOutput, tofRangeFactor=args.input_tof_range_factor)
      tofLimits[0] = (fit['mean'] - fit['fwhm'] * 0.5 * args.input_tof_range_factor) * 1e-3
      tofLimits[1] = (fit['mean'] + fit['fwhm'] * 0.5 * args.input_tof_range_factor) * 1e-3
      if args.figure_output is not None:
      # Terminate the script execution because an 'only plotting' has been selected by the user
        from inputOutput import printTofLimits
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

def propagateToSampleSurface(events, sample_xwidth, sample_yheight):
  """Propagate neutron events to z=0, the sample surface.
  Discard those neutrons which would miss the sample.
  """
  p, x, y, z, vx, vy, vz, t = events.T
  t_propagate = -z/vz #negative sign because z axis is pointing down (toward the sample) in the Rotated McStas coord system used for MCPL output
  x += vx * t_propagate
  y += vy * t_propagate
  z += vz * t_propagate
  t += t_propagate

  # Create a boolean mask for neutrons to select those which hit the sample
  hitSampleMask = (abs(x) < sample_xwidth*0.5) & (abs(y) < sample_yheight*0.5)
  sampleHitEvents = np.vstack([p, x, y, z, vx, vy, vz, t]).T[hitSampleMask]

  eventNr = len(events)
  sampleHitEventNr = len(sampleHitEvents)
  if sampleHitEventNr != eventNr:
    print(f"    WARNING: {eventNr - sampleHitEventNr} out of {eventNr} neutrons avoided the sample!")
  return sampleHitEvents

def applyT0Correction(events, args): #TODO update docstring
  """Apply t0 TOF correction for all neutrons unless explicitly asked not to.
  A fixed t0correction value can be given to be subtracted, or a McStas TOFLambda
  monitor result with a selected wavelength is used, in which case t0correction
  is retrieved as the mean value from fitting a Gaussian function to the TOF
  spectrum of the wavelength bin including the selected wavelength. The fitting
  is done for the full TOF range unless the WFM mode is used, in which case
  it is done within the wavelength dependent subpulse TOF limits. Rebinning along
  the wavelength axis can be applied beforehand to improve the reliability of
  the fitting.
  WARNING: the TOF axis of the monitor is assumed to have microsecond units!
  """
  if args.no_t0_correction:
    print("No T0 correction is applied.")
  else:
    if args.t0_fixed: #T0 correction with fixed input value
      t0correction = args.t0_fixed
    else: #T0 correction based on McStas (TOFLambda) monitor
      if not args.wfm:
        tofLimits = [None, None] #Do not restrict the monitor TOF spectrum for T0 correction fitting
        t0monitor = instrumentParameters[args.instrument]['t0_monitor_name']
      else: # Wavelength Frame Multiplication (WFM)
        from instruments import getSubpulseTofLimits
        tofLimits = getSubpulseTofLimits(args.wavelength)
        t0monitor = instrumentParameters[args.instrument]['wfm_t0_monitor_name']
      print(f"Applying T0 correction based on McStas monitor: {t0monitor}")
      mcstasDir = Path(args.filename).resolve().parent
      fit = fitGaussianToMcstasMonitor(mcstasDir, t0monitor, args.wavelength, tofLimits=tofLimits, wavelength_rebin=args.t0_wavelength_rebin)
      t0correction = fit['mean'] * 1e-6
    print(f"  T0 correction value: {t0correction} second")

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

def virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ, sample_detector_distance, alpha_inc):
  """Calculate x,y,z position on the detector surface and the corresponding TOF for the sample to detector propagation"""
  rot_matrix = np.array([[np.cos(alpha_inc), -np.sin(alpha_inc)],[np.sin(alpha_inc), np.cos(alpha_inc)]])
  #Calculate time until the detector surface with coord system rotation
  _, yRot = np.dot(rot_matrix, [z, y])
  _, vyRot = np.matmul(rot_matrix, np.vstack((VZ, VY))) # get [vz, vy]

  #propagate to detector surface perpendicular to the y-axis
  t_propagate = (sample_detector_distance - yRot) / vyRot

  return t_propagate, (VX * t_propagate + x), (VY * t_propagate + y), (VZ * t_propagate + z)

def getQsAtDetector(x, y, z, t, alpha_inc, VX, VY, VZ, nominal_source_sample_distance, sample_detector_distance, notTOFInstrument, qConvFactorFixed):
  """Calculate Q values (x,y,z) from positions at the detector surface.
  All outgoing directions from the BornAgain simulation of a single neutron are
  handled at the same time using operations on vectors.
  - Outgoing direction is calculated by propagating neutrons to the detector surface,
  and assuming that the neutron is scattered at the centre of the sample (the origin).
  - Incident direction is an input value.
  - For non-TOF instruments the (2*pi/(wavelength)) factor (qConvFactorFixed) is an input value.
    For TOF instruments this factor is calculated from the TOF at the detector surface position
    and the nominal distance travelled by the neutron until that position.

  Note that the current implementation doesn't take detector resolution into account (infinite resolution).
  """
  sample_detector_tof, xDet, yDet, zDet = virtualPropagationToDetectorVectorised(x, y, z, VX, VY, VZ, sample_detector_distance, alpha_inc)
  posDetector = np.vstack((xDet, yDet, zDet)).T
  sample_detector_path_length = np.linalg.norm(posDetector, axis=1)

  v_out_det = posDetector / sample_detector_path_length[:, np.newaxis]
  v_in_alpha = np.array([0, np.cos(alpha_inc), np.sin(alpha_inc)])

  if notTOFInstrument is False: #TOF instruments
    wavelengthDet = calcWavelength(t+sample_detector_tof, nominal_source_sample_distance+sample_detector_path_length)
    qFactor = qConvFactor(wavelengthDet)[:, np.newaxis]
  else: #not TOF instruments
    qFactor = qConvFactorFixed

  return (v_out_det - v_in_alpha) * qFactor

def processNeutrons(neutron, sc=None):
  """Carry out the BornAgain simulation and subsequent calculations of single
  incident neutron.
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
  if sc is None:
    sc = sharedMemoryHandler.getConstants() #get constants from shared memory

  sim_module = import_module('models.'+sc['sim_module_name'])
  get_sample = sim_module.get_sample
  if sc['sim_module_name'] == "silica_100nm_air":
    sample = get_sample(radius=sc['silicaRadius'])
  else:
    sample = get_sample()

  notTOFInstrument = sc['wavelengthSelected'] is not None
  qConvFactorFixed = None if sc['wavelengthSelected'] is None else qConvFactor(sc['wavelengthSelected'])

  p, x, y, z, vx, vy, vz, t = neutron
  alpha_i = np.rad2deg(np.arctan(vz/vy)) #deg
  phi_i = np.rad2deg(np.arctan(vx/vy)) #deg
  v = np.sqrt(vx**2+vy**2+vz**2)
  wavelength = velocityToWavelength(v) #angstrom

  # calculate (pixelNr)^2 outgoing beams with a random angle within one pixel range
  Ry = 2*np.random.random()-1
  Rz = 2*np.random.random()-1
  sim = get_simulation(sample, sc['pixelNr'], sc['angle_range'], wavelength, alpha_i, p, Ry, Rz)
  sim.options().setUseAvgMaterials(True)
  sim.options().setIncludeSpecular(True)
  # sim.options().setNumberOfThreads(n) ##Experiment with this? If not parallel processing?

  res = sim.simulate()
  # get probability (intensity) for all pixels
  pout = res.array()
  # calculate beam angle relative to coordinate system, including incident beam direction
  alpha_f = sc['angle_range']*(np.linspace(1., -1., sc['pixelNr'])+Ry/(sc['pixelNr']-1))
  phi_f = phi_i+sc['angle_range']*(np.linspace(-1., 1., sc['pixelNr'])+Rz/(sc['pixelNr']-1))
  alpha_grid, phi_grid = np.meshgrid(np.deg2rad(alpha_f), np.deg2rad(phi_f))

  # These are expressed in the rotated McStas coord system (X - left; y - forward; Z - downward)
  VX_grid = v * np.cos(alpha_grid) * np.sin(phi_grid) #this is Y in BA coord system) (horizontal - to the left)
  VY_grid = v * np.cos(alpha_grid) * np.cos(phi_grid) #this is X in BA coord system) (horizontal - forward)
  VZ_grid = -v * np.sin(alpha_grid)                   #this is Z in BA coord system) (horizontal - up)
  qArray = getQsAtDetector(x, y, z, t, sc['alpha_inc'], VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten(), sc['nominal_source_sample_distance'], sc['sample_detector_distance'], notTOFInstrument, qConvFactorFixed)
  if sc['raw_output']: #raw q events output format
    return np.column_stack([pout.T.flatten(), qArray])
  else: #histogrammed output format
    sharedMemoryHandler.incrementSharedHistograms(qArray, weights=pout.T.flatten())

def createConstantsDict(args, instParams):
  return {
    'nominal_source_sample_distance': instParams['nominal_source_sample_distance'] - (0 if not args.wfm else instParams['wfm_virtual_source_distance']),
    'sample_detector_distance': instParams['sample_detector_distance'],
    'sim_module_name': args.model,
    'silicaRadius': args.silicaRadius,
    'pixelNr': args.pixel_number,
    'wavelengthSelected':  None if instParams['tof_instrument'] else args.wavelengthSelected,
    'alpha_inc': float(np.deg2rad(args.alpha)),
    'angle_range': args.angle_range,
    'raw_output': args.raw_output
  }

def main(args):
  """The actual script execution after the input arguments are parsed."""
  #Constant values necessary for neutron processing, that are stored in shared memory if parallel processing is used
  sharedConstants = createConstantsDict(args, instrumentParameters[args.instrument])

  ### Getting neutron events from the MCPL file ###
  mcplTofLimits = getTofFilteringLimits(args, instrumentParameters[args.instrument])
  from inputOutput import getNeutronEvents
  events = getNeutronEvents(args.filename, mcplTofLimits)

  ### Preconditioning ###
  events = coordTransformToSampleSystem(events, sharedConstants['alpha_inc'])
  events = propagateToSampleSurface(events, args.sample_xwidth, args.sample_yheight)
  events = applyT0Correction(events, args)

  ### BornAgain simulation ###
  savename = f"q_events_pix{sharedConstants['pixelNr']}" if args.savename == '' else args.savename
  print('Number of events being processed: ', len(events))
  
  if args.all_q: #old, non-vectorised, non-parallel processing, resulting in multiple q values with different definitions
    from oldProcessing import processNeutronsNonVectorised
    processNeutronsNonVectorised(events, get_simulation, sharedConstants, savename)
    return

  if args.no_parallel: #not using parallel processing, iterating over each neutron sequentially, mainly intended for profiling
    #NOTE the storing histogrammed results could be implemented, but right now it is raw_output only
    q_events_calc_detector = []
    for in_ID, neutron in enumerate(events):
      if in_ID%200==0:
        print(f'{in_ID:10}/{len(events)}')
      tmp = processNeutrons(neutron, sharedConstants)
      q_events_calc_detector.extend(tmp) #this solution can cause memory issues for high incident event and pixel number
    np.savez_compressed(savename, q_events_calc_detector=q_events_calc_detector)
    print(f"Created {savename}.npz with raw Q events.")
    return

  num_processes = args.parallel_processes if args.parallel_processes else (cpu_count() - 2)
  print(f"Number of parallel processes: {num_processes} (number of CPU cores: {cpu_count()})")
  try:
    #TODO the histogram memory blocks should not be created if args.raw_output
    sharedMemoryHandler.createSharedMemoryBlocks(sharedConstants, args.bins, args.x_range, args.y_range, args.z_range) #using shared memory to pass in constants for the parallel processes and store result by incrementing shared histograms
    with Pool(processes=num_processes) as pool:
      # Use tqdm to wrap the iterable returned by pool.imap for the progressbar
      q_events = list(tqdm(pool.imap_unordered(processNeutrons, events), total=len(events)))

    if not args.raw_output:
      final_hist, final_error, xEdges, yEdges, zEdges = sharedMemoryHandler.getFinalHistograms()
  finally:
    sharedMemoryHandler.cleanup()

  ### Create Output ###
  if args.raw_output: # Create raw list of Q events (old output)
    #storing all Q events in a list can cause memory issues (earlier at the BornAgain simulation stage) for high incident event and pixel number
    q_events_calc_detector = [item for sublist in q_events for item in sublist] 
    np.savez_compressed(savename, q_events_calc_detector=q_events_calc_detector)
    print(f"Created {savename}.npz with raw Q events.")
    return 
  
  np.savez_compressed(savename, hist=final_hist, error=final_error, xEdges=xEdges, yEdges=yEdges, zEdges=zEdges)
  print(f"Created {savename}.npz")

  if args.quick_plot:
    hist2D = np.sum(final_hist, axis=1)
    from plotting_utilities import logPlot2d
    logPlot2d(hist2D.T, -xEdges, -zEdges, xRange=args.x_range, yRange=args.z_range, output='show')

if __name__=='__main__':
  def getBornAgainModels():
    """Get all the Born Again sample models from the ./models directory."""
    import sys
    scriptDir = Path(sys.argv[0]).resolve().parent
    return[f.stem for f in Path(scriptDir / 'models').iterdir() if f.is_file() and f.stem != '__init__']

  import argparse
  parser = argparse.ArgumentParser(description = 'Execute BornAgain simulation of a GISANS sample with incident neutrons taken from an input file. The output of the script is a .npz file (or files) containing the derived Q values for each outgoing neutron. The default Q value calculated is aiming to be as close as possible to the Q value from a measurement.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('filename',  help = 'Input filename. (Preferably MCPL file from the McStas MCPL_output component, but .dat file from McStas Virtual_output works as well)')
  parser.add_argument('-i','--instrument', required=True, choices=list(instrumentParameters.keys()), help = 'Instrument (from instruments.py).')
  parser.add_argument('-p','--parallel_processes', required=False, type=int, help = 'Number of processes to be used for parallel processing.')
  parser.add_argument('--no_parallel', default=False, action='store_true', help = 'Do not use multiprocessing. This makes the simulation significantly slower, but enables profiling. Uses --raw_output implicitly.')
  parser.add_argument('-n','--pixel_number', default=10, type=int, help = 'Number of pixels in x and y direction of the "detector".')
  parser.add_argument('--wavelengthSelected', default=6.0, type=float, help = 'Wavelength (mean) in Angstrom selected by the velocity selector. Only used for non-time-of-flight instruments.')
  parser.add_argument('--angle_range', default=1.7, type=float, help = 'Scattering angle covered by the simulation. [deg]')

  outputGroup = parser.add_argument_group('Output', 'Control the generated outputs. By default a histogram (and corresponding uncertainty) is generated as an output, saved in a npz file, loadable with the plotQ script.')
  outputGroup.add_argument('-s', '--savename', default='', required=False, help = 'Output filename (can be full path).')
  outputGroup.add_argument('--raw_output', default=False, action='store_true', help = 'Create a raw list of Q events as output instead of the default histogrammed data. Warning: this option may require too much memory for high incident event and pixel numbers.')
  outputGroup.add_argument('--bins', nargs=3, type=int, default=[256, 1, 128], help='Number of histogram bins in x,y,z directions.')
  outputGroup.add_argument('--x_range', nargs=2, type=float, default=[-0.55, 0.55], help='Qx range of the histogram. (In horizontal plane left to right)')
  outputGroup.add_argument('--y_range', nargs=2, type=float, default=[-1000, 1000], help='Qy range of the histogram. (In horizontal plane back to front)')
  outputGroup.add_argument('--z_range', nargs=2, type=float, default=[-0.5, 0.6], help='Qz range of the histogram. (In vertical plane bottom to top')
  outputGroup.add_argument('--quick_plot', default=False, action='store_true', help='Show a quick Qx-Qz plot from the histogram result.')
  outputGroup.add_argument('--all_q', default=False, action='store_true', help = 'Calculate and save multiple Q values, each with different levels of approximation (from real Q calculated from all simulation parameters to the default output value, that is Q calculated at the detector surface). This results in significantly slower simulations (especially due to the lack of parallelisation), but can shed light on the effect of e.g. divergence and TOF to lambda conversion on the derived Q value, in order to gain confidence in the results.')

  sampleGroup = parser.add_argument_group('Sample', 'Sample related parameters and options.')
  sampleGroup.add_argument('-a', '--alpha', default=0.24, type=float, help = 'Incident angle on the sample. [deg] (Could be thought of as a sample rotation, but it is actually achieved by an incident beam coordinate transformation.)')
  sampleGroup.add_argument('-m','--model', default="silica_100nm_air", choices=getBornAgainModels(), help = 'BornAgain model to be used.')
  sampleGroup.add_argument('-r', '--silicaRadius', default=53, type=float, help = 'Silica particle radius for the "Silica particles on Silicon measured in air" sample model.')
  sampleGroup.add_argument('--sample_xwidth', default=0.06, type=float, help = 'Size of sample perpendicular to beam. [m]')
  sampleGroup.add_argument('--sample_yheight', default=0.08, type=float, help = 'Size of sample along the beam. [m]')

  mcplFilteringGroup = parser.add_argument_group('MCPL filtering', 'Parameters and options to control which neutrons are used from the MCPL input file. By default no filtering is applied, but if a (central) wavelength is provided, an accepted TOF range is defined based on a McStas TOFLambda monitor (defined as mcpl_monitor_name for each instrument in instruments.py) that is assumed to correspond to the input MCPL file. The McStas monitor is looked for in the directory of the MCPL input file, and after fitting a Gaussian function, neutrons within a single FWHM range centred around the selected wavelength are used for the BornAgain simulation.')
  mcplFilteringGroup.add_argument('-w', '--wavelength', type=float, help = 'Central wavelength used for filtering based on the McStas TOFLambda monitor. (Also used for t0 correction.)')
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
      parser.error(f"The --figure_output option can not be used when TOF no filtering is selected with --no_mcpl_filtering.")

  if args.no_t0_correction:
    if args.t0_fixed:
      parser.error(f"The --no_t0_correction option can not be used together with --t0_fixed ")
    if args.t0_wavelength_rebin:
      parser.error(f"The --no_t0_correction option can not be used together with --t0_wavelength_rebin ")
    if args.wfm:
      parser.error(f"The --no_t0_correction option can not be used together with --wfm ")

  if args.t0_fixed:
    if args.t0_wavelength_rebin:
      parser.error(f"The --t0_fixed option can not be used together with --t0_wavelength_rebin ")

  if args.all_q and args.wfm:
    parser.error(f"The --all_q option can not be used together with --wfm. (Simply because --all_q is not well maintained, but it can be changed on demand.)")

  main(args)
