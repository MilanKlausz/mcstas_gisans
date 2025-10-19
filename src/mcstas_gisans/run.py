#!/usr/bin/env python3

"""
Handles the simulation and processing of particle scattering experiments.
It takes particles from an MCPL file, applies transformations, and processes
them through BornAgain simulations to generate Q values for each incident
particle, and saves the result for further analysis or plotting.
"""

import numpy as np
from multiprocessing import cpu_count, Queue
import multiprocessing

import bornagain as ba
from bornagain import deg, angstrom

from .input_output import get_particles, save_q_histogram_file, save_raw_q_list_file
from .preconditioning import precondition
from .tof_filtering import get_tof_filtering_limits
from .parameters import pack_parameters

def get_simulation(sample, pixel_number, angle_range, wavelength, alpha_i, p, rand_y, rand_z):
  """
  Create a simulation with pixel_number pixels that cover an angular range of
  angle_range degrees. The rand_deg_y and rand_deg_z values are relative
  rotations of the detector within one pixel to finely sample the outgoing
  direction space.
  """
  beam = ba.Beam(p, wavelength*angstrom, alpha_i*deg)

  rand_deg_y = rand_y*angle_range[1]*deg/(pixel_number-1)
  rand_deg_z = rand_z*angle_range[0]*deg/(pixel_number-1)

  # Define detector
  detector = ba.SphericalDetector(pixel_number, -angle_range[0]*deg + rand_deg_z, angle_range[0]*deg + rand_deg_z,
                                  pixel_number, -angle_range[1]*deg + rand_deg_y, angle_range[1]*deg + rand_deg_y)

  return ba.ScatteringSimulation(beam, sample, detector)


def process_particles(particles, params, queue=None):
  """Carry out the BornAgain simulation and subsequent calculations of each
  incident particle (separately) in the input array.
  1) The BornAgain simulation for a certain sample model is set up with an array
     out outgoing directions
  2) The BornAgain simulation is performed, resulting in an array of outgoing
     beams with weights (outgoing probabilities)
  3) The Q values are calculated after a virtual propagation to the detector
     surface.
  4) Depending on the input options, the list of Q events (weight,qx,qy,qz) are
     either returned (old raw format), or histogrammed and added to a cumulative
     histogram where all other incident particle results are added.
  """
  sample = params['sample']
  sample_module = sample.get_module()
  sample_model = sample_module.get_sample(**sample.kwargs)

  calculate_q = params['instrument'].calculate_q
  outgoing_direction_number = params['outgoing_direction_number']
  angle_range = params['angle_range']
  raw_output = params['raw_output']
  hist_ranges = params['hist_ranges']
  bins = params['bins']
  use_avg_materials = params['use_avg_materials']
  include_specular = params['include_specular']

  if raw_output:
    q_events = [] #p, Qx, Qy, Qz, t
  else:
    q_hist = np.zeros(tuple(bins))
    q_hist_weights_squared = np.zeros(bins)

  ## Carry out BornAgain simulation for all incident particle one-by-one
  for id, particle in enumerate(particles):
    if id%200==0:
      print(f'{id:10}/{len(particles)}') #print output to indicate progress
    # Particle positions, velocities and corresponding calculations are expressed
    # in the McStas coord system (X - left; Y - up; Z - forward 'along the beam')
    # not in the BornAgain coord system (X - forward, Y - left, Z - up),
    # but with the SphericalDetector, BornAgain only deals with alpha_i (input),
    # alpha_f and phi_f (output), which are the same if calculated correctly
    p, x, y, z, vx, vy, vz, wavelength, t = particle
    alpha_i = np.rad2deg(np.arctan(-vy/vz)) #[deg]
    phi_i = np.rad2deg(np.arctan(vx/vz)) #[deg], not used in sim, added to phi_f
    v = np.sqrt(vx**2+vy**2+vz**2)

    if sample.sample_missed(x,z):
      # Particles missed the sample so the q value is calculated after propagation
      # to the detector surface without scattering simulation
      q_array = calculate_q(x, y, z, t, [vx], [vy], [vz])
      weights = np.array([p])
    else:
      # Calculate scattering probability for (outgoing_direction_number)^2
      # outgoing beams. The outgoing direction grid is evenly spaced within the
      # sampled angle range, but random angle offset of the whole grid in both
      # directions is applied for better sampling of the outgoing directions
      rand_y = 2*np.random.random()-1
      rand_z = 2*np.random.random()-1
      sim = get_simulation(sample_model, outgoing_direction_number, angle_range, wavelength, alpha_i, p, rand_y, rand_z)
      sim.options().setUseAvgMaterials(use_avg_materials)
      sim.options().setIncludeSpecular(include_specular)
      # sim.options().setNumberOfThreads(n) ##Experiment with this? If not parallel processing?
      res = sim.simulate()

      # get probability (intensity) for all outgoing directions
      pout = res.array()
      # calculate the components of the velocity vector for all outgoing directions
      alpha_f = angle_range[1] * (np.linspace(1., -1., outgoing_direction_number) + rand_y / (outgoing_direction_number - 1))
      phi_f = phi_i + angle_range[0] * (np.linspace(-1., 1., outgoing_direction_number) + rand_z / (outgoing_direction_number - 1))
      alpha_grid, phi_grid = np.meshgrid(np.deg2rad(alpha_f), np.deg2rad(phi_f))
      VX_grid = v * np.cos(alpha_grid) * np.sin(phi_grid) #this is Y in BA coord system) (horizontal - to the left)
      VY_grid = v * np.sin(alpha_grid)                    #this is Z in BA coord system) (horizontal - up)
      VZ_grid = v * np.cos(alpha_grid) * np.cos(phi_grid) #this is X in BA coord system) (horizontal - forward)

      q_array = calculate_q(x, y, z, t, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten())
      weights = pout.T.flatten()
    if raw_output:
      q_events.append(np.column_stack([weights, q_array]))
    else: #histogrammed output format
      q_hist_of_particle, _ = np.histogramdd(q_array, weights=weights, bins=bins, range=hist_ranges)
      q_hist_weights_squared_of_particle, _ = np.histogramdd(q_array, weights=weights**2, bins=bins, range=hist_ranges)
      q_hist += q_hist_of_particle
      q_hist_weights_squared += q_hist_weights_squared_of_particle

  if raw_output:
    result = [item for sublist in q_events for item in sublist] #flatten sublists
  else:
    result = {'qHist': q_hist, 'qHistWeightsSquared': q_hist_weights_squared}

  if queue: #return result from multiprocessing process
    queue.put(result)
  else:
    return result

def process_particles_parallelly(particles, params, process_number):
  """
  Spawn parallel processes to carry out the BornAgain simulation and subsequent
  calculation of the incident particles.
  """
  print(f"Number of parallel processes: {process_number} (number of CPU cores: {cpu_count()})")

  processes = []
  results = []
  queue = Queue() #a queue to get results from each process

  particle_number = len(particles)
  chunk_size = particle_number // process_number
  def get_particles_chunk(process_index):
    """Distribute the events array among the processes as evenly as possible"""
    start = process_index * chunk_size
    end = (process_index + 1) * chunk_size if process_index < process_number - 1 else particle_number
    return particles[start:end]

  for i in range(process_number):
    p = multiprocessing.Process(target=process_particles, args=(get_particles_chunk(i), params, queue,))
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
    q_hist = np.zeros(tuple(params['bins']))
    q_hist_weights_squared = np.zeros(tuple(params['bins']))
    for process_result in results:
      q_hist += process_result['qHist']
      q_hist_weights_squared += process_result['qHistWeightsSquared']
    result = {'qHist': q_hist, 'qHistWeightsSquared': q_hist_weights_squared}
  return result

def main():
  from .run_cli import create_argparser, parse_args
  parser = create_argparser()
  args = parse_args(parser)

  ### Get particles from the MCPL file ###
  tof_limits = get_tof_filtering_limits(args)
  particles, particle_type = get_particles(args.filename, tof_limits, args.intensity_factor)

  ### Preconditioning the particles ###
  particles = precondition(particles, args)

  ### BornAgain simulation ###
  savename = f"q_events_pix{args.outgoing_direction_number}" if args.savename == '' else args.savename
  print('Number of particles being processed: ', len(particles))

  #pack parameters necessary for processing
  params = pack_parameters(args, particle_type)

  if args.no_parallel: #not using parallel processing, iterating over each particle sequentially, mainly intended for profiling
    result = process_particles(particles, params)
  else:
    process_number = args.parallel_processes if args.parallel_processes else (cpu_count() - 2)
    result = process_particles_parallelly(particles, params, process_number)

  ### Create Output ###
  if args.raw_output: #raw list of Q events (old output)
    q_array = result
    save_raw_q_list_file(savename, q_array)
    return # no further processing, early return

  ## Create Q histogram with corresponding uncertainty array (new output format)
  q_hist = result['qHist']
  q_hist_weights_squared = result['qHistWeightsSquared']
  q_hist_error = np.sqrt(q_hist_weights_squared)

  #Get the bin edges of the histograms
  edges = [np.array(np.histogram_bin_edges(None, bins=b, range=r), dtype=np.float64)
               for b, r in zip(params['bins'], params['hist_ranges'])]

  save_q_histogram_file(savename, q_hist, q_hist_error, edges)

  if args.quick_plot:
    hist2D = np.sum(q_hist, axis=2)
    from .plotting_utils import logPlot2d
    logPlot2d(hist2D, edges[0], edges[1], xRange=params['hist_ranges'][0], yRange=params['hist_ranges'][1], output='show')

if __name__=='__main__':
  main()
