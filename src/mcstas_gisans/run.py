#!/usr/bin/env python3

"""
Handles the simulation and processing of particle scattering experiments.
It takes particles from an MCPL file, applies transformations, and processes
them through BornAgain simulations to generate Q values for each incident
particle, and saves the result for further analysis or plotting.
"""

import os
import sys
import tempfile
import traceback
import multiprocessing
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

import bornagain as ba
from bornagain import deg, angstrom

from .hardware import get_available_cores
from .input_output import get_particles, save_simulation_results_as_scipp
from .preconditioning import precondition
from .tof_filtering import get_tof_filtering_limits
from .parameters import pack_parameters

def get_simulation(
    sample: Any,
    outgoing_directions_horizontal: int,
    outgoing_directions_vertical: int,
    angle_range: List[float],
    wavelength: float,
    alpha_i: float,
    p: float,
    rand_y: float,
    rand_z: float,
    polarization: List[float],
    analyzer_direction: List[float],
    analyzer_efficiency: float,
    analyzer_transmission: float
) -> ba.ScatteringSimulation:
    """
    Create a simulation with outgoing_directions_horizontal and _vertical pixels covering the 4-element angle_range
    [horiz_min, horiz_max, vert_min, vert_max] in degrees.
    """
    beam = ba.Beam(p, wavelength*angstrom, alpha_i*deg)

    horiz_min, horiz_max, vert_min, vert_max = angle_range

    step_phi = (horiz_max - horiz_min) / (outgoing_directions_horizontal - 1) if outgoing_directions_horizontal > 1 else 0.0
    step_alpha = (vert_max - vert_min) / (outgoing_directions_vertical - 1) if outgoing_directions_vertical > 1 else 0.0

    rand_deg_phi = rand_z * step_phi
    rand_deg_alpha = rand_y * step_alpha

    detector = ba.SphericalDetector(
        outgoing_directions_horizontal, (horiz_min + rand_deg_phi)*deg, (horiz_max + rand_deg_phi)*deg,
        outgoing_directions_vertical, (vert_min + rand_deg_alpha)*deg, (vert_max + rand_deg_alpha)*deg
    )
    if polarization:
        beam.setPolarization(ba.R3(*polarization))
        if analyzer_direction:
            detector.setAnalyzer(ba.R3(*analyzer_direction), analyzer_efficiency, analyzer_transmission)

    return ba.ScatteringSimulation(beam, sample, detector)

def get_simulation_specular(sample: Any, wavelength: float, alpha_i: float) -> ba.SpecularSimulation:
    """
    Create a specular simulation for evaluating reflection probabilities.
    """
    scan = ba.AlphaScan(2, alpha_i*deg, alpha_i*deg+1e-6)
    scan.setWavelength(wavelength*angstrom)
    return ba.SpecularSimulation(scan, sample)

def get_result_intensities(res: Any) -> np.ndarray:
    """
    Extract the 2D array of simulated intensities from a BornAgain result.
    """
    if hasattr(res, 'intensities'):
        pout = np.flipud(res.intensities())
    elif hasattr(res, 'array'):
        pout = res.array()
    elif hasattr(res, 'flatVector'):
        nx = res.xAxis().size()
        ny = res.yAxis().size()
        pout = np.array(res.flatVector()).reshape(ny, nx)
        pout = np.flipud(pout)
    else:
        print("ERROR: Could not extract data from the simulation result.")
        sys.exit("Terminating script due to incompatible BornAgain version.")
    return pout

def _execute_bornagain_simulation(
    sample_model: Any,
    wavelength: float,
    alpha_i: float,
    p: float,
    polarization: List[float],
    params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Setup and run the BornAgain scattering simulation for a single particle.
    Returns:
        (weights, alpha_f_grid, phi_f_grid)
    """
    angle_range = params['angle_range']
    outgoing_directions_horizontal = params['outgoing_directions_horizontal']
    outgoing_directions_vertical = params['outgoing_directions_vertical']

    rand_y = 2*np.random.random()-1
    rand_z = 2*np.random.random()-1
    
    sim = get_simulation(
        sample_model, outgoing_directions_horizontal, outgoing_directions_vertical,
        angle_range, wavelength, alpha_i, p, rand_y, rand_z, polarization,
        analyzer_direction=params.get('analyzer_direction', [0,0,0]),
        analyzer_efficiency=params.get('analyzer_efficiency', 1.0),
        analyzer_transmission=params.get('analyzer_transmission', 0.5)
    )
    sim.options().setUseAvgMaterials(params.get('use_avg_materials', False))
    sim.options().setIncludeSpecular(params.get('specular') == 'include_specular')
    bornagain_number_of_threads = params.get('bornagain_number_of_threads')
    if bornagain_number_of_threads is not None:
        sim.options().setNumberOfThreads(bornagain_number_of_threads)

    res = sim.simulate()
    pout = get_result_intensities(res)

    horiz_min, horiz_max, vert_min, vert_max = angle_range
    step_phi = (horiz_max - horiz_min) / (outgoing_directions_horizontal - 1) if outgoing_directions_horizontal > 1 else 0.0
    step_alpha = (vert_max - vert_min) / (outgoing_directions_vertical - 1) if outgoing_directions_vertical > 1 else 0.0

    rand_deg_phi = rand_z * step_phi
    rand_deg_alpha = rand_y * step_alpha

    alpha_f = np.linspace(vert_max, vert_min, outgoing_directions_vertical) + rand_deg_alpha
    phi_f = np.linspace(horiz_min, horiz_max, outgoing_directions_horizontal) + rand_deg_phi
    
    weights = pout.T.flatten()
    return weights, alpha_f, phi_f

def _calculate_nexus_hits(
    instrument: Any,
    x: float, y: float, z: float, t: float,
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the pixels hit on the detector given an array of outgoing velocity vectors.
    """
    idx_x_nexus, idx_y_nexus, valid_mask, sd_tof = instrument.calculate_pixel_hit(x, y, z, t, vx, vy, vz)
    return idx_x_nexus, idx_y_nexus, valid_mask, sd_tof

def _update_non_tof_buffer(
    valid_idx_x: np.ndarray,
    valid_idx_y: np.ndarray,
    valid_weights: np.ndarray,
    pixel_hist: np.ndarray,
    pixel_hist_weights_squared: np.ndarray
) -> None:
    """
    Update the non-TOF histogram arrays.
    """
    np.add.at(pixel_hist, (valid_idx_x, valid_idx_y), valid_weights)
    np.add.at(pixel_hist_weights_squared, (valid_idx_x, valid_idx_y), valid_weights**2)

def _update_tof_buffer(
    valid_weights: np.ndarray,
    detector_id: np.ndarray,
    total_tof: np.ndarray,
    event_buffer: Dict[str, np.ndarray],
    f_temp: Any,
    buffer_idx: int,
    buffer_capacity: int
) -> int:
    """
    Update the TOF HDF5 event buffer.
    """
    num_hits = len(detector_id)
    hits_processed = 0

    while hits_processed < num_hits:
        space_left = buffer_capacity - buffer_idx
        chunk_size = min(space_left, num_hits - hits_processed)

        end_processed = hits_processed + chunk_size
        end_buffer = buffer_idx + chunk_size

        event_buffer['detector_id'][buffer_idx:end_buffer] = detector_id[hits_processed:end_processed]
        event_buffer['tof'][buffer_idx:end_buffer] = total_tof[hits_processed:end_processed]
        event_buffer['weight'][buffer_idx:end_buffer] = valid_weights[hits_processed:end_processed]

        buffer_idx += chunk_size
        hits_processed += chunk_size

        if buffer_idx == buffer_capacity:
            for col in ['detector_id', 'tof', 'weight']:
                dset = f_temp[col]
                old_size = dset.shape[0]
                dset.resize((old_size + buffer_capacity,))
                dset[old_size:] = event_buffer[col]
            f_temp.flush()
            buffer_idx = 0

    return buffer_idx

def process_particles(particles: Any, params: Dict[str, Any], queue: Optional[multiprocessing.Queue] = None) -> Dict[str, Any]:
    """
    Carry out the BornAgain simulation and subsequent processing for a batch of incident particles.
    """
    try:
        sample = params['sample']
        sample_module = sample.get_module()
        sample_model = sample_module.get_sample(**sample.kwargs)
        instrument = params['instrument']
        is_tof = instrument.is_tof_instrument
        specular = params['specular']

        pixel_hist = None
        pixel_hist_weights_squared = None
        event_buffer = None
        buffer_idx = 0
        buffer_capacity = 1_000_000
        h5_temp_path = None
        f_temp = None

        if not is_tof:
            pixel_hist = np.zeros((instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus), dtype=np.float64)
            pixel_hist_weights_squared = np.zeros((instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus), dtype=np.float64)
        else:
            import h5py
            pid = multiprocessing.current_process().pid or os.getpid()
            h5_temp_path = os.path.join(tempfile.gettempdir(), f"mcstas_gisans_events_{pid}.h5")

            event_buffer = {
                'detector_id': np.zeros(buffer_capacity, dtype=np.int32),
                'tof': np.zeros(buffer_capacity, dtype=np.float32),
                'weight': np.zeros(buffer_capacity, dtype=np.float64)
            }

            f_temp = h5py.File(h5_temp_path, 'w')
            f_temp.create_dataset('detector_id', shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(100_000,))
            f_temp.create_dataset('tof', shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(100_000,))
            f_temp.create_dataset('weight', shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(100_000,))

        for id, particle in enumerate(particles):
            if id % 200 == 0:
                ident = getattr(multiprocessing.current_process(), '_identity', (1,))
                if not ident or ident[0] == 1:
                    print(f'{id:10}/{len(particles)}')
                    sys.stdout.flush()
                
            p, x, y, z, vx, vy, vz, wavelength, t, *polarization = particle
            alpha_i = np.rad2deg(np.arctan(-vz/vx))
            phi_i = np.rad2deg(np.arctan(vy/vx))
            v = np.sqrt(vx**2 + vy**2 + vz**2)

            if sample.sample_missed(x, y, z, vz):
                weights_pixel_combined = np.array([p])
                idx_x_nexus, idx_y_nexus, valid_mask, sd_tof = _calculate_nexus_hits(
                    instrument, x, y, z, t, np.array([vx]), np.array([vy]), np.array([vz])
                )
            else:
                weights, alpha_f_grid, phi_f_grid = _execute_bornagain_simulation(
                    sample_model, wavelength, alpha_i, p, polarization, params
                )
                
                phi_f_total = phi_i + phi_f_grid
                alpha_grid_2d, phi_grid_2d = np.meshgrid(np.deg2rad(alpha_f_grid), np.deg2rad(phi_f_total))
                
                VX_grid = v * np.cos(alpha_grid_2d) * np.cos(phi_grid_2d)
                VY_grid = v * np.cos(alpha_grid_2d) * np.sin(phi_grid_2d)
                VZ_grid = v * np.sin(alpha_grid_2d)

                idx_x_nexus, idx_y_nexus, valid_mask, sd_tof = _calculate_nexus_hits(
                    instrument, x, y, z, t, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten()
                )

                if specular == 'specular_simulation':
                    ssim = get_simulation_specular(sample_model, wavelength, alpha_i)
                    res = ssim.simulate()
                    refl_fraction = np.array(res.flatVector())[0]

                    weight_specular_sim = [np.array([p * refl_fraction])]
                    idx_x_refl, idx_y_refl, valid_refl, sd_tof_refl = _calculate_nexus_hits(
                        instrument, x, y, z, t, np.array([vx]), np.array([vy]), np.array([-vz])
                    )
                    
                    idx_y_list = [idx_y_nexus, idx_y_refl]
                    idx_x_list = [idx_x_nexus, idx_x_refl]
                    valid_list = [valid_mask, valid_refl]
                    sd_tof_list = [sd_tof, sd_tof_refl]
                    weights_pixel = [weights, np.array([p * refl_fraction])]

                    ptrans = p * (1.0 - refl_fraction)
                    if ptrans > 1e-10:
                        idx_x_trans, idx_y_trans, valid_trans, sd_tof_trans = _calculate_nexus_hits(
                            instrument, x, y, z, t, np.array([vx]), np.array([vy]), np.array([vz])
                        )
                        idx_y_list.append(idx_y_trans)
                        idx_x_list.append(idx_x_trans)
                        valid_list.append(valid_trans)
                        sd_tof_list.append(sd_tof_trans)
                        weights_pixel.append(np.array([ptrans]))

                    idx_y_nexus = np.concatenate(idx_y_list)
                    idx_x_nexus = np.concatenate(idx_x_list)
                    valid_mask = np.concatenate(valid_list)
                    sd_tof = np.concatenate(sd_tof_list)
                    weights_pixel_combined = np.concatenate(weights_pixel)
                else:
                    weights_pixel_combined = weights

            if np.any(valid_mask):
                valid_idx_x = idx_x_nexus[valid_mask]
                valid_idx_y = idx_y_nexus[valid_mask]
                valid_weights = weights_pixel_combined[valid_mask]
                detector_id = valid_idx_x * instrument.detector.pixels_y_nexus + valid_idx_y

                if not is_tof:
                    _update_non_tof_buffer(
                        valid_idx_x, valid_idx_y, valid_weights,
                        pixel_hist, pixel_hist_weights_squared
                    )
                else:
                    valid_sd_tof = sd_tof[valid_mask]
                    total_tof = t + valid_sd_tof
                    buffer_idx = _update_tof_buffer(
                        valid_weights, detector_id, total_tof,
                        event_buffer, f_temp, buffer_idx, buffer_capacity
                    )

        if is_tof and buffer_idx > 0 and f_temp is not None:
            for col in ['detector_id', 'tof', 'weight']:
                dset = f_temp[col]
                old_size = dset.shape[0]
                dset.resize((old_size + buffer_idx,))
                dset[old_size:] = event_buffer[col][:buffer_idx]
            buffer_idx = 0

        result = {
            'pixelHist': pixel_hist,
            'pixelHistWeightsSquared': pixel_hist_weights_squared,
            'temp_h5_path': h5_temp_path
        }

        if queue:
            queue.put(result)
        else:
            return result

    except Exception as e:
        err_log_path = os.path.join(tempfile.gettempdir(), 'mcstas_worker_err.log')
        with open(err_log_path, 'a') as f:
            traceback.print_exc(file=f)
        raise e
    finally:
        if params['instrument'].is_tof_instrument and f_temp is not None:
            try:
                f_temp.close()
            except:
                pass

def process_particles_parallelly(particles: Any, params: Dict[str, Any], process_number: int) -> Dict[str, Any]:
    """
    Spawn parallel processes to carry out the BornAgain simulation and subsequent
    calculation of the incident particles.
    """
    print(f"Number of parallel processes: {process_number} (number of physical CPU cores: {get_available_cores()})")

    err_log_path = os.path.join(tempfile.gettempdir(), 'mcstas_worker_err.log')
    if os.path.exists(err_log_path):
        try:
            os.remove(err_log_path)
        except OSError:
            pass

    particle_number = len(particles)
    chunk_size = particle_number // process_number
    chunks = []
    for i in range(process_number):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < process_number - 1 else particle_number
        chunks.append(particles[start:end])

    with multiprocessing.Pool(processes=process_number) as pool:
        results = pool.starmap(process_particles, [(chunk, params) for chunk in chunks])

    if os.path.exists(err_log_path) and os.path.getsize(err_log_path) > 0:
        with open(err_log_path, 'r') as f:
            err_content = f.read()
        raise RuntimeError(f"One or more parallel workers failed with a traceback:\n{err_content}")

    is_tof = params['instrument'].is_tof_instrument
    pixel_hist = np.zeros((params['instrument'].detector.pixels_x_nexus, params['instrument'].detector.pixels_y_nexus))
    pixel_hist_weights_squared = np.zeros((params['instrument'].detector.pixels_x_nexus, params['instrument'].detector.pixels_y_nexus))
    
    for process_result in results:
        if process_result['pixelHist'] is not None:
            pixel_hist += process_result['pixelHist']
            pixel_hist_weights_squared += process_result['pixelHistWeightsSquared']
            
    result: Dict[str, Any] = {
        'pixelHist': pixel_hist,
        'pixelHistWeightsSquared': pixel_hist_weights_squared
    }
    
    if is_tof:
        result['temp_h5_paths'] = [r['temp_h5_path'] for r in results if 'temp_h5_path' in r]
        
    return result

def main() -> None:
    """
    Main entry point for the mg_run CLI tool.
    
    Parses command-line arguments, preconditions the incident neutrons,
    runs the BornAgain DWBA simulation, and saves the output dataset.
    """
    from .run_cli import create_argparser, parse_args
    parser = create_argparser()
    args = parse_args(parser)

    tof_limits = get_tof_filtering_limits(args)
    particles, particle_type, mcpl_metadata = get_particles(args.filename, args.intensity_factor, tof_limits, args.input_weight_limit, use_polarization=args.use_polarization)

    particles = precondition(particles, args)

    if args.outgoing_directions is not None:
        suffix = args.outgoing_directions
    else:
        suffix = f"{args.outgoing_directions_horizontal}_{args.outgoing_directions_vertical}"
    savename = f"q_events_pix{suffix}" if args.savename == '' else args.savename
    print('Number of particles being processed: ', len(particles))

    params = pack_parameters(args, particle_type)

    if args.no_parallel:
        result = process_particles(particles, params)
    else:
        process_number = args.parallel_processes if args.parallel_processes else (get_available_cores() - 1)
        result = process_particles_parallelly(particles, params, process_number)

    save_simulation_results_as_scipp(savename, params, result, args, mcpl_metadata, getattr(args, 'temp_read_chunk_size', 1000000))

if __name__=='__main__':
    main()
