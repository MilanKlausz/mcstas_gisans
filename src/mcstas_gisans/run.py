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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

import bornagain as ba
from bornagain import deg, angstrom

from .hardware import get_available_cores
from .input_output import get_particles, save_simulation_results_as_scipp
from .preconditioning import precondition
from .tof_filtering import get_tof_filtering_limits
from .parameters import pack_parameters, set_outgoing_directions_from_sampling, neutron_angle_windows, ANGLE_WINDOW_RESOLUTION_SIGMAS

def _grid_centres(angle_range: List[float], n_horizontal: int, n_vertical: int, jitter_phi: float, jitter_alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jittered centres [deg] of the outgoing-direction bins: alpha_f from the highest downwards (the row order of the
    BornAgain result), phi_f increasing. BornAgain's SphericalDetector(n, min, max) treats min/max as bin EDGES.
    """
    horiz_min, horiz_max, vert_min, vert_max = angle_range
    bin_phi = (horiz_max - horiz_min) / n_horizontal
    bin_alpha = (vert_max - vert_min) / n_vertical
    alpha_f = np.linspace(vert_max - bin_alpha / 2, vert_min + bin_alpha / 2, n_vertical) + jitter_alpha
    phi_f = np.linspace(horiz_min + bin_phi / 2, horiz_max - bin_phi / 2, n_horizontal) + jitter_phi
    return alpha_f, phi_f

def _skipped_direction_rectangles(skip: np.ndarray, alpha_f: np.ndarray, phi_f: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """
    BornAgain mask rectangles (phi_low, alpha_low, phi_high, alpha_high) [deg] covering exactly the bins to skip.
    skip has shape (len(phi_f), len(alpha_f)). Runs of skipped bins along phi_f are merged over neighbouring alpha_f
    rows with the same runs. Each rectangle reaches a quarter bin beyond the centres of its first and last bins, so it
    contains their centres and overlaps no other bin (whichever of the two BornAgain uses to decide).
    """
    bin_phi = abs(phi_f[1] - phi_f[0]) if len(phi_f) > 1 else 1.0
    bin_alpha = abs(alpha_f[1] - alpha_f[0]) if len(alpha_f) > 1 else 1.0
    def runs(column):
        edges = np.diff(np.concatenate([[0], column.astype(np.int8), [0]]))
        return tuple(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))
    rectangles = []
    row_runs = [runs(skip[:, j]) for j in range(skip.shape[1])]
    j = 0
    while j < len(row_runs):
        k = j
        while k + 1 < len(row_runs) and row_runs[k + 1] == row_runs[j]:
            k += 1
        a_low, a_high = min(alpha_f[j], alpha_f[k]), max(alpha_f[j], alpha_f[k])
        for i0, i1 in row_runs[j]:
            rectangles.append((phi_f[i0] - bin_phi / 4, a_low - bin_alpha / 4, phi_f[i1] + bin_phi / 4, a_high + bin_alpha / 4))
        j = k + 1
    return rectangles

def _reach_table(keep_pixels: np.ndarray) -> np.ndarray:
    """Summed-area table of the raw-pixel mask of the region (padded with a zero row and column)."""
    table = np.zeros((keep_pixels.shape[0] + 1, keep_pixels.shape[1] + 1), dtype=np.int64)
    table[1:, 1:] = np.cumsum(np.cumsum(keep_pixels.astype(np.int64), axis=0), axis=1)
    return table

def directions_reaching_region(
    instrument: Any, reach_table: np.ndarray, x: float, y: float, z: float, v: float, phi_i: float,
    alpha_f: np.ndarray, phi_f: np.ndarray
) -> np.ndarray:
    """
    Which outgoing directions (grid of phi_f x alpha_f [deg], as in process_particles) can reach a pixel of the
    region (reach_table of its raw-pixel mask): the unsmeared hit point of the ray, widened by
    ANGLE_WINDOW_RESOLUTION_SIGMAS sigma of the detector resolution along each detector axis (the margin of the
    per-neutron windows), touches a pixel of the region. Deterministic (no random numbers are drawn), so the
    resolution smearing of the simulated rays is unchanged. Returns a bool array of shape (len(phi_f), len(alpha_f)).
    """
    det = instrument.detector
    alpha_grid, phi_grid = np.meshgrid(np.deg2rad(alpha_f), np.deg2rad(phi_i + phi_f))
    vx = v * np.cos(alpha_grid) * np.cos(phi_grid)
    vy = v * np.cos(alpha_grid) * np.sin(phi_grid)
    vz = v * np.sin(alpha_grid)
    _, x_int, y_int, z_int = det.detector_plane_intersection(x, y, z, vx, vy, vz, instrument.sample_detector_distance)
    x_nexus, y_nexus, _ = det.coords.bornagain_to_nexus(x_int, y_int, z_int)
    margin_x = ANGLE_WINDOW_RESOLUTION_SIGMAS * det.sigma_x_nexus
    margin_y = ANGLE_WINDOW_RESOLUTION_SIGMAS * det.sigma_y_nexus
    nx, ny = det.pixels_x_nexus, det.pixels_y_nexus
    with np.errstate(invalid='ignore'):
        ix0 = np.floor((x_nexus - margin_x - det.min_edge_x_nexus) / det.pixel_size_x_nexus)
        ix1 = np.floor((x_nexus + margin_x - det.min_edge_x_nexus) / det.pixel_size_x_nexus)
        iy0 = np.floor((y_nexus - margin_y - det.min_edge_y_nexus) / det.pixel_size_y_nexus)
        iy1 = np.floor((y_nexus + margin_y - det.min_edge_y_nexus) / det.pixel_size_y_nexus)
        on_detector = np.isfinite(ix0) & np.isfinite(iy0) & (ix1 >= 0) & (ix0 < nx) & (iy1 >= 0) & (iy0 < ny)
    reach = np.zeros(alpha_grid.shape, dtype=bool)
    if np.any(on_detector):
        a0 = np.clip(ix0[on_detector], 0, nx - 1).astype(int)
        a1 = np.clip(ix1[on_detector], 0, nx - 1).astype(int) + 1
        b0 = np.clip(iy0[on_detector], 0, ny - 1).astype(int)
        b1 = np.clip(iy1[on_detector], 0, ny - 1).astype(int) + 1
        count = reach_table[a1, b1] - reach_table[a0, b1] - reach_table[a1, b0] + reach_table[a0, b0]
        reach[on_detector] = count > 0
    return reach

def _grid_jitter(angle_range: List[float], n_horizontal: int, n_vertical: int, rand_y: float, rand_z: float) -> Tuple[float, float]:
    """
    Random shift [deg] of the whole outgoing-direction grid for one incident neutron.
    rand_y, rand_z are uniform in [-1, 1]; the shift is up to half a bin width, so the
    jittered bin centres sample the angle range uniformly and never leave it.
    """
    horiz_min, horiz_max, vert_min, vert_max = angle_range
    half_bin_phi = 0.5 * (horiz_max - horiz_min) / n_horizontal
    half_bin_alpha = 0.5 * (vert_max - vert_min) / n_vertical
    return rand_z * half_bin_phi, rand_y * half_bin_alpha

def set_analyzer(target: Any, direction: List[float], efficiency: float, transmission: float) -> None:
    """
    Set the polarisation analyzer of a detector or a specular scan. BornAgain 22+ takes the Bloch vector (the unit
    direction times the efficiency) and the mean transmission; older versions took (direction, efficiency,
    transmission). A zero direction means no analyzer.
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return
    unit = direction / norm
    try:
        target.setAnalyzer(ba.R3(*(efficiency * unit)), transmission)
    except TypeError:
        target.setAnalyzer(ba.R3(*unit), efficiency, transmission)

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
    analyzer_transmission: float,
    masked_rectangles: Optional[List[Tuple[float, float, float, float]]] = None
) -> ba.ScatteringSimulation:
    """
    Create a simulation with outgoing_directions_horizontal and _vertical pixels covering the 4-element angle_range
    [horiz_min, horiz_max, vert_min, vert_max] in degrees. The bins inside masked_rectangles (phi_low, alpha_low,
    phi_high, alpha_high) [deg] are masked: BornAgain does not compute them (their intensity is 0).
    """
    beam = ba.Beam(p, wavelength*angstrom, alpha_i*deg)

    horiz_min, horiz_max, vert_min, vert_max = angle_range

    rand_deg_phi, rand_deg_alpha = _grid_jitter(angle_range, outgoing_directions_horizontal, outgoing_directions_vertical, rand_y, rand_z)

    detector = ba.SphericalDetector(
        outgoing_directions_horizontal, (horiz_min + rand_deg_phi)*deg, (horiz_max + rand_deg_phi)*deg,
        outgoing_directions_vertical, (vert_min + rand_deg_alpha)*deg, (vert_max + rand_deg_alpha)*deg
    )
    for phi_low, alpha_low, phi_high, alpha_high in masked_rectangles or []:
        detector.addMask(ba.Rectangle(phi_low*deg, alpha_low*deg, phi_high*deg, alpha_high*deg), True)
    if polarization:
        beam.setPolarization(ba.R3(*polarization))
        if analyzer_direction:
            set_analyzer(detector, analyzer_direction, analyzer_efficiency, analyzer_transmission)

    return ba.ScatteringSimulation(beam, sample, detector)

def get_simulation_specular(
    sample: Any,
    wavelength: float,
    alpha_i: float,
    use_avg_materials: bool = True,
    polarization: Optional[List[float]] = None,
    analyzer_direction: Optional[List[float]] = None,
    analyzer_efficiency: Optional[float] = None,
    analyzer_transmission: Optional[float] = None
) -> ba.SpecularSimulation:
    """
    Reflectivity of the sample for one particle (--specular specular_simulation): a SpecularSimulation with the
    options of the ScatteringSimulation of get_simulation (average materials, polarisation, analyzer), so that the
    reflected ray has the reflectivity that include_specular puts into its grid bin.
    """
    scan = ba.AlphaScan(2, alpha_i*deg, alpha_i*deg+1e-6)
    scan.setWavelength(wavelength*angstrom)
    if polarization:
        scan.setPolarization(ba.R3(*polarization))
        if analyzer_direction is not None:
            set_analyzer(scan, analyzer_direction, analyzer_efficiency, analyzer_transmission)
    simulation = ba.SpecularSimulation(scan, sample)
    simulation.options().setUseAvgMaterials(use_avg_materials)
    return simulation

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
    params: Dict[str, Any],
    angle_range: Optional[List[float]] = None,
    direction_filter: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Setup and run the BornAgain scattering simulation for a single particle, with its outgoing directions in
    angle_range (default: params['angle_range']). direction_filter(alpha_f, phi_f) -> bool array (phi x alpha), if
    given, selects the directions to simulate; the others are masked in BornAgain (weight 0).
    Returns:
        (weights, alpha_f_grid, phi_f_grid, number of skipped directions)
    """
    if angle_range is None:
        angle_range = params['angle_range']
    outgoing_directions_horizontal = params['outgoing_directions_horizontal']
    outgoing_directions_vertical = params['outgoing_directions_vertical']

    rand_y = 2*np.random.random()-1
    rand_z = 2*np.random.random()-1

    rand_deg_phi, rand_deg_alpha = _grid_jitter(angle_range, outgoing_directions_horizontal, outgoing_directions_vertical, rand_y, rand_z)
    # each intensity belongs to its bin centre; rows are ordered from the highest alpha_f downwards
    alpha_f, phi_f = _grid_centres(angle_range, outgoing_directions_horizontal, outgoing_directions_vertical, rand_deg_phi, rand_deg_alpha)

    masked_rectangles = None
    n_skipped = 0
    if direction_filter is not None:
        skip = ~direction_filter(alpha_f, phi_f)
        n_skipped = int(skip.sum())
        if skip.all():  # no direction can reach the region: nothing to simulate
            return np.zeros(skip.size), alpha_f, phi_f, n_skipped
        if n_skipped:
            masked_rectangles = _skipped_direction_rectangles(skip, alpha_f, phi_f)

    sim = get_simulation(
        sample_model, outgoing_directions_horizontal, outgoing_directions_vertical,
        angle_range, wavelength, alpha_i, p, rand_y, rand_z, polarization,
        analyzer_direction=params.get('analyzer_direction', [0,0,0]),
        analyzer_efficiency=params.get('analyzer_efficiency', 1.0),
        analyzer_transmission=params.get('analyzer_transmission', 0.5),
        masked_rectangles=masked_rectangles
    )
    sim.options().setUseAvgMaterials(params.get('use_avg_materials', True))
    sim.options().setIncludeSpecular(params.get('specular') == 'include_specular')
    bornagain_number_of_threads = params.get('bornagain_number_of_threads')
    if bornagain_number_of_threads is not None:
        sim.options().setNumberOfThreads(bornagain_number_of_threads)

    res = sim.simulate()
    pout = get_result_intensities(res)

    weights = pout.T.flatten()
    return weights, alpha_f, phi_f, n_skipped

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

def _update_tof_histogram(
    detector_id: np.ndarray,
    total_tof: np.ndarray,
    valid_weights: np.ndarray,
    tof_bin_edges: np.ndarray,
    tof_hist: np.ndarray,
    tof_hist_weights_squared: np.ndarray
) -> None:
    """
    Update the (pixel, TOF bin) histograms of the TOF histogram mode (params['tof_bin_edges'], mg_fit
    --wavelength_slice); hits outside the TOF bins are dropped.
    """
    tof_bin = np.searchsorted(tof_bin_edges, total_tof, side='right') - 1
    inside = (tof_bin >= 0) & (tof_bin < len(tof_bin_edges) - 1)
    if np.any(inside):
        np.add.at(tof_hist, (detector_id[inside], tof_bin[inside]), valid_weights[inside])
        np.add.at(tof_hist_weights_squared, (detector_id[inside], tof_bin[inside]), valid_weights[inside]**2)

def seed_particle_rng(random_seed: Optional[int], particle_index: int) -> None:
    """
    Seed numpy's global random generator for one incident particle from the run seed and the
    particle's global index. The random numbers used for a particle (outgoing-direction grid
    jitter, detector resolution smearing) then do not depend on how the particles are split
    between processes, so results are reproducible and identical for any number of processes.
    """
    if random_seed is not None:
        np.random.seed(np.random.SeedSequence([random_seed, particle_index]).generate_state(4))


def process_particles(particles: Any, params: Dict[str, Any], start_index: int = 0) -> Dict[str, Any]:
    """
    Carry out the BornAgain simulation and subsequent processing for a batch of incident particles.
    start_index is the global index of the first particle of the batch (for per-particle seeding).
    The outgoing directions of every particle cover params['angle_range'], or with params['angle_window_region']
    (mg_fit --simulate_mask_angle_range_factor auto) the particle's own window of directions that can reach that
    region (neutron_angle_windows, calculated here from the particles of the batch).
    TOF instruments: the detected neutrons are written to a temporary event file (detector ID, TOF, weight), or with
    params['tof_bin_edges'] (TOF bin edges [s], mg_fit --wavelength_slice) histogrammed per (pixel, TOF bin)
    ('tofHist', 'tofHistWeightsSquared', shape (number of pixels, number of TOF bins)) without an event file.
    """
    f_temp = None
    h5_temp_path = None
    try:
        sample = params['sample']
        sample_module = sample.get_module()
        sample_model = sample_module.get_sample(**sample.kwargs)
        instrument = params['instrument']
        is_tof = instrument.is_tof_instrument
        specular = params['specular']

        pixel_hist = None
        pixel_hist_weights_squared = None
        tof_hist = None
        tof_hist_weights_squared = None
        tof_bin_edges = params.get('tof_bin_edges') if is_tof else None
        event_buffer = None
        buffer_idx = 0
        buffer_capacity = 1_000_000
        random_seed = params.get('random_seed')
        window_region = params.get('angle_window_region')
        windows = neutron_angle_windows(window_region, particles, instrument) if window_region is not None else None
        # directions aimed only at pixels outside the region (raw-pixel mask) are masked in BornAgain (not computed)
        keep_pixels = params.get('direction_keep_pixels')
        reach_table = _reach_table(keep_pixels) if keep_pixels is not None else None
        n_directions, n_skipped = 0, 0

        if not is_tof:
            pixel_hist = np.zeros((instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus), dtype=np.float64)
            pixel_hist_weights_squared = np.zeros((instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus), dtype=np.float64)
        elif tof_bin_edges is not None:
            tof_bin_edges = np.asarray(tof_bin_edges, dtype=np.float64)
            tof_hist_shape = (instrument.detector.pixels_x_nexus * instrument.detector.pixels_y_nexus, len(tof_bin_edges) - 1)
            tof_hist = np.zeros(tof_hist_shape, dtype=np.float64)
            tof_hist_weights_squared = np.zeros(tof_hist_shape, dtype=np.float64)
        else:
            import h5py
            # unique file per call: a pool worker may process more than one batch
            fd, h5_temp_path = tempfile.mkstemp(prefix='mcstas_gisans_events_', suffix='.h5')
            os.close(fd)

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
                
            seed_particle_rng(random_seed, start_index + id)
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
                direction_filter = None
                if reach_table is not None:
                    direction_filter = lambda a_f, p_f: directions_reaching_region(instrument, reach_table, x, y, z, v, phi_i, a_f, p_f)
                weights, alpha_f_grid, phi_f_grid, skipped = _execute_bornagain_simulation(
                    sample_model, wavelength, alpha_i, p, polarization, params,
                    angle_range=windows[id] if windows is not None else None,
                    direction_filter=direction_filter
                )
                n_directions += weights.size
                n_skipped += skipped
                
                phi_f_total = phi_i + phi_f_grid
                alpha_grid_2d, phi_grid_2d = np.meshgrid(np.deg2rad(alpha_f_grid), np.deg2rad(phi_f_total))
                
                VX_grid = v * np.cos(alpha_grid_2d) * np.cos(phi_grid_2d)
                VY_grid = v * np.cos(alpha_grid_2d) * np.sin(phi_grid_2d)
                VZ_grid = v * np.sin(alpha_grid_2d)

                idx_x_nexus, idx_y_nexus, valid_mask, sd_tof = _calculate_nexus_hits(
                    instrument, x, y, z, t, VX_grid.flatten(), VY_grid.flatten(), VZ_grid.flatten()
                )

                if specular == 'specular_simulation':
                    # the specular reflection as one extra ray in the exact mirror direction (z is the surface
                    # normal here), with the reflectivity of the sample; the transmitted part (1 - reflectivity) is
                    # not added: it enters the substrate (e.g. a liquid) and does not reach the detector
                    ssim = get_simulation_specular(
                        sample_model, wavelength, alpha_i, params.get('use_avg_materials', True), polarization,
                        analyzer_direction=params.get('analyzer_direction', [0, 0, 0]),
                        analyzer_efficiency=params.get('analyzer_efficiency', 1.0),
                        analyzer_transmission=params.get('analyzer_transmission', 0.5)
                    )
                    if params.get('bornagain_number_of_threads') is not None:
                        ssim.options().setNumberOfThreads(params['bornagain_number_of_threads'])
                    refl_fraction = np.array(ssim.simulate().flatVector())[0]

                    idx_x_refl, idx_y_refl, valid_refl, sd_tof_refl = _calculate_nexus_hits(
                        instrument, x, y, z, t, np.array([vx]), np.array([vy]), np.array([-vz])
                    )
                    
                    idx_y_list = [idx_y_nexus, idx_y_refl]
                    idx_x_list = [idx_x_nexus, idx_x_refl]
                    valid_list = [valid_mask, valid_refl]
                    sd_tof_list = [sd_tof, sd_tof_refl]
                    weights_pixel = [weights, np.array([p * refl_fraction])]

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
                elif tof_hist is not None:
                    _update_tof_histogram(
                        detector_id, t + sd_tof[valid_mask], valid_weights, tof_bin_edges,
                        tof_hist, tof_hist_weights_squared
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

        if f_temp is not None:
            f_temp.close()
            f_temp = None
        return {
            'pixelHist': pixel_hist,
            'pixelHistWeightsSquared': pixel_hist_weights_squared,
            'tofHist': tof_hist,
            'tofHistWeightsSquared': tof_hist_weights_squared,
            'temp_h5_path': h5_temp_path,
            'directions': n_directions,
            'directions_skipped': n_skipped
        }
    except Exception:
        # the exception propagates to the caller (multiprocessing re-raises it in the parent);
        # remove this batch's partial temporary file
        if f_temp is not None:
            f_temp.close()
        if h5_temp_path is not None and os.path.exists(h5_temp_path):
            os.remove(h5_temp_path)
        raise

_WORKER_PARAMS: Optional[List[Dict[str, Any]]] = None


def _set_worker_params(params_list: List[Dict[str, Any]]) -> None:
    """Initializer of the worker processes: the parameters are sent once per process, not with every chunk."""
    global _WORKER_PARAMS
    _WORKER_PARAMS = params_list


def _process_member_chunk_local(particles: Any, start_index: int, member: int) -> Dict[str, Any]:
    return process_particles(particles, _WORKER_PARAMS[member], start_index)


def _process_member_chunk(particles: Any, params_list: List[Dict[str, Any]], start_index: int, member: int) -> Dict[str, Any]:
    """A chunk of the neutrons with the parameter set `member` of a batch (several nodes, cluster.Coordinator.map)."""
    return process_particles(particles, params_list[member], start_index)


def process_particles_parallelly(particles: Any, params: Dict[str, Any], process_number: int,
                                 chunks_per_process: int = 4) -> Dict[str, Any]:
    """
    Spawn parallel processes to carry out the BornAgain simulation and subsequent
    calculation of the incident particles.

    The neutrons are split into chunks_per_process chunks per process; a process takes the next chunk when it becomes
    free, so that neutrons of different cost (e.g. different numbers of outgoing directions) are balanced. The result
    does not depend on the chunking (every neutron is seeded by its global index).
    """
    from .cluster import get_coordinator
    if get_coordinator() is None:
        process_number = max(1, min(int(process_number), len(particles)))
        print(f"Number of parallel processes: {process_number} (number of physical CPU cores: {get_available_cores()})")
        if process_number == 1:
            result = process_particles(particles, params)
            if params['instrument'].is_tof_instrument:
                temp_h5_path = result.pop('temp_h5_path')
                result['temp_h5_paths'] = [temp_h5_path] if temp_h5_path else []
            return result
    return process_batch_parallelly(particles, [params], process_number, chunks_per_process)[0]


def process_batch_parallelly(particles: Any, params_list: List[Dict[str, Any]], process_number: int,
                             chunks_per_process: int = 4) -> List[Dict[str, Any]]:
    """
    Simulate the same neutrons with several parameter sets at once (e.g. the members of a generation of
    differential evolution, mg_fit --de_parallel): the chunks of all parameter sets go to one pool of processes (of all
    nodes with --cluster_nodes), about chunks_per_process chunks per process in total, so that no process waits for
    the end of one parameter set before it starts the next. Returns one merged result per parameter set, each the same
    as from process_particles_parallelly with that parameter set alone.
    """
    from .cluster import get_coordinator
    coordinator = get_coordinator()
    n_members = len(params_list)
    if coordinator is not None:
        if any(p['instrument'].is_tof_instrument and p.get('tof_bin_edges') is None for p in params_list):
            raise ValueError("--cluster_nodes: TOF event files are not supported on several nodes (the temporary event "
                             "files would stay on the worker nodes); use one node, or mg_fit with --wavelength_slice.")
        total_processes = coordinator.total_processes
    else:
        total_processes = process_number = max(1, int(process_number))
    # chunks of every parameter set: chunks_per_process x (processes of all nodes) chunks in total
    n_chunks = max(1, min(len(particles), -(-max(1, int(chunks_per_process)) * total_processes // n_members)))
    starts = np.linspace(0, len(particles), n_chunks + 1).astype(int)
    tasks = [(particles[starts[i]:starts[i + 1]], int(starts[i]), member) for member in range(n_members) for i in range(n_chunks)]

    if coordinator is not None:
        # the parameter sets go to every node once (shared), the chunk tasks carry only the neutrons
        results = coordinator.map(_process_member_chunk, tasks, shared=params_list, shared_index=1)
    elif process_number == 1:
        results = [process_particles(chunk, params_list[member], start) for chunk, start, member in tasks]
    else:
        # A worker that raises passes its exception on (f.result()); a worker that is killed (e.g. by the
        # OOM killer) breaks the pool, which raises BrokenProcessPool. (multiprocessing.Pool replaces a
        # killed worker and waits forever for its lost task.)
        executor = ProcessPoolExecutor(max_workers=min(process_number, len(tasks)), initializer=_set_worker_params,
                                       initargs=(params_list,))
        futures = [executor.submit(_process_member_chunk_local, *task) for task in tasks]
        try:
            results = [f.result() for f in futures]
        except BaseException as e:
            executor.shutdown(wait=True, cancel_futures=True)
            for f in futures:  # temporary TOF event files of the workers that did finish
                if f.done() and not f.cancelled() and f.exception() is None and f.result().get('temp_h5_path'):
                    if os.path.exists(f.result()['temp_h5_path']):
                        os.remove(f.result()['temp_h5_path'])
            if isinstance(e, BrokenProcessPool):
                raise RuntimeError("A parallel process was terminated abruptly (e.g. killed by the out-of-memory killer "
                                   "or a signal); the run is stopped. Fewer --parallel_processes need less memory.") from e
            raise
        executor.shutdown(wait=True)
    return [_merge_results(results[member * n_chunks:(member + 1) * n_chunks], params_list[member]) for member in range(n_members)]


def _merge_results(results: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """Sum the results of the chunks of process_particles_parallelly (in chunk order)."""
    is_tof = params['instrument'].is_tof_instrument
    pixel_hist = np.zeros((params['instrument'].detector.pixels_x_nexus, params['instrument'].detector.pixels_y_nexus))
    pixel_hist_weights_squared = np.zeros((params['instrument'].detector.pixels_x_nexus, params['instrument'].detector.pixels_y_nexus))
    
    for process_result in results:
        if process_result['pixelHist'] is not None:
            pixel_hist += process_result['pixelHist']
            pixel_hist_weights_squared += process_result['pixelHistWeightsSquared']
            
    result: Dict[str, Any] = {
        'pixelHist': pixel_hist,
        'pixelHistWeightsSquared': pixel_hist_weights_squared,
        'directions': sum(r.get('directions', 0) for r in results),
        'directions_skipped': sum(r.get('directions_skipped', 0) for r in results)
    }
    tof_hists = [r['tofHist'] for r in results if r.get('tofHist') is not None]
    if tof_hists:  # TOF histogram mode (params['tof_bin_edges'])
        result['tofHist'] = np.sum(tof_hists, axis=0)
        result['tofHistWeightsSquared'] = np.sum([r['tofHistWeightsSquared'] for r in results if r.get('tofHist') is not None], axis=0)

    if is_tof:
        result['temp_h5_paths'] = [r['temp_h5_path'] for r in results if r.get('temp_h5_path')]
        
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
    if len(particles) == 0:
        sys.exit("ERROR: no incident particles are left to simulate (none hit the sample). Check --alpha, the sample size "
                 "and --sample_orientation, or use --allow_sample_miss to propagate missing particles to the detector.")

    set_outgoing_directions_from_sampling(args, particles, particle_type)  # only with --sampling/--rays_per_pixel
    if args.outgoing_directions is not None:
        suffix = args.outgoing_directions
    else:
        suffix = f"{args.outgoing_directions_horizontal}_{args.outgoing_directions_vertical}"
    savename = f"q_events_pix{suffix}" if args.savename == '' else args.savename
    print('Number of particles being processed: ', len(particles))

    params = pack_parameters(args, particle_type)
    from .cluster import setup_from_args
    setup_from_args(args)  # --cluster_nodes > 1: wait for the worker nodes

    if args.no_parallel:
        result = process_particles(particles, params)
        if params['instrument'].is_tof_instrument:
            result['temp_h5_paths'] = [result.pop('temp_h5_path')]
    else:
        process_number = args.parallel_processes if args.parallel_processes else max(1, get_available_cores() - 1)
        result = process_particles_parallelly(particles, params, process_number, getattr(args, 'chunks_per_process', 4))

    save_simulation_results_as_scipp(savename, params, result, args, mcpl_metadata, getattr(args, 'temp_read_chunk_size', 1000000))

if __name__=='__main__':
    main()
