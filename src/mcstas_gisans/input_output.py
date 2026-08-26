"""
Read particles from MCPL files (or .dat file)
Read and write Scipp DataArrays from/to HDF5 files
"""

import numpy as np
import mcpl
from typing import List, Tuple, Any, Dict

from .particle_calculations import get_particle_converter


def print_tof_limits(tof_limits: List[float]) -> None:
    """
    Small utility function to print out TOF limits used for filtering.

    Parameters
    ----------
    tof_limits : List[float]
        A list of two floats representing the minimum and maximum time-of-flight (TOF) limits.
        Use [-inf, inf] for no filtering.
    """
    if tof_limits == [float('-inf'), float('inf')]:
        print("No TOF filtering applied. Processing all particles from the input file.")
    else:
        print(f"Using MCPL input TOF limits: {tof_limits[0]:.3f} - {tof_limits[1]:.3f} [millisecond]")


def get_particles(filename: str, intensity_factor: float, tof_limits: List[float], input_weight_limit: float, use_polarization: bool) -> Tuple[np.ndarray, Any]:
    """
    Read particles from an MCPL file or a .dat file (created with the Virtual_output McStas component).

    In case of an MCPL file, it converts units and applies filtering based on the time (TOF) field
    using the provided limits.

    Parameters
    ----------
    filename : str
        Path to the input particle file (e.g., .mcpl, .mcpl.gz, or .dat).
    intensity_factor : float
        A scaling factor applied to the intensity of the particles.
    tof_limits : List[float]
        A list of two floats representing the minimum and maximum TOF limits in milliseconds.
    input_weight_limit : float
        Minimum weight threshold for particles to be included.
    use_polarization : bool
        If True, includes polarization data when converting particles.

    Returns
    -------
    Tuple[np.ndarray, Any]
        - A NumPy array containing the parsed and converted particle properties.
        - The particle type information extracted from the file (None for .dat files).
    """
    import sys  # Imported locally to keep top-level namespace clean

    print(f'Reading particles from {filename}...')
    print_tof_limits(tof_limits)
    
    mcpl_metadata = None
    if filename.endswith('.mcpl') or filename.endswith('.mcpl.gz'):
        with mcpl.MCPLFile(filename) as myfile:
            # Determine how to convert properties based on particle type
            convert_particle_properties, particle_type = get_particle_converter(myfile.opt_universalpdgcode)
            
            # Read and filter particles, applying limits and intensity scaling
            particles = np.array([
                convert_particle_properties(p, intensity_factor, use_polarization=use_polarization) 
                for p in myfile.particles
                if (p.weight > input_weight_limit and tof_limits[0] < p.time and p.time < tof_limits[1])
            ])
            mcpl_metadata = {
                'sourcename': myfile.sourcename,
                'nparticles': myfile.nparticles,
                'comments': myfile.comments
            }
    elif filename.endswith('.dat'): 
        # Handle legacy .dat file type
        particles = np.loadtxt(filename)
        particle_type = None  # .dat files don't carry PDG codes
    else:
        sys.exit("Wrong input file extension. Expected: '.mcpl', '.mcpl.gz' or '.dat'")
        
    # Check if any particles matched the criteria and abort early if none found
    if len(particles) == 0:
        if tof_limits != [float('-inf'), float('inf')]:
            sys.exit(f"No particles found in the input file ({filename}) within the TOF filtering limits: {tof_limits[0]:.3f} - {tof_limits[1]:.3f} [millisecond]!")
        else:
            sys.exit(f"No particles found in the input file {filename}.")
            
    return particles, particle_type, mcpl_metadata


def save_scipp_file(savename: str, scipp_da: Any) -> None:
    """
    Save a Scipp DataArray into an HDF5 file.

    Parameters
    ----------
    savename : str
        The target filename. If it doesn't end with '.h5', the extension is appended.
    scipp_da : Any
        The Scipp DataArray object to be saved.
    """
    import scipp as sc
    filename = savename if savename.endswith('.h5') else f"{savename}.h5"
    sc.io.hdf5.save_hdf5(scipp_da, filename)
    print(f"Created {filename} (Scipp format)")


def load_scipp_file(filename: str) -> Any:
    """
    Load a Scipp DataArray from an HDF5 file (.h5).

    Supports both standard Scipp HDF5 format and a custom legacy TOF event format.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file to load.

    Returns
    -------
    Any
        The loaded Scipp DataArray containing the event or histogram data.
    """
    import scipp as sc
    import h5py

    print(f"Loading {filename}...")
    with h5py.File(filename, 'r') as f:
        # Check for the custom TOF event format
        if 'detector_id' in f and 'tof' in f and 'weight' in f:
            print("Detected custom TOF event format. Reading arrays...")
            
            n = f['detector_id'].shape[0]
            det_ids = np.empty(n, dtype=np.int32)
            tofs = np.empty(n, dtype=np.float64)
            weights = np.empty(n, dtype=np.float64)
            
            chunk_size = 1000000
            # Read data in chunks to prevent excessive memory usage with large arrays
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                det_ids[i:end] = f['detector_id'][i:end]
                tofs[i:end] = f['tof'][i:end]
                weights[i:end] = f['weight'][i:end]

            print("Creating Scipp DataArray...")
            da = sc.DataArray(
                data=sc.array(dims=['event'], values=weights, variances=weights, unit='counts'),
                coords={
                    'detector_id': sc.array(dims=['event'], values=det_ids, unit=None),
                    'tof': sc.array(dims=['event'], values=tofs, unit='s')
                }
            )

            # Reconstruct pixel positions and bin events if geometry data is available
            if 'pixel_positions' in f:
                print("Reading pixel_positions...")
                pixel_positions = f['pixel_positions'][:]
                num_pixels = len(pixel_positions)
                
                print(f"Binning into {num_pixels} detectors...")
                # Bin events back into specific detectors
                binned = da.bin(detector_id=sc.arange('detector_id', 0, num_pixels + 1, unit=None))
                
                print("Adding geometry metadata...")
                binned.coords['position'] = sc.vectors(dims=['detector_id'], values=pixel_positions, unit='m')
                
                if 'sample_position' in f:
                    binned.coords['sample_position'] = sc.vector(value=f['sample_position'][:], unit='m')
                if 'source_position' in f:
                    binned.coords['source_position'] = sc.vector(value=f['source_position'][:], unit='m')
                    
                print("Finished loading.")
                return binned
            else:
                return da
        else:
            print("Loading standard Scipp HDF5 format...")
            return sc.io.hdf5.load_hdf5(filename)


def save_simulation_results_as_scipp(savename: str, params: Dict[str, Any], result: Dict[str, Any], args: Any, mcpl_metadata: Any = None, chunk_size: int = 1000000) -> None:
    """
    Saves the output of a McStas-GISANS simulation as a Scipp DataGroup container.
    """
    import h5py
    import os
    import scipp as sc
    import json
    import sys
    import datetime
    import mcpl
    import bornagain

    # Prepare instrument metadata
    inst = params['instrument']
    instrument_metadata = {
        'name': sc.scalar(params['instrument_name']),
        'is_tof_instrument': sc.scalar(inst.is_tof_instrument),
        'detector_centre_offset_x': sc.scalar(inst.detector.direct_beam_centre_offset_x_nexus, unit='m'),
        'detector_centre_offset_y': sc.scalar(inst.detector.direct_beam_centre_offset_y_nexus, unit='m'),
        'alpha_inc_deg': sc.scalar(np.rad2deg(inst.alpha_inc), unit='deg'),
        'beam_angle': sc.scalar(inst.beam_angle, unit='deg'),
        'sample_orientation': sc.scalar(inst.detector.sample_orientation),
        'sample_position': sc.vector(value=[0, 0, 0], unit='m'),
        'source_position': sc.vector(value=[0, 0, -inst.nominal_source_sample_distance], unit='m')
    }
    if not inst.is_tof_instrument:
        instrument_metadata['wavelength_selected'] = sc.scalar(inst.wavelength_selected if inst.wavelength_selected is not None else 0.0, unit='angstrom')

    # Prepare sample metadata
    import inspect
    sample_module = params['sample'].get_module()
    try:
        if hasattr(sample_module, '__file__') and sample_module.__file__:
            with open(sample_module.__file__, 'r') as f:
                sample_content = f.read()
        else:
            sample_content = inspect.getsource(sample_module)
    except Exception:
        sample_content = "Could not extract sample source."

    sample_metadata = {
        'name': sc.scalar(getattr(args, 'sample', getattr(args, 'model', 'unknown'))),
        'arguments_json': sc.scalar(json.dumps(getattr(args, 'sample_args', getattr(args, 'sample_arguments', '')))),
        'script_content': sc.scalar(sample_content)
    }

    # Prepare MCPL metadata
    mcpl_group = None
    if mcpl_metadata:
        mcpl_group = sc.DataGroup({
            'filename': sc.scalar(getattr(args, 'filename', 'unknown')),
            'sourcename': sc.scalar(mcpl_metadata.get('sourcename', '')),
            'nparticles': sc.scalar(mcpl_metadata.get('nparticles', 0)),
            'comments': sc.scalar("\n".join(mcpl_metadata.get('comments', [])))
        })

    # Prepare provenance metadata
    # Try to safely get mcstas_gisans version
    try:
        from . import __version__ as mg_version
    except ImportError:
        mg_version = 'unknown'

    provenance_metadata = {
        'cli_command': sc.scalar(" ".join(sys.argv)),
        'cli_args_json': sc.scalar(json.dumps(vars(args))),
        'bornagain_version': sc.scalar(str(getattr(bornagain, 'version', 'unknown'))),
        'mcstas_gisans_version': sc.scalar(mg_version),
        'mcpl_version': sc.scalar(mcpl.__version__),
        'timestamp': sc.scalar(datetime.datetime.now().isoformat())
    }

    if inst.is_tof_instrument:
        final_h5 = savename if savename.endswith('.h5') else f"{savename}.h5"
        temp_files = result.get('temp_h5_paths', [result.get('temp_h5_path')])
        temp_files = [tf for tf in temp_files if tf]

        total_events = 0
        for tf in temp_files:
            with h5py.File(tf, 'r') as fin:
                if 'detector_id' in fin:
                    total_events += fin['detector_id'].shape[0]

        det_ids = np.empty(total_events, dtype=np.int32)
        tofs = np.empty(total_events, dtype=np.float64)
        weights = np.empty(total_events, dtype=np.float64)

        current_idx = 0
        for tf in temp_files:
            try:
                with h5py.File(tf, 'r') as fin:
                    if 'detector_id' in fin:
                        n = fin['detector_id'].shape[0]
                        hits_processed = 0
                        while hits_processed < n:
                            c = min(chunk_size, n - hits_processed)
                            det_ids[current_idx:current_idx+c] = fin['detector_id'][hits_processed:hits_processed+c]
                            tofs[current_idx:current_idx+c] = fin['tof'][hits_processed:hits_processed+c]
                            weights[current_idx:current_idx+c] = fin['weight'][hits_processed:hits_processed+c]
                            hits_processed += c
                            current_idx += c
            finally:
                if os.path.exists(tf):
                    os.remove(tf)

        da = sc.DataArray(
            data=sc.array(dims=['event'], values=weights, variances=weights, unit='counts'),
            coords={
                'detector_id': sc.array(dims=['event'], values=det_ids, unit=None),
                'tof': sc.array(dims=['event'], values=tofs, unit='s')
            }
        )

        num_pixels = inst.detector.pixels_x_nexus * inst.detector.pixels_y_nexus
        binned = da.bin(detector_id=sc.arange('detector_id', 0, num_pixels + 1, unit=None))
        positions = inst.detector.get_pixel_positions(inst.sample_detector_distance)
        binned.coords['position'] = sc.vectors(dims=['detector_id'], values=positions, unit='m')
        
        main_data = binned
    else:
        num_pixels = inst.detector.pixels_x_nexus * inst.detector.pixels_y_nexus
        positions = inst.detector.get_pixel_positions(inst.sample_detector_distance)
        coords = {
            'position': sc.vectors(dims=['detector_id'], values=positions, unit='m'),
        }
        main_data = sc.DataArray(
            data=sc.array(dims=['detector_id'], values=result['pixelHist'].flatten(), variances=result['pixelHistWeightsSquared'].flatten(), unit='counts'),
            coords=coords
        )

    # Assemble final DataGroup
    dataset_dict = {
        'data': main_data,
        'instrument': sc.DataGroup(instrument_metadata),
        'sample': sc.DataGroup(sample_metadata),
        'provenance': sc.DataGroup(provenance_metadata)
    }
    if mcpl_group is not None:
        dataset_dict['mcpl'] = mcpl_group

    dataset = sc.DataGroup(dataset_dict)
    
    final_h5 = savename if savename.endswith('.h5') else f"{savename}.h5"
    sc.io.hdf5.save_hdf5(dataset, final_h5)
    print(f"Created {final_h5} (Scipp DataGroup Format)")
    if not inst.is_tof_instrument:
        print("Sum intensity in the scipp pixel-histogram: ", np.sum(result['pixelHist']))
