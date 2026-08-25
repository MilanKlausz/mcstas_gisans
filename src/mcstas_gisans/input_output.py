"""
Read particles from MCPL files (or .dat file)
Read and write Scipp DataArrays from/to HDF5 files
"""

import numpy as np
import mcpl

from .particle_calculations import get_particle_converter

def print_tof_limits(tof_limits):
  """Small utility function to print out TOF limits used for filtering"""
  if tof_limits == [float('-inf'), float('inf')]:
    print(f"No TOF filtering applied. Processing all particles from the input file.")
  else:
    print(f"Using MCPL input TOF limits: : {tof_limits[0]:.3f} - {tof_limits[1]:.3f} [millisecond]")

def get_particles(filename, intensity_factor, tof_limits, input_weight_limit, use_polarization):
  """
  Read particles from an MCPL file or a .dat file (created with the
  Virtual_output McStas component). In case of an MCPL file, convert units and
  apply filtering based on the time (TOF) field using the provided limits.
  """
  print(f'Reading particles from {filename}...')
  print_tof_limits(tof_limits)
  if filename.endswith('.mcpl') or filename.endswith('.mcpl.gz'):
    with mcpl.MCPLFile(filename) as myfile:
      convert_particle_properties, particle_type = get_particle_converter(myfile.opt_universalpdgcode)
      particles = np.array(
        [convert_particle_properties(p, intensity_factor, use_polarization=use_polarization) for p in myfile.particles
         if (p.weight > input_weight_limit and
             tof_limits[0] < p.time and p.time < tof_limits[1])]
        )
  elif filename.endswith('.dat'): #legacy file type
    particles = np.loadtxt(filename)
  else:
    import sys
    sys.exit("Wrong input file extension. Expected: '.mcpl', '.mcpl.gz' or '.dat'" )
  if len(particles)==0:
    import sys
    if tof_limits != [float('-inf'), float('inf')]:
      sys.exit(f"No particles found in the input file ({filename}) within the TOF filtering limits: {tof_limits[0]:.3f} - {tof_limits[1]:.3f} [millisecond]!")
    else:
      sys.exit(f"No particles found in the input file {filename}.")
  return particles, particle_type

def save_scipp_file(savename, scipp_da):
  """Save Scipp DataArray into an HDF5 file"""
  import scipp as sc
  filename = savename if savename.endswith('.h5') else f"{savename}.h5"
  sc.io.hdf5.save_hdf5(scipp_da, filename)
  print(f"Created {filename} (Scipp format)")

def load_scipp_file(filename):
  """Load Scipp DataArray from an HDF5 file (.h5)"""
  import scipp as sc
  import h5py

  print(f"Loading {filename}...")
  with h5py.File(filename, 'r') as f:
      if 'detector_id' in f and 'tof' in f and 'weight' in f:
          print("Detected custom TOF event format. Reading arrays...")
          # This is our custom TOF event format (legacy)
          n = f['detector_id'].shape[0]
          det_ids = np.empty(n, dtype=np.int32)
          tofs = np.empty(n, dtype=np.float64)
          weights = np.empty(n, dtype=np.float64)
          chunk_size = 1000000
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

          if 'pixel_positions' in f:
              print("Reading pixel_positions...")
              pixel_positions = f['pixel_positions'][:]
              num_pixels = len(pixel_positions)
              print(f"Binning into {num_pixels} detectors...")
              # Bin into detectors
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

def save_simulation_results_as_scipp(savename, params, result, chunk_size=1000000):
  """
  Saves the output of a McStas-GISANS simulation (from mg_run or mg_fit) as a Scipp DataArray container.
  For TOF instruments, it consolidates temporary HDF5 event files into a native Scipp HDF5 file.
  For non-TOF instruments, it flattens the pixel histograms into a Scipp DataArray.
  """
  import h5py
  import os
  import scipp as sc

  if params['instrument'].is_tof_instrument:
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

    print("Creating native Scipp DataArray...")
    da = sc.DataArray(
        data=sc.array(dims=['event'], values=weights, variances=weights, unit='counts'),
        coords={
            'detector_id': sc.array(dims=['event'], values=det_ids, unit=None),
            'tof': sc.array(dims=['event'], values=tofs, unit='s')
        }
    )

    num_pixels = params['instrument'].detector.pixels_x_nexus * params['instrument'].detector.pixels_y_nexus
    binned = da.bin(detector_id=sc.arange('detector_id', 0, num_pixels + 1, unit=None))

    # Add geometry metadata
    positions = params['instrument'].detector.get_pixel_positions(params['instrument'].sample_detector_distance)
    coords = {
        'position': sc.vectors(dims=['detector_id'], values=positions, unit='m'),
        'sample_position': sc.vector(value=[0, 0, 0], unit='m'),
        'source_position': sc.vector(value=[0, 0, -params['instrument'].nominal_source_sample_distance], unit='m'),
        'is_tof_instrument': sc.scalar(params['instrument'].is_tof_instrument),
        'wavelength_selected': sc.scalar(params['instrument'].wavelength_selected if params['instrument'].wavelength_selected is not None else 0.0, unit='angstrom'),
        'alpha_inc_deg': sc.scalar(np.rad2deg(params['instrument'].alpha_inc), unit='deg'),
        'instrument_name': sc.scalar(params['instrument_name']),
        'sample_orientation': sc.scalar(params['instrument'].detector.sample_orientation),
        'beam_angle': sc.scalar(params['instrument'].beam_angle, unit='deg'),
        'instrument_detector_centre_offset_x': sc.scalar(params['instrument'].detector.direct_beam_centre_offset_x_nexus, unit='m'),
        'instrument_detector_centre_offset_y': sc.scalar(params['instrument'].detector.direct_beam_centre_offset_y_nexus, unit='m'),
    }
    for k, v in coords.items():
        binned.coords[k] = v

    sc.io.hdf5.save_hdf5(binned, final_h5)
    print(f"Created {final_h5} (Native Scipp Format)")
  else:
    scipp_da = params['instrument'].create_scipp_container()
    scipp_da.values = result['pixelHist'].flatten()
    scipp_da.variances = result['pixelHistWeightsSquared'].flatten()
    scipp_da.coords['instrument_name'] = sc.scalar(params['instrument_name'])
    scipp_da.coords['instrument_detector_centre_offset_x'] = sc.scalar(params['instrument'].detector.direct_beam_centre_offset_x_nexus, unit='m')
    scipp_da.coords['instrument_detector_centre_offset_y'] = sc.scalar(params['instrument'].detector.direct_beam_centre_offset_y_nexus, unit='m')
    save_scipp_file(savename, scipp_da)
    print("Sum intensity in the scipp pixel-histogram: ", np.sum(result['pixelHist']))
