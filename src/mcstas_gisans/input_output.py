"""
Read particles from MCPL files (or .dat file)
Output Q histogram files (or raw Q list files)
Unpack the Q histogram files
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

def save_q_histogram_file(savename, q_hist, q_hist_error, edges):
  """Save the histograms are corresponding bin edges in an NPZ file"""
  np.savez_compressed(savename, hist=q_hist, error=q_hist_error, xEdges=edges[0], yEdges=edges[1], zEdges=edges[2])
  print(f"Created {savename}.npz")

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
              
  print("Loading standard Scipp HDF5 format...")
  return sc.io.hdf5.load_hdf5(filename)

def unpack_q_histogram_file(np_file):
  """Unpack the content (histograms are corresponding bin edges) of an NPZ file
  created with the save_q_histogram_file function"""
  hist = np_file['hist'] #keys are hardcoded in the save_q_histogram_file function
  hist_error = np_file['error']
  xEdges = np_file['xEdges']
  yEdges = np_file['yEdges']
  zEdges = np_file['zEdges']
  return hist, hist_error, xEdges, yEdges, zEdges

def save_raw_q_list_file(savename, qArray):
  """Save the list of Q events in an NPZ file"""
  np.savez_compressed(savename, qArray=qArray)
  print(f"Created {savename}.npz with raw Q events.")

def unpack_raw_q_list_file(np_file):
  """Unpack the content (list of Q events) of an NPZ file created with the
  save_raw_q_list_file function"""
  np_file_array_key = np_file.files[0] #might as well use hardcoded 'qArray' key from save_raw_q_list_file function
  qArray = np_file[np_file_array_key]
  weights = qArray[:, 0]
  x = qArray[:, 1]
  y = qArray[:, 2]
  z = qArray[:, 3]
  return x, y, z, weights
