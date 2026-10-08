# Changelog

All notable changes to `mcstas_gisans` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Changes since 2.0.0. This release contains breaking changes of the output
format, command line options, the Q convention and the detector offset:
results and scripts made with 2.0.0 must be revisited (see *Changed* and
*Removed*).

### Added
- `mg_fit`: parameter scans (`--scan`) and automated fits (`--fit`) of sample
  model parameters against measured NeXus data, with Nelder-Mead, Powell and
  Differential Evolution optimizers, Q-space masks (`--mask_*`, `--mask_view`),
  integer parameters (`--fit_integer`), joint fits of two measurements
  (`--nxs2`, `--fit2`, `--fit_common`), summed multi-file measurements
  (several `--nxs` files), CSV summaries, comparison plots and fit animations
  (`--png`, `--gif`). The default loss, a Poisson deviance with the Monte Carlo
  uncertainty of the simulation folded in, is unbiased also at low counts. A
  warning is printed if the Monte Carlo variance exceeds the counting variance
  in more than 5% of the pixels, or if it lowers the loss of the fit by more
  than 10% (with the loss with and without it).
- `mg_fit --simulate_mask_angle_range`: only the outgoing directions that can
  reach the unmasked pixels are simulated. With
  `--simulate_mask_angle_range_factor auto` (default) every neutron hitting the
  sample gets its own window: the range enclosing the unmasked pixels, shifted
  by the neutron's incident horizontal direction, hit point and gravity drop,
  plus 2 sigma of the detector resolution. No ray reaching the region is lost
  without resolution smearing (about 0.01% with it), and the directions are
  denser than with one range containing all windows. A numeric factor instead
  scales the range enclosing the unmasked pixels about its centre, for all
  neutrons, and a warning is printed if it does not contain all windows.
- `mg_fit --fit_background`: the flat background is fitted for each
  evaluation (a 1D minimisation of the loss, no extra simulation), printed,
  stored in the summary CSV and reported for the best evaluation; cannot be
  combined with a fixed `--background` (now unset = 0 by default).
- `mg_beam_centre_correction`: detector position from a direct beam
  measurement, and optional cross-checks: ray-traced McStas direct beam compared
  with the measurement, intensity factor (`--mcpl`, `--experiment_time`,
  `--figure png|pdf|show`), and the incident angle measured from the specular
  spot (`--sample_nxs`, `--alpha`).
- Time-of-flight instruments: event-mode output with the TOF of every event;
  wavelength slicing in `mg_plot`.
- `--split_view` (`mg_plot`, and the `mg_fit` comparison plots): one 2D Q map
  with the measurement for Qy < 0 and the simulation for Qy > 0, on a common
  colour scale; with `mg_plot --overlay` the 1D Qy slices of both are shown
  below. In figures with several panels the 2D maps share their axes and the 1D
  slices their Qy axis, so zooming one panel zooms all; a 1D slice below a
  single map has the width of the map.
- Sampling presets for the outgoing directions: `--sampling quick|standard|long`
  (about 100 / 2000 / 10000 rays per detector pixel, from the effective number
  of neutrons in the MCPL file; `quick` is the default) and `--rays_per_pixel`
  for any other target; the chosen grid is printed with the options that
  reproduce it.
- `--seed`: reproducible simulations, independent of the number of parallel
  processes; `mg_fit` uses one seed for all evaluations.
- `--instrument_*` options to override every instrument parameter (detector
  size, pixels, resolution, offset, distances, beam angle, monitors).
- `--nxs_data_path` for NeXus files with other layouts; warning if
  `--experiment_time` differs from the duration stored in the NeXus file.
- `--bornagain_number_of_threads`, experimental polarisation options
  (`--use_polarization`, `--analyzer_*`).
- Output metadata: the resolved instrument parameters, sample model source,
  command line, random seed, software versions and git commit, MCPL file path
  and size.
- Built-in sample models `gisans_model_air_d2o_interface` and
  `gisans_model_air_d2o_interface_microgel` (silica nanoparticles and microgels
  at the air/D2O interface, by Nicolo Paracini), with the parameters of the
  current fits of the D22 liquid-sample measurements as defaults.
- Apptainer definitions for BornAgain 23 and 24 images with scipp
  (`resources/apptainer/`) and a script that builds them without root
  privileges.
- Documentation: quickstart, replication guide, fitting guide, technical
  details (coordinate systems, Q convention and gravity, T0 correction),
  output format, CLI and API reference.
- Tests: physics invariants for all sample orientations (gravity direction,
  direct beam at Q = 0, specular position, detector image orientation),
  regression tests on the D22 paper data.

### Changed
- **Dependencies**: BornAgain 22 or 23 (`bornagain>=22.0,<24`; 23.0 is the default in `conda.yml`;
  BornAgain 24 changed the material and layout API and is not supported yet;
  PyPI provides Linux wheels, on macOS a locally built wheel is needed, see
  `INSTALL.md`); NumPy 2 is supported. scipp, scippneutron, h5py, matplotlib
  and scipy are required.
- **Output format**: `mg_run` writes a Scipp HDF5 file (`.h5`) with the
  simulated detector image (rates with Monte Carlo variances, per pixel) and
  metadata, instead of a Q histogram in an `.npz` file. Q is calculated from the
  pixels by `mg_plot`/`mg_fit`, exactly as for measured data. Legacy `.npz`
  files can still be plotted; they carry no instrument metadata, so the
  measured data shown with them need the instrument options on the command line
  in the new conventions, e.g. for a vertical sample (D22 paper data)
  `--sample_orientation 2 --instrument_detector_centre_offset 0.290838 -0.016061`
  (an offset of the old convention puts the measured beam off the plot).
- **Q convention with gravity**: the incident direction is the straight beam
  axis and the outgoing direction the launch direction of the neutron (the
  detection point raised by the gravity drop at that wavelength), as in Mantid
  and scippneutron/esssans. The measured direct beam is at Q = 0 and the
  specular reflection at Qz = 2k sin(alpha). Previously gravity was counted
  twice.
- **Detector offset** (`direct_beam_centre_offset`,
  `--instrument_detector_centre_offset`): now the position of the detector
  centre relative to the undeflected beam axis in the NeXus frame, the same for
  every sample orientation (instrument key `direct_beam_centre_offset`, formerly
  `centre_offset`). Offsets determined with 2.0.0 must be recalculated
  with `mg_beam_centre_correction` (D22 paper data: `0.290838 -0.016061`). The
  beam centre is the intensity centroid in a rectangular window around the beam,
  sized from the beam itself (`--beam_window` to set it).
- **Beam angle**: the instrument parameter `beam_declination_angle` was
  replaced by `beam_angle` with the **opposite sign** (positive = beam rising
  towards the sample normal; SAGA: `-0.5`), overridable with
  `--instrument_beam_angle`. The old option names stop with an error that gives
  the converted value.
- **Sample orientation** codes are defined by the direction of the surface
  normal looking along the beam (0 = right, 1 = up, 2 = left), consistently for
  simulated hits and measured detector images.
- Each outgoing ray gets the direction at which BornAgain evaluated its
  intensity (the bin centre), jittered by up to half a bin; previously the
  directions were offset from the evaluated ones by up to half a bin.
- `mg_plot` reads the instrument configuration of each file from its metadata;
  command line options take precedence.
- NeXus reading moved from `read_d22.py` to the instrument-independent
  `nexus_reader.py`.
- `--specular specular_simulation`: only the reflected ray is added (the
  transmitted part enters the substrate), only for neutrons hitting the sample,
  and the reflectivity uses the same `--use_avg_materials`, polarisation and
  analyser options as the scattering simulation. It is recommended over
  `include_specular`, which smears the specular reflection over one grid bin and
  replaces that bin's diffuse intensity.
- Paper example `run_d22_sim.sh`: the McStas output of the paper (`d22_1e9`)
  with `--sampling quick` and `specular_simulation` (about a minute); the small
  `d22_1e8` output is kept for the tests only (`tests/data`).
- Internal: type hints and NumPy-style docstrings across the core modules
  (`coordinates.py`, `detector.py`, `instrument.py`, `tof_filtering.py`,
  `preconditioning.py`, `input_output.py`, `masking.py`, `plotting_utils.py`);
  `run.py`, `fit.py` and `plot.py` split into smaller helper functions (e.g.
  `_execute_bornagain_simulation`, `_calculate_nexus_hits`, the TOF and non-TOF
  buffers, the NeXus/simulation loaders of `plot.py`). No change of behaviour;
  verified with the D22 regression tests.

### Removed
- `mg_run`: `--raw_output`, `--quick_plot`, `--bins`, `--x_range`,
  `--y_range`, `--z_range` (the output is a detector image; binning and ranges
  are chosen when plotting).
- `mg_plot`: `--bins` and the other options of the raw Q-event `.npz` workflow.

### Fixed
- A parallel process killed e.g. by the out-of-memory killer made the run wait
  forever; the run now stops with an error.
- Identical random numbers in all worker processes on Linux (the outgoing
  direction jitter and the resolution smearing were correlated between
  processes).
- Particles missing the sample (`--allow_sample_miss`) were moved to the plane
  of the sample surface, far away for a grazing beam, and lost.
- Sample orientations 0 and 2 were mixed up between the simulation and the
  detector image rotation.
- 1D Qz slices summed one bin more than the requested range.
- The polarisation analyser used the pre-22 BornAgain interface, which fails
  with BornAgain 23.
- The built-in model `silica_100nm_air` failed with BornAgain 22 and later
  (`ba.MultiLayer` was replaced by `ba.Sample`; the lattice size must be an
  integer).
