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
- `mg_fit`: an `--output_dir` holds one fit, and an unfinished fit there (stopped by a job time limit,
  `scancel`, a node failure or an error) is **continued automatically** by running the same command again: the
  optimiser is started again, the evaluations it requests in the same order with the same parameters are answered
  from the new `fit_evaluations.jsonl` instead of being simulated, then new ones follow (the result equals that of an
  uninterrupted fit). A finished fit, an unfinished fit with other settings, or a `fit_summary.csv` without the new
  `fit_state.json` stop `mg_fit` with a message instead of being overwritten; a fit that ran out of evaluations goes
  on with a larger `--max_evals`/`--polish_max_evals`. `--overwrite` starts from scratch (the previous behaviour).
  SIGTERM (slurm time limit) records the status `interrupted`. The fitting guide (5.3) shows slurm jobs that continue
  themselves.
- `mg_fit --polish nelder-mead|powell`: refine the best point of a Differential Evolution fit with a local,
  derivative-free optimiser in the same fit (integer parameters fixed; `--polish_max_evals`, `--polish_xatol`,
  `--polish_fatol`); `fit_summary.csv` has a `phase` column.
- Modules `fit_restart` (state of the output directory, evaluation cache), `fit_optimize` (the optimisers and the
  polish) and `fit_continue` (`--de_continue`), split from `fit`.
- `mg_fit --de_continue FIT_SUMMARY`: start a new Differential Evolution fit (in its own `--output_dir`) from
  the population of another one, e.g. a fit of an older `mg_fit` stopped before converging: its evaluations are
  replayed with scipy's update rules (immediate updating) to recover the population where it stopped; the new fit
  starts from it without simulating it again, copies the earlier evaluations into its `fit_summary.csv` and continues
  the evaluation numbers. `--max_evals` counts the new evaluations.
- Instrument `csns_vsans` (CSNS VSANS, BL14, TOF), for the McStas model
  `CSNS_VSANS_GISANS.instr` derived from the CSNS normal-mode model: source to
  sample 22.1077 m, sample to detector 11.5 m; the detector is a placeholder
  flat 2D detector (0.5 x 0.5 m, 2 mm pixels) until the VSANS detector
  geometry is implemented.
- `mg_plot --qy_bins START STOP NBINS` and `--qz_bins START STOP NBINS`: the Q
  bins [1/nm] of TOF event data (simulated or raw SANS2D), as `QxBins`/`QyBins`
  of ESS esssans. Default: as many bins as detector pixels over the Q extent of
  the events (of the wavelength slice).
- `mg_plot`: without `--intensity_min`, the colour scale of maps that are all in
  counts (measured NeXus data alone, e.g. a wavelength slice of raw TOF data,
  or with simulations normalised by `--normalise_to_nxs`) starts at half a
  count instead of 1e-9.
- `mg_fit` for time-of-flight instruments (`--wavelength_slice MIN MAX`,
  repeatable): the measured and the simulated counts of every detector pixel
  and TOF bin are converted to (Qy, Qz) in the same way (the wavelength of a
  TOF bin from L1 + the sample-pixel distance) and compared in the Q grid of
  each wavelength slice; one parameter set for all slices, the loss is the sum
  over the slices (also reported per slice), the flat background is fitted per
  slice. The simulation histograms the detected neutrons per pixel and TOF bin
  (no event files), the MCPL input is filtered for the slices and only the
  neutrons that can reach a slice are simulated. Raw ISIS SANS2D NeXus files
  (rear tube detector) are read (`--nxs_format`, recognised automatically).
- `mg_fit --mask_exit_angle_band LOW HIGH` and `--band_exit_angle LOW HIGH`:
  regions defined by the exit angle relative to the specular direction
  (LOW <= alpha_f - alpha <= HIGH [deg]) instead of fixed Q ranges, for the
  compared pixels and for the band of the 1D Qy profile (instead of
  `--q_min`/`--q_max`). The exit angle of a pixel is that of the Q conversion
  (with the gravity correction) at the wavelength; with `--wavelength_slice`
  at the centre of each slice, so the region follows the specular
  Qz = 2 k sin(alpha) from slice to slice and every TOF bin of a slice uses
  the same pixels (only their counts are compared, for the measurement, the
  simulation and the background). Monochromatic data: a fixed pixel region.
- `mg_fit --intensity_factor_spectrum FILE` (TOF): a wavelength-dependent
  intensity calibration (columns wavelength [AA], factor), e.g. measured /
  simulated direct beam; the simulated count of every (pixel, TOF bin) is
  multiplied by the factor at its wavelength, in addition to
  `--intensity_factor`.
- `mg_fit` with `--wavelength_slice` prints the measured, simulated and
  background counts in the unmasked Q bins of every slice with each evaluation.
- Instrument `sans2d` (ISIS SANS2D, time-of-flight): source-sample 19.2625 m
  (the McStas model `ISIS_SANS2d_12m.instr`), the rear tube detector of the
  2019 measurements (120 tubes x 512 pixels, 1.0414 x 0.972 m, 11.999 m from
  the sample, centre offset [-0.0018, 0.0825] m from the direct beam: the
  direct beam of run 60993 2.6 mm above the centre of its tube), detector
  resolution 8 mm FWHM along the tubes (the SANS2d tube array), the T0
  monitor `Source_TOF_Lambda`; no MCPL TOF-filtering monitor.
- `mg_plot --nxs` reads raw ISIS SANS2D NeXus files as event data (the
  detection events of the rear tube detector, `--nxs_format`, recognised
  automatically) and reduces them like simulated TOF events (Q per event,
  `--wavelength_slice`), e.g. in a split view with a simulation.
- `mg_fit --simulate_mask_angle_range`: the outgoing directions aimed only at masked pixels are no longer
  simulated. Before each neutron's BornAgain simulation its rays are traced to the detector (without the
  resolution smearing, so the random numbers of the simulated rays are unchanged); directions whose hit point,
  widened by 2 sigma of the detector resolution along each detector axis, touches no unmasked pixel are masked in
  the BornAgain detector, which does not compute them. The unmasked pixels get the same counts (up to rays smeared
  by more than 2 sigma, as for the windows). The skipped fraction is printed after the first evaluation; measured:
  a T-shaped fit region of a liquid-surface fit (46 x 36 directions) skips 56% and is 3.5 times faster, the D22
  silica example (band with an excluded specular box, small grid) skips 30% and is 1.2 times faster.
  `--keep_masked_directions` simulates all directions as before. Also for TOF fits (`--wavelength_slice`):
  the region is the pixels with a TOF bin in an unmasked Q bin of any slice.
- `mg_fit --mask_view`: the masked map shows the masked pixels as the measured data on a grey scale, under the
  fitted pixels in colour (`--mask_view_style grey`, default); `--mask_view_style black` blacks them out as
  before.
- `mg_beam_centre_correction --mcpl --figure`: a second figure `<savename>_profiles` with horizontal and
  vertical cuts through the measured and the simulated direct beam (linear and log scale, RMS width and
  FWHM of each cut), to check that the McStas beam (slits, divergence) and the detector resolution reproduce
  the measured beam; `--profile_resolutions FWHM ...` adds simulated beams with other detector resolutions
  along the NeXus y axis (the tubes of D22).
- `mg_plot`: joint figure of two samples (`--nxs2`, `--sample_labels`,
  `--background2`, `--experiment_time2`, two simulation files): with
  `--split_view` one split map per sample in the top row and the Qy profiles of
  the Qz band of all four datasets spanning the bottom row; the same figure
  function as the joint-fit comparison plots of `mg_fit`.
- `mg_fit`: the losses of the 1D Qy profile of the Qz band `--q_min`/`--q_max`
  (the 1D panel of the comparison plots: the unmasked pixels of the band summed
  per Qy bin) are computed next to the 2D losses and written to the log and the
  summaries (`poisson_deviance_1d`, `reduced_chi2_1d`, `log_residual_1d`);
  `--fit_objective {2d,1d}` selects which one is minimised (default `2d`).
- Built-in sample models for several BornAgain versions: one implementation per
  BornAgain API in version folders (`bornagain_samples/ba21`, `ba22` for 22-23,
  `ba24`), each file declaring the major versions it is tested with
  (`BORNAGAIN_VERSIONS`). The implementation for the installed version is used;
  without one the run stops with an error (no silent fallback), unless
  `--allow_untested_bornagain_version` is given. Tests check the selection and
  stored reference results of each implementation.
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
- `mg_plot`, TOF event data: `--y_plot_range`/`--z_plot_range` only set the axis
  limits; they no longer set the range of the Q bins (zoomed plots had bins
  narrower than one pixel at short wavelengths, i.e. combs of empty bins). Pixel
  centres give Q, the bins are a separate choice (`--qy_bins`/`--qz_bins`), as in
  Mantid (Q1D/Qxy) and ESS esssans.
- D22: the default detector position resolution along the tubes is 7 mm FWHM (was 4 mm), from the
  core widths of four measured direct beams (individual best values 6-8.5 mm). Results of D22 simulations
  change slightly near sharp features (the specular, Bragg rods); use
  `--instrument_detector_resolution 0 0.004` to reproduce older results.
- `mg_fit` comparison plots (`fit_eval_*.png`): the simulation curve is labelled
  'Simulation' and the parameters (4 significant digits) are a wrapped title of
  the simulated map, instead of the full parameter string in the legend, which
  squeezed the figure; the file names are unchanged.
- **Average materials by default**: the simulations use BornAgain's average
  materials (`--use_avg_materials`, now the default, as in BornAgain 24: the
  reflection and transmission use the effective refractive index of layers with
  particles); `--no_use_avg_materials` switches it off (the previous default, and
  the default of BornAgain 21-23). All fit scripts used the option already.
- **Dependencies**: BornAgain 21 to 24 (`bornagain>=21.0,<25`; 23.0 is the default in
  `conda.yml`; not every built-in sample model is available with every version: all with
  22 and 23, the `ba21/` models with 21, `silica_100nm_air` and the liquid-surface models
  with 24, with the same results as 23 with average materials; BornAgain 21 needs
  `numpy<2`; PyPI provides Linux wheels of 23 and 24, on macOS only of 21, otherwise a
  locally built wheel is needed, see `INSTALL.md`);
  NumPy 2 is supported. scipp, scippneutron, h5py, matplotlib
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

- Example data: the stored simulation of the paper example
  (`data/paper/bornagain_output`) is a detector image (`.h5`, `--sampling long`)
  instead of a Q histogram (`.npz`), so `examples/paper/run_paper_plot.sh` plots the
  differences to the measurement as well. `*.h5` files are ignored by git, except
  this example.
- Documentation: the D22 measurement of the paper (`data/paper`) is the example
  throughout (quickstart, command reference, workflow, calibration, custom sample
  template, cluster); TOF figures use a SAGA simulation.

### Removed
- `mg_run`: `--raw_output`, `--quick_plot`, `--bins`, `--x_range`,
  `--y_range`, `--z_range` (the output is a detector image; binning and ranges
  are chosen when plotting).
- `mg_plot`: `--bins` and the other options of the raw Q-event `.npz` workflow.
- The built-in models `silica_100nm_air_ba22_ba23` and `silica_100nm_air_ba24`
  (now the implementations of `silica_100nm_air` for these versions) and
  `depthsensitivitysample_v23` (now `depthsensitivitysample`; the earlier
  different `depthsensitivitysample` sample is removed).

### Fixed
- `mg_plot --overlay --plot_differences`: the legend of the difference axis
  no longer covers the legend of the profiles.
- `mg_plot --dual_plot` cut off the axis labels of the tall figure.
- The built-in models `silica_air`, `silica_100nm_D2O`, `lamellas_and_spheres`,
  `hexagonal_spheres` and `depthsensitivitysample` used the BornAgain 21 API
  (`ba.MultiLayer`, `ba.LayerRoughness`) and failed with BornAgain 22 and later;
  they have BornAgain 21 and 22-23 implementations now.
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
