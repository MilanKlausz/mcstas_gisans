# TODO

Possible future improvements and planned work, not yet implemented (see `CHANGELOG.md` for what is done).

## General

Possible future improvements, not yet implemented. Most were deliberately deferred during the dev-scipp review
(Sept-Oct 2026); see `review_dev_scipp_findings_local.md` for context.

* **Direct-beam attenuation / dead time in the intensity factor.** Direct-beam measurements are often taken with an attenuator (and at high count rates detector dead time matters). The intensity-factor formula `NXS sum / NXS time / MCPL sum(weights)` ignores both. A user-supplied attenuation factor (and optionally a dead-time correction) could be added once the necessary information is available for each instrument. Not read from NeXus files, since their layout differs between facilities.
* **Parallel performance tuning.** Processes × BornAgain threads should not exceed the allocated cores (BornAgain defaults to all node cores). Needs a benchmark grid on the cluster before changing defaults.
* **Instrument definitions, stage B.** Built-in instrument definitions as data files; user-supplied instrument files (`extends: d22` + overrides) so users need not edit version-controlled code.
* **Instrument definitions, stage C.** Move to scipp/NeXus JSON-template instrument definitions (separate project).
* **TOF incident angle vs wavelength.** For horizontal samples in TOF mode the real incident angle depends on wavelength (gravity along the collimation, e.g. REFSANS: 0.62°→0.69°). Support both horizontal and vertical samples in TOF mode; the direct-beam centroid vs λ² can calibrate it.
* **Optional NeXus sanity checks.** Where a facility's NeXus layout is known (e.g. D22 `entry0/D22/selector/wavelength`), compare with user input and warn (never used as the value itself).
* **Automatic background from the measurement (`--background auto` / `--background_region`).** Estimate the flat background once, before a fit, from measured bins where the sample cannot scatter, and use it as a fixed value (model-independent, unlike `--fit_background`, whose value is conditional on the simulated pattern and absorbs model errors far from the optimum). Example (D22 high-res microgel, 8 files, 14064 s): 6.1 counts/pixel below the sample horizon (Qz < 0), the same at |Qy| > 0.25 and at high Qz, with a Poisson-like spread; the hand-chosen `--background 1.0` was far too low, and the fit compensated with a too intense pattern. Needs:
  - a CLI to determine and inspect it before fitting (e.g. `mg_background`): the user selects one or more Q-boxes (`--background_region qy_min qy_max qz_min qz_max`, repeatable), with a dry-run plot of the measured map showing the regions, the mean counts per bin in each region, their spread vs. Poisson, and the resulting value;
  - a default region for `auto` (e.g. below the sample horizon, excluding a box around the direct/transmitted beam) — not general: it depends on the geometry (horizontal/vertical sample, beam declination) and on the instrument, and may not exist for every measurement;
  - a robust estimator (mean after outlier rejection; the median is biased for low integer counts);
  - a check of flatness (compare regions; e.g. incoherent scattering of the subphase is not necessarily flat);
  - printing the estimate next to the fitted value in `--fit_background` runs as a sanity check;
  - mutually exclusive with `--fit_background` and a numeric `--background`; per sample in joint fits (`--background2`).
* **Adaptive outgoing-direction sampling.** The `--sampling` presets choose the outgoing-direction grid from the number of rays per detector pixel (R = N_eff * rho, direction-sampling noise ~1.3/sqrt(R)), before anything is known about the pattern. What matters for a fit is the Monte Carlo variance of the simulated counts relative to the counting variance of the measurement (`mc_to_poisson_variance`, which `mg_fit` already reports and warns about above 1). Option: after the first evaluation, scale the grid (both directions, keeping the bin width balanced in pixels) so that the MC variance stays below the counting variance in most unmasked pixels, and keep that grid for the rest of the fit (a changing grid between evaluations would add noise to the loss). Sharp features (BornAgain evaluates each virtual pixel at its centre) may need a finer grid than the noise alone suggests.

## From the first-time-user documentation reviews (2026-10-09)

Details, file:line references and more minor points in `DOCS_REVIEW_A_D22_newcomer_local.md` (a D22 user on macOS with a
liquid-surface sample) and `DOCS_REVIEW_B_TOF_instrument_scientist_local.md` (a TOF-SANS instrument scientist adding an
instrument). Quick fixes (dead McStas links, `--max_evals` in the fit examples, section numbering, thread advice,
quickstart working directory) are done.

* **Liquid-surface example.** No page covers a horizontal sample with a downward-inclined beam (sign of
  `--instrument_beam_angle`, `--sample_orientation 1`, particles already below the surface in the MCPL file); the
  liquid-surface McStas models and MCPL outputs are only in `*_local` files. Add a worked example (with a small MCPL
  file and a measurement), and a catalogue of the built-in sample models with their parameters.
* **Adding an instrument.** Only possible by editing `instrument_defaults.py` (one sentence in technical_details);
  `--instrument` accepts only built-in names; `mcpl_monitor_name` has no override; `mg_beam_centre_correction` has no
  `--instrument_*` overrides (e.g. the detector distance). Needs a how-to page (and see "Instrument definitions, stage B").
* **TOF workflow and example.** Explain the roles of `--wavelength` (TOF filter centre, T0 wavelength, WFM sub-pulse),
  that by default only one TOF-FWHM window is kept (`--no_mcpl_filtering` for the full band), and the TOF units (ms,
  s, µs); store the T0 and the TOF limits in the `.h5`; add a small TOF example (SAGA/LoKI model with MCPL file and
  `mccode.sim`); show `mg_fit_monitor` in use.
* **Measured TOF NeXus data.** 3D (TOF) detector data are summed over time and converted to Q at a single wavelength
  with only a warning: physically wrong. Error out until TOF NeXus data are supported (event mode, ISIS spectra).
* **`--wfm` sub-pulse limits** are hard-coded for SAGA and used for any instrument with the WFM keys, without a
  warning: make them instrument keys or refuse other instruments.
* **macOS installation.** A tested build script or a prebuilt arm64 wheel of BornAgain 23, or a Docker/apptainer
  fallback; INSTALL.md lacks repository URLs and the dependency/CMake commands.
* **McStas model for one's own measurement.** `ILL_D22.instr` hard-codes the pinhole and the sample aperture
  (`D22_collimation` other than 17.6 m puts the pinhole inside the guide without warning); document which NeXus
  fields hold the detector distance, wavelength, detector translation, slits and attenuator.
* **Consistent CLI.** `mg_run`/`mg_fit` require `--wavelength_selected` for monochromatic instruments, `mg_plot` and
  `mg_beam_centre_correction` only accept `--wavelength`. Defaults are the paper values (`--alpha 0.24`, 6 x 8 cm
  sample) or not the recommended ones (`--specular none`, while `specular_simulation` is recommended everywhere);
  `mg_fit --output_dir` defaults to `scan_results` also for fits.
* **Small code issues.** An empty `--wavelength_slice` range crashes with a numpy `ValueError`; a misspelled `--fit`
  parameter only warns (costly on a cluster: consider an error); the detector image orientation (x to the left looking
  along the beam) cannot be flipped/transposed, a mirrored detector gives a mirrored Qy silently.
* **Documentation structure.** The loss functions are explained three times and the BornAgain version list four times;
  troubleshooting/known issues sit under the internals; the calibration guide ("Replicating a Measurement") should come
  before the long workflow page; worked examples need expected results and run times; the output-format page needs a
  loading snippet and TOF file-size guidance; the intensity-factor procedure is described three times.
* **mg_plot legend overlap.** mg_plot overlay/dual plot: the legend of the "Relative absolute difference" axis covers the "measured" entry of the profile legend (seen in the NP joint figures, 2026-10-10).

## Time of flight

### T0 correction for several wavelength slices
`mg_run` subtracts one T0 for the whole run: the mean emission time (centre of mass of the TOF spectrum of the source
`TOFLambda_monitor`) in the wavelength bin of `--wavelength`. This is correct for a simulation of one narrow
wavelength band. For a whole band (`--wavelength_slice`, several slices) the T0 depends on the wavelength (e.g.
SANS2D: 305, 361, 398 us at 4, 7, 10 AA), so one value is wrong for all but one slice; `mg_run` therefore refuses the
monitor T0 with `--wavelength_slice` (use `--no_t0_correction` or `--t0_fixed`).
- Option: a T0 per wavelength, i.e. a lookup table T0(lambda) from the same monitor, applied per particle.
- In measured data the wavelength is only known after the correction: index the table by the measured arrival time
  (see the ESS lookup table below) or iterate once (lambda from the uncorrected TOF -> T0(lambda)), which is enough for
  short-pulse sources where T0 changes slowly with the wavelength.
- Whatever the simulation does must match the reduction of the measured data (`--no_t0_correction` when the measured
  TOF starts at the source pulse and is not corrected, e.g. ISIS SANS2D, CSNS).

### ESS instruments (SKADI): time of flight from the arrival time
The ESS pulse is long (2.86 ms), so the time of flight cannot be the arrival time minus one T0 per wavelength. The
ESS reduction (`essreduce`, `ess.reduce.unwrap` / `ess.reduce.time_of_flight`) converts `event_time_offset` to a time
of flight with a lookup table: for every total flight distance (e.g. 9-35 m in 0.1 m steps) and arrival-time bin (e.g.
250 us), the weighted mean time of flight of simulated neutrons (with its spread; bins with a large relative spread
are masked). The table is computed per instrument and chopper setting (`unwrap.LookupTableWorkflow`: disk choppers,
source position, distance range, resolutions; by default a generic ESS source pulse), and it also handles frame
unwrapping / pulse skipping.
- Goal: simulate SKADI with the same time-of-flight treatment as its future data reduction.
- esssans has no SKADI workflow yet (only LoKI and ISIS instruments), so no ready table.
- We can make the table ourselves: McStas SKADI model with TOF-wavelength monitors / MCPL output at the relevant
  distances (instrument-specific source/moderator view and choppers), then either essreduce's table code (it accepts
  McStas readings: `ess.reduce.time_of_flight.lut`) or the mean TOF per (distance, arrival time) bin. Prefer the
  essreduce code to a re-implementation.
- Then: apply the table to the simulated events (`mg_run` output) instead of the T0 correction.

### Wavelength frame multiplication (WFM): general sub-pulse assignment
`--wfm` uses the virtual source at the WFM chopper system: T0 = centre of mass of the TOF distribution of the
sub-pulse at the WFM choppers (`wfm_t0_monitor_name`), path shortened by `wfm_virtual_source_distance`. The sub-pulse
TOF limits are hard-coded for SAGA (`saga_subpulse_tof_limits` in `instrument_defaults.py`), because the sub-pulses
overlap in wavelength: the TOF spectrum of one wavelength bin can contain two sub-pulses, so the centre of mass of the
whole bin would be wrong.
- Do it correctly for any WFM instrument: assign every neutron (or wavelength/TOF region) to its sub-pulse from the
  chopper settings (opening times and positions of the WFM choppers), instead of hard-coded limits.
- Not needed at the moment (no WFM instrument in use).

## Data reduction on ESS/scipp code (review 2026-10-15)
Rule: no own data-reduction code where scippneutron / scippnexus / essreduce / esssans provide it (dependency +
tests); simulated and measured data in the same format, converters per facility. Today scippneutron is used once
(tof -> wavelength in mg_plot); readers, geometry, mg_fit tof -> wavelength, Q with gravity, binning, masks and
beam-centre finding are own code. Simulation-only parts stay own (MCPL handling, BornAgain, ray tracing, detector
smearing, virtual-experiment Poisson noise, fit losses).

Open decisions:
- Python 3.12 environments (the current essreduce / essreflectometry / scippneutron releases need >= 3.12; ours are
  3.11, where essreduce <= 26.8, esssans 26.6, essreflectometry 26.6 without the Freia sample-normal angle code):
  check BornAgain for 3.12 and the cluster container, or stay on 3.11 and copy the few Freia functions (BSD-3).
- Weighted events: NeXus has no weight field (scippnexus sets weights to 1, essreduce sets variance = value).
  Proposal: keep weights as a documented `event_weight` field + our own loader step (variance w^2, as ESS's
  BIFROST/Freia McStas loaders), plus an optional unit-weight "virtual measurement" export (Poisson(weight x time),
  folded into frames) that any standard tool reads unchanged.

Plan:
1. No new dependencies: scippneutron for the mg_fit tof -> wavelength (tof_slices.py), scippnexus for the SANS2D
   events (event_time_zero kept), regression tests against the current results.
2. NeXus writer for simulated events: NXevent_data (one pulse costs 16 bytes; event_time_offset uint32 ns,
   event_id, event_weight, gzip/lzf + shuffle, sorted by pixel: ~10 bytes/event vs 28 now), NXdetector geometry,
   NXsource (L1), NXsample (alpha), provenance. Converters: raw SANS2D (copy events + our geometry + monitors),
   D22 images (histogram + wavelength; esssans has no monochromatic path), CSNS placeholder (40 ms frames).
3. Loading with essreduce GenericNeXusWorkflow + weight provider; tof from event_time_offset (ISIS) or essreduce
   lookup tables (ESS, CSNS frames); wavelength with scippneutron; scipp masks.
4. GISANS Q (alpha_i, alpha_f, in-plane angle, gravity) as a scippneutron conversion-graph step on the Freia /
   scippneutron building blocks (no GISANS conversion exists in ESS code), regression-tested against
   Instrument.calculate_q_from_nexus_positions.
5. mg_fit histogram path (per pixel and TOF bin) on the same graph; esssans binning / normalisation where it fits.
Details: ESS_REUSE_REVIEW_local.md (owner's notes, not in the repository).

## Feature branch (planned): event data compatible with ESS tools
Separate branch from dev (after feature/tof-fit is merged), e.g. `feature/nexus-events`. Covers steps 1-3 of the plan
above; nothing of it is started.

0. First, decide whether it is worth it (gain vs cost), before writing code:
   - Possible gains: the ESS functions for frame unwrapping / time-of-flight lookup tables (needed for ESS (SKADI)
     and for CSNS frames at 25 Hz), detector geometry from NeXus files instead of our presets, the same reduction
     for simulated and measured ESS data, standard tools (Mantid, plain scippnexus) able to open simulated files.
   - Costs: Python >= 3.12 environments for the current releases (BornAgain + cluster container), dependencies
     (sciline, essreduce, ...), weights that the standard loaders drop (own loader step needed anyway), keeping up
     with ESS API changes (the packages moved to the scipp/ess monorepo in 2026).
   - The gain comes mainly from the in-memory scipp objects the ESS functions work on (event DataArray with
     event_time_offset / event_time_zero, pixel positions, source and sample positions), not from the file layout:
     the on-disk format only matters for our own readers. Option A: write NeXus files (as planned above).
     Option B: keep our own file format, and make every reader (simulation, raw SANS2D, D22, CSNS, ESS NeXus)
     return the same in-memory structure, on which the scippneutron / essreduce functions are called directly.
     Compare the two before choosing.
1. Must stay general for TOF instruments outside ESS: ISIS (SANS2D: 10 Hz, no frame wrapping, tof = offset),
   CSNS (25 Hz: frames to unwrap; time-zero convention to confirm), ESS (14 Hz, long pulse: lookup table).
   Facility-specific readers / converters bring each format to the common structure; the reduction after that is
   facility-independent (pulse period, source position and time-zero offset as parameters).
2. Event times: event_time_zero (per pulse) + event_index (per pulse) + event_time_offset (per event, from the
   pulse during which the neutron is detected), as in ESS files (ev44 stream: reference_time, reference_time_index,
   time_of_flight = ns since reference_time). One simulated pulse: event_time_zero = [0], event_index = [0], offset
   = McStas time. Folding into frames (optional, to look like measured data): frame k = floor(t / T),
   offset = t - k T, events grouped by frame, event_time_zero = [0, T, 2T, ...], event_index from the counts per
   frame. The same code supports McStas runs with several pulses (if the McStas time includes each pulse start
   k T; check per source component, e.g. the ESS source with n_pulses) -- an option only, not for normal studies:
   one folded pulse equals a steady multi-pulse measurement. Test: two identical pulses folded = 2 x one pulse.
   Today multi-pulse McStas output is NOT supported (the McStas time is taken as the TOF since the pulse).
3. Weights: documented extra field (e.g. event_weight) + our loader step setting variance = w^2; never let a
   standard loader turn them into unit weights silently (test: sum of weights after loading).
4. Time zero: record what t = 0 of the McStas source means (proton pulse at ISIS/CSNS sources?; ESS pulse start)
   and the offset to the facility's T0, so the unwrapping of simulated and measured data uses the same convention.
