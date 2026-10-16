# TODO

Planned work that is not in the code yet. See `CHANGELOG.md` for what is done.

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
