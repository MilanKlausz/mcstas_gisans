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
