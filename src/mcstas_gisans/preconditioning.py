"""
Preconditioning the particles before the BornAgain simulation
"""

import numpy as np
from pathlib import Path

from .instrument_defaults import instrument_defaults

from .coordinates import CoordinateTransform

# Threshold [deg] above which a mismatch between the beam angle actually used
# for a simulation and the value independently estimated from the MCPL file's
# particle velocities is reported. This is a sanity check only (e.g. to catch
# a forgotten/wrong --instrument_beam_angle, or an MCPL file that does not
# match the intended measurement) -- the estimate itself is never used as a
# fallback value, since the same beam angle must be used consistently with
# mg_beam_centre_correction, which has no MCPL data to estimate it from and
# therefore always requires it to be known and provided explicitly.
BEAM_ANGLE_MISMATCH_WARNING_DEG = 0.05

def calculate_beam_angle(vx_nexus, vy_nexus, vz_nexus, p, sample_orientation):
    """
    Calculate the beam declination angle from the average particle velocities in the NeXus frame.
    NeXus: X=horizontal left, Y=vertical up, Z=longitudinal forward.
    """
    avg_vx = np.average(vx_nexus, weights=p)
    avg_vy = np.average(vy_nexus, weights=p)
    avg_vz = np.average(vz_nexus, weights=p)

    if sample_orientation == 1:
        # Horizontal sample: declination is in the Y-Z plane (vertical plane)
        angle_rad = np.arctan2(avg_vy, avg_vz)
    elif sample_orientation in [0, 2]:
        # Vertical sample: declination is in the X-Z plane (horizontal plane)
        angle_rad = np.arctan2(avg_vx, avg_vz)
    else:
        raise ValueError(f"Unknown sample orientation: {sample_orientation}")

    return float(np.rad2deg(angle_rad))

def transform_to_bornagain_coordinate_system(particles, alpha_inc_deg, sample_orientation, beam_angle, nexus_y_shift=0.0):
    """Apply coordinate transformation to express particle parameters in a
    coordinate system with the sample in the centre and being horizontal.
    Also calculates and returns the actual beam angle used.
    """
    p, x_nexus, y_nexus, z_nexus, vx_nexus, vy_nexus, vz_nexus, w, t, *polarization = particles.T

    if nexus_y_shift != 0.0:
        y_nexus = y_nexus + nexus_y_shift

    calculated_beam_angle = calculate_beam_angle(vx_nexus, vy_nexus, vz_nexus, p, sample_orientation)
    print(f"    Calculated beam angle from MCPL particle velocities (for sanity-checking only): {calculated_beam_angle:.6f} deg")

    actual_beam_angle = beam_angle if beam_angle is not None else 0.0
    print(f"    Actual beam angle used for simulation: {actual_beam_angle:.6f} deg")

    angle_diff = abs(calculated_beam_angle - actual_beam_angle)
    if angle_diff > BEAM_ANGLE_MISMATCH_WARNING_DEG:
        print(
            f"    WARNING: the beam angle used for this simulation ({actual_beam_angle:.6f} deg) differs from the "
            f"value calculated from this MCPL file's particle velocities ({calculated_beam_angle:.6f} deg) by "
            f"{angle_diff:.6f} deg. This does not stop execution, but double-check --instrument_beam_angle (and "
            f"the instrument's configured default) and that this MCPL file matches the intended measurement."
        )

    # In case the beam is not horizontal (beam_angle is not 0), the
    # beam_angle must be taken into account when calculating the
    # rotation angle that needs to be applied to the particle coordinates.
    # We want the incident beam (angle: calculated_beam_angle) to have an angle
    # of -alpha_inc_deg in the BornAgain frame.
    # So: calculated_beam_angle + rotation = -alpha_inc_deg
    # rotation = calculated_beam_angle + alpha_inc_deg
    rotation_angle_deg = actual_beam_angle + alpha_inc_deg
    alpha_inc = float(np.deg2rad(rotation_angle_deg))

    transform = CoordinateTransform(alpha_inc, sample_orientation)

    x_bornagain, y_bornagain, z_bornagain = transform.nexus_to_bornagain(x_nexus, y_nexus, z_nexus)
    vx_bornagain, vy_bornagain, vz_bornagain = transform.nexus_to_bornagain(vx_nexus, vy_nexus, vz_nexus)

    if polarization:
        polx_bornagain, poly_bornagain, polz_bornagain = transform.nexus_to_bornagain(polarization[0], polarization[1], polarization[2])
        return np.vstack([p, x_bornagain, y_bornagain, z_bornagain, vx_bornagain, vy_bornagain, vz_bornagain, w, t, polx_bornagain, poly_bornagain, polz_bornagain]).T, actual_beam_angle
    else:
        return np.vstack([p, x_bornagain, y_bornagain, z_bornagain, vx_bornagain, vy_bornagain, vz_bornagain, w, t]).T, actual_beam_angle

def propagate_to_sample_surface(particles, sample_size_y, sample_size_x, allow_sample_miss):
    """Propagate particles to z=0, the sample surface (in BornAgain coordinates, z is up).
    Discard those which would miss the sample unless allow_sample_miss is True.
    Particles not moving toward the sample surface are not propagated here.
    """
    p, x, y, z, vx, vy, vz, w, t, *polarization = particles.T
    z_original = z.copy()

    # Initialize t_propagate with zeros.
    # This handles cases where vz is zero (particle moves parallel to z=0 or is already on it)
    t_propagate = np.zeros_like(z, dtype=float)

    # Create a mask for particles where vz is not zero to avoid division by zero.
    non_zero_vz_mask = (vz != 0)

    # Calculate t_propagate for particles with non-zero vz.
    # Then, ensure t_propagate is non-negative to avoid back propagation.
    calculated_t_propagate = -z[non_zero_vz_mask] / vz[non_zero_vz_mask]
    t_propagate[non_zero_vz_mask] = np.maximum(0, calculated_t_propagate)

    x += vx * t_propagate
    y += vy * t_propagate
    z += vz * t_propagate
    t += t_propagate

    # Create a boolean mask for the particles to select those which hit the sample
    hit_sample_mask = (
        (abs(y) < sample_size_y * 0.5) &  # Within transverse bounds (left/right, BA Y axis)
        (abs(x) < sample_size_x * 0.5) &  # Within longitudinal bounds (forward/backward, BA X axis)
        (z_original > -1e-12) &           # Not already below the surface
        (vz < 0)                          # Moving toward the surface
    )
    events_on_sample_surface = np.vstack([p, x, y, z, vx, vy, vz, w, t, *polarization]).T if allow_sample_miss else np.vstack([p, x, y, z, vx, vy, vz, w, t, *polarization]).T[hit_sample_mask]

    event_number = len(particles)
    sample_hit_event_number = np.sum(hit_sample_mask)
    if sample_hit_event_number != event_number:
        if np.any(z_original < -1e-12):
            print(f"    WARNING: {np.sum(z_original < 0)} out of {event_number} incident particles are already below the sample surface (z < 0) in the input file.")
        sum_weight_in = sum(p)
        sum_weight_sample_hit = sum(p[hit_sample_mask])
        print(f"    WARNING: {event_number - sample_hit_event_number} out of {event_number} incident particles missed the sample!({sum_weight_in-sum_weight_sample_hit} out of {sum_weight_in} in terms of sum particle weight)")

        if not allow_sample_miss:
            print(f"    WARNING: Incident particles missing the sample are not propagated to the detectors! This can be changed with the --allow_sample_miss option.")
    return events_on_sample_surface

def apply_t0_correction(particles, args):
    """Apply t0 TOF correction for all particles. A fixed t0correction value can be
    given to be subtracted, or a McStas TOFLambda monitor result with a selected
    wavelength is used, in which case t0correction is retrieved as the mean value
    from fitting a Gaussian function to the TOF spectrum of the wavelength bin
    including the selected wavelength. The fitting is done for the full TOF range
    unless the WFM mode is used, in which case it is done within the wavelength
    dependent subpulse TOF limits. Rebinning along the wavelength axis can be
    applied beforehand to improve the reliability of the fitting.
    WARNING: the TOF axis of the monitor is assumed to have microsecond units!
    """
    if args.t0_fixed is not None: #T0 correction with fixed input value
        t0_correction = args.t0_fixed
    else: #T0 correction based on McStas (TOFLambda) monitor
        if not args.wfm:
            tof_limits = [None, None] #Do not restrict the monitor TOF spectrum for T0 correction fitting
            t0_monitor = instrument_defaults[args.instrument]['t0_monitor_name']
        else: # Wavelength Frame Multiplication (WFM)
            from .instrument_defaults import get_saga_subpulse_tof_limits
            tof_limits = get_saga_subpulse_tof_limits(args.wavelength)
            t0_monitor = instrument_defaults[args.instrument]['wfm_t0_monitor_name']
        print(f"Applying T0 correction based on McStas monitor: {t0_monitor}")
        figure_output = f"{args.savename}_t0_correction.{args.t0_correction_figure}" if args.t0_correction_figure in ['png', 'pdf'] else args.t0_correction_figure
        mcstas_directory = Path(args.filename).resolve().parent
        from .fit_monitor import find_mcstas_monitor_tof_centre
        tof_centre = find_mcstas_monitor_tof_centre(mcstas_directory, t0_monitor, args.wavelength, method='com', tof_limits=tof_limits, wavelength_rebin=args.t0_wavelength_rebin, figure_output=figure_output)
        t0_correction = tof_centre * 1e-6
        if args.t0_correction_figure is not None:
            # Terminate the script execution because 'only plotting' has been selected by the user
            import sys
            sys.exit()
    print(f"T0 correction value: {t0_correction} second")

    p, x, y, z, vx, vy, vz, w, t, *polarization = particles.T
    t -= t0_correction
    particles = np.vstack([p, x, y, z, vx, vy, vz, w, t, *polarization]).T
    return particles

def precondition(particles, args):
    """
    Precondition particles (beam) to bridge the gap between Mcstas and BornAgain
    1) Apply coordinate transformation
    2) Propagate particles to the sample surface
    3) Optionally apply T0 (time-of-flight) correction
    """
    instr_params = instrument_defaults.get(args.instrument, {})
    beam_angle = getattr(args, 'instrument_beam_angle', None)
    if beam_angle is None:
        beam_angle = instr_params.get('beam_angle', None)

    particles, actual_beam_angle = transform_to_bornagain_coordinate_system(
        particles, args.alpha, args.sample_orientation, beam_angle, getattr(args, 'nexus_y_shift', 0.0))

    args.instrument_beam_angle = actual_beam_angle
    particles = propagate_to_sample_surface(particles, args.sample_size_y, args.sample_size_x, args.allow_sample_miss)
    if args.no_t0_correction or not instrument_defaults[args.instrument]['tof_instrument']:
        print("No T0 correction is applied.")
    else:
        particles = apply_t0_correction(particles, args)

    return particles
