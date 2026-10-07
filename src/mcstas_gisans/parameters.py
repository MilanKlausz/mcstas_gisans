
import numpy as np

from .instrument_defaults import get_instrument_parameters
from .instrument import Instrument
from .sample import Sample


# Noise of the direction sampling per pixel: about SAMPLING_NOISE_COEFFICIENT / sqrt(rays per pixel), with the rays
# counted with the effective number of neutrons (measured for the D22 paper data: 1.5/sqrt(N rho) with N_eff = 0.75 N)
SAMPLING_NOISE_COEFFICIENT = 1.3

# Per-neutron outgoing-angle windows (mg_fit --simulate_mask_angle_range_factor auto): the window of every neutron is
# widened by this many standard deviations of the detector position resolution on every side, for the rays launched
# just outside it and smeared into the region (without smearing the windows lose no ray). Fraction of the rays reaching
# the region that are launched outside the windows (ray tracing with real D22 beams, 4 mm FWHM resolution, three fit
# regions): 0.25-0.9% / 0.05-0.19% / 0.008-0.016% / <0.0015% for 0 / 1 / 2 / 3 sigma.
ANGLE_WINDOW_RESOLUTION_SIGMAS = 2

def outgoing_directions_for_sampling(angle_range, detector, sample_detector_distance, n_effective, rays_per_pixel):
    """
    Number of outgoing directions (horizontal, vertical) with which every detector pixel collects about
    rays_per_pixel rays over the run, in the angle range [horiz_min, horiz_max, vert_min, vert_max] (deg).
    The grid of every neutron is shifted randomly by up to half a bin, so any grid is unbiased; the grid
    only sets the noise of the direction sampling, about SAMPLING_NOISE_COEFFICIENT/sqrt(rays per pixel)
    per pixel. A pixel collects n_effective * rho rays, where n_effective = (sum w)^2 / sum w^2 is the
    effective number of the incident neutrons hitting the sample (their number, for equal weights w) and
    rho the number of directions per pixel of angular area. The directions are split with the same bin
    width (in pixels) along both axes. The pixel sizes are those along the BornAgain horizontal and
    vertical axes, i.e. swapped for vertical samples.
    """
    if n_effective < 1:
        print("WARNING: No incident neutron hits the sample, the outgoing directions are chosen as for a single neutron.")
        n_effective = 1
    pixel_deg_h = np.degrees(detector.pixel_size_y_bornagain / sample_detector_distance)
    pixel_deg_v = np.degrees(detector.pixel_size_z_bornagain / sample_detector_distance)
    pixels_h = (angle_range[1] - angle_range[0]) / pixel_deg_h
    pixels_v = (angle_range[3] - angle_range[2]) / pixel_deg_v
    directions_per_pixel_1d = np.sqrt(rays_per_pixel / n_effective)
    n_h = int(np.ceil(directions_per_pixel_1d * pixels_h - 1e-9))
    n_v = int(np.ceil(directions_per_pixel_1d * pixels_v - 1e-9))
    return max(n_h, 1), max(n_v, 1)


def build_instrument(args, particle_type):
    """The Instrument of the run (instrument parameters incl. CLI overrides, incident angle, wavelength, orientation)."""
    instr_params = get_instrument_parameters(args)
    no_gravity = args.no_gravity if particle_type != 'photon' else True
    return Instrument(instr_params, args.alpha, args.wavelength_selected, args.sample_orientation, args.wfm, no_gravity)


def get_simulated_angle_range(args, instrument):
    """The simulated angle range: --angle_range (or the fit's mask range set into it), else the full detector."""
    return list(args.angle_range) if args.angle_range else list(instrument.get_detector_angle_maximum())


def _arrival_angles(instrument, x, y, v, phi_i, phi_f, alpha_f):
    """
    Angles [deg] (horizontal, vertical) of the detector points hit by rays launched from (x, y) on the sample surface
    with speed v into the outgoing grid directions phi_f (relative to the incident horizontal direction phi_i [rad])
    and alpha_f [deg], seen from the sample centre as get_masked_angle_range sees the pixels (gravity included, no
    resolution smearing).
    """
    L = instrument.sample_detector_distance
    a, p = np.deg2rad(alpha_f), phi_i + np.deg2rad(phi_f)
    _, x_hit, y_hit, z_hit = instrument.detector.detector_plane_intersection(
        x, y, np.zeros_like(x), v * np.cos(a) * np.cos(p), v * np.cos(a) * np.sin(p), v * np.sin(a), L)
    return np.rad2deg(np.arctan2(y_hit, L)), np.rad2deg(np.arctan2(z_hit, x_hit))


def angle_window_margin(instrument):
    """The margin [deg] added on every side of the per-neutron windows: ANGLE_WINDOW_RESOLUTION_SIGMAS sigma of the
    detector position resolution (the larger of its two axes), as seen from the sample."""
    det = instrument.detector
    sigma = max(det.sigma_x_nexus, det.sigma_y_nexus) / instrument.sample_detector_distance
    return float(np.rad2deg(ANGLE_WINDOW_RESOLUTION_SIGMAS * sigma))


def neutron_angle_windows(region, particles, instrument, iterations=3):
    """
    Outgoing-angle window [phi_f_min, phi_f_max, alpha_f_min, alpha_f_max] (deg) of every incident neutron: the
    outgoing grid directions from which its rays can reach the angle range region = [horiz_min, horiz_max, vert_min,
    vert_max] (seen from the sample centre, e.g. the range enclosing the unmasked pixels), widened by
    angle_window_margin on every side for the detector resolution.

    The grid's horizontal angle phi_f is relative to the neutron's incident horizontal direction phi_i, its vertical
    angle alpha_f is absolute, so for a hit point (x, y) on the sample a ray arrives at about phi_f + phi_i + y/L
    horizontally and alpha_f - x tan(alpha_f)/L vertically, plus the gravity drop along the lab vertical (the
    BornAgain z axis for horizontal samples, y for vertical ones). The window is the region shifted by these terms,
    of about the same size for every neutron. Each edge is found exactly with the ray tracing of the simulation:
    Newton iterations from the region edge (the arrival angle changes with the grid angle at a rate of about 1), at
    the value of the other grid angle where the edge is the most extreme (its window edges, or zero in between), so
    that without resolution smearing every ray reaching the region is launched inside the window.

    Parameters
    ----------
    region : list of float
        [horiz_min, horiz_max, vert_min, vert_max] [deg] to be reached.
    particles : np.ndarray
        Preconditioned particles (BornAgain frame, columns p, x, y, z, vx, vy, vz, ...), one row or several; the
        windows of particles missing the sample are meaningless (they are not scattered).
    instrument : Instrument
        The instrument (detector distance and plane, gravity, resolution).
    iterations : int, optional
        Number of Newton iterations (each reduces the error by a factor of about abs(x)/L).

    Returns
    -------
    np.ndarray
        Array of shape (number of particles, 4).
    """
    particles = np.atleast_2d(np.asarray(particles, dtype=float))
    x, y, vx, vy, vz = (particles[:, i] for i in (1, 2, 4, 5, 6))
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    phi_i = np.arctan2(vy, vx)
    region = np.asarray(region, dtype=float)
    window = np.tile(region, (len(particles), 1))
    for _ in range(iterations):
        new = np.empty_like(window)
        # horizontal edges, at the vertical grid angles where they are the most extreme
        alphas = (window[:, 2], window[:, 3], np.clip(0.0, window[:, 2], window[:, 3]))
        for edge, extreme in ((0, np.minimum), (1, np.maximum)):
            new[:, edge] = extreme.reduce([window[:, edge] + region[edge]
                                           - _arrival_angles(instrument, x, y, v, phi_i, window[:, edge], a)[0] for a in alphas])
        # vertical edges, at the horizontal grid angles where they are the most extreme (pointing straight ahead in between)
        phis = (new[:, 0], new[:, 1], np.clip(-np.rad2deg(phi_i), new[:, 0], new[:, 1]))
        for edge, extreme in ((2, np.minimum), (3, np.maximum)):
            new[:, edge] = extreme.reduce([window[:, edge] + region[edge]
                                           - _arrival_angles(instrument, x, y, v, phi_i, p, window[:, edge])[1] for p in phis])
        window = new
    margin = angle_window_margin(instrument)
    return window + np.array([-margin, margin, -margin, margin])


def hits_sample(args, particles):
    """Boolean array: which of the (preconditioned) particles hit the sample, i.e. are scattered."""
    if not len(particles):
        return np.zeros(0, dtype=bool)
    sample = Sample(args.sample_size_y, args.sample_size_x, args.model, None)
    columns = np.asarray(particles).T  # p, x, y, z, vx, vy, vz, ... (BornAgain frame)
    return ~np.asarray(sample.sample_missed(columns[1], columns[2], columns[3], columns[6]), dtype=bool)


def angle_window_extent(args, particles, instrument, region=None):
    """
    The per-neutron outgoing-angle windows (neutron_angle_windows) of region (default: args.angle_window_region) for
    the particles hitting the sample: their union [horiz_min, horiz_max, vert_min, vert_max], i.e. the common angle
    range that would contain every ray reaching the region, and their mean size (horizontal, vertical) [deg].
    """
    region = args.angle_window_region if region is None else region
    hit = hits_sample(args, particles)
    if np.any(hit):
        windows = neutron_angle_windows(region, np.asarray(particles)[hit], instrument)
    else:
        margin = angle_window_margin(instrument)
        windows = np.atleast_2d(np.asarray(region, dtype=float) + np.array([-margin, margin, -margin, margin]))
    union = [float(windows[:, 0].min()), float(windows[:, 1].max()), float(windows[:, 2].min()), float(windows[:, 3].max())]
    size = (float(np.mean(windows[:, 1] - windows[:, 0])), float(np.mean(windows[:, 3] - windows[:, 2])))
    return union, size


def set_outgoing_directions_from_sampling(args, particles, particle_type):
    """
    With a sampling target (--sampling/--rays_per_pixel), choose the outgoing-direction grid from the
    (preconditioned) particles and the final simulated angle range (with args.angle_window_region: the size of
    the per-neutron windows), and store it in args.outgoing_directions_horizontal/_vertical, which
    pack_parameters then uses. Does nothing otherwise.
    """
    rays_per_pixel = getattr(args, 'rays_per_pixel', None)
    if not rays_per_pixel:
        return
    instrument = build_instrument(args, particle_type)
    angle_range = get_simulated_angle_range(args, instrument)

    hit = hits_sample(args, particles)
    weights = (np.asarray(particles).T[0] if len(particles) else np.zeros(0))[hit]
    n_hit = int(np.count_nonzero(hit))
    n_effective = float(weights.sum() ** 2 / (weights ** 2).sum()) if n_hit else 0.0  # effective number for unequal weights

    window_text = ""
    if getattr(args, 'angle_window_region', None) is not None:
        # per-neutron windows (all of about the same size): the grid of every neutron covers its own window
        _, (size_h, size_v) = angle_window_extent(args, particles, instrument)
        angle_range = [0.0, size_h, 0.0, size_v]
        window_text = f" in the outgoing-angle window of each neutron ({size_h:.3f} x {size_v:.3f} deg)"
    n_h, n_v = outgoing_directions_for_sampling(angle_range, instrument.detector, instrument.sample_detector_distance, n_effective, rays_per_pixel)
    args.outgoing_directions = None
    args.outgoing_directions_horizontal, args.outgoing_directions_vertical = n_h, n_v

    # rays per pixel actually reached (above the target, since the direction numbers are rounded up)
    pixel_area_deg2 = np.degrees(instrument.detector.pixel_size_y_bornagain / instrument.sample_detector_distance) \
        * np.degrees(instrument.detector.pixel_size_z_bornagain / instrument.sample_detector_distance)
    angle_area_deg2 = (angle_range[1] - angle_range[0]) * (angle_range[3] - angle_range[2])
    rays = max(n_effective, 1) * n_h * n_v * pixel_area_deg2 / angle_area_deg2
    sampling = getattr(args, 'sampling', None)
    target = f"sampling '{sampling}'" if sampling else f"rays_per_pixel {rays_per_pixel:g}"
    print(f"Outgoing directions: {n_h} x {n_v}{window_text} ({target}: about {rays:.0f} rays per detector pixel (target {rays_per_pixel:g}) "
          f"from {n_hit} neutrons hitting the sample, {n_effective:.0f} effective for their weights). Noise of the direction "
          f"sampling about {100 * SAMPLING_NOISE_COEFFICIENT / np.sqrt(rays):.2g}% per pixel (the statistical noise of the MCPL file adds to it). "
          f"To reproduce: --outgoing_directions_horizontal {n_h} --outgoing_directions_vertical {n_v}")


def pack_parameters(args, particle_type):
    """Pack parameters necessary for processing in a single dictionary"""
    instr_params = get_instrument_parameters(args)
    no_gravity = args.no_gravity if particle_type != 'photon' else True
    instrument = Instrument(instr_params, args.alpha, args.wavelength_selected, args.sample_orientation, args.wfm, no_gravity)

    angle_range = get_simulated_angle_range(args, instrument)
    angle_window_region = getattr(args, 'angle_window_region', None)
    if getattr(args, 'verbose', False):
        if angle_window_region is not None:
            r = angle_window_region
            print(f"Simulated angles [deg]: the window of each neutron reaching horiz=[{r[0]:.4f}, {r[1]:.4f}], vert=[{r[2]:.4f}, {r[3]:.4f}]")
        else:
            print(f"Simulated angle range [deg]: horiz=[{angle_range[0]:.4f}, {angle_range[1]:.4f}], vert=[{angle_range[2]:.4f}, {angle_range[3]:.4f}]")

    sample = Sample(args.sample_size_y, args.sample_size_x, args.model, args.sample_arguments)

    if getattr(args, 'outgoing_directions_horizontal', None) is not None:
        outgoing_directions_horizontal = args.outgoing_directions_horizontal
        outgoing_directions_vertical = args.outgoing_directions_vertical
    elif getattr(args, 'rays_per_pixel', None):
        raise ValueError("The outgoing directions of the sampling target are not set yet, call set_outgoing_directions_from_sampling first.")
    else:
        outgoing_directions = getattr(args, 'outgoing_directions', 20)
        outgoing_directions_horizontal = outgoing_directions
        outgoing_directions_vertical = outgoing_directions

    return {
        'outgoing_directions_horizontal': outgoing_directions_horizontal,
        'outgoing_directions_vertical': outgoing_directions_vertical,
        'angle_range': angle_range,
        'angle_window_region': list(angle_window_region) if angle_window_region is not None else None,
        'sample': sample,
        'instrument': instrument,
        'use_avg_materials': args.use_avg_materials,
        'specular': args.specular,
        'analyzer_direction': args.analyzer_direction if any(args.analyzer_direction) else None,
        'analyzer_efficiency': args.analyzer_efficiency,
        'instrument_name': args.instrument,
        'instrument_params': instr_params,
        'no_gravity': no_gravity,
        'wfm': bool(args.wfm),
        'analyzer_transmission': args.analyzer_transmission,
        'bornagain_number_of_threads': args.bornagain_number_of_threads,
        'random_seed': getattr(args, 'seed', None),
    }