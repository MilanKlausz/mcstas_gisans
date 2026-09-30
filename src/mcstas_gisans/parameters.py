
import numpy as np

from .instrument_defaults import get_instrument_parameters
from .instrument import Instrument
from .sample import Sample


def outgoing_directions_for_sampling(angle_range, detector, sample_detector_distance, n_hit, rays_per_pixel):
    """
    Number of outgoing directions (horizontal, vertical) with which every detector pixel collects about
    rays_per_pixel rays over the run, in the angle range [horiz_min, horiz_max, vert_min, vert_max] (deg).
    The grid of every neutron is shifted randomly by up to half a bin, so any grid is unbiased; the grid
    only sets the statistical noise, about 1.5/sqrt(rays per pixel) per pixel. A pixel collects
    n_hit * rho rays, where n_hit is the number of incident neutrons hitting the sample and rho the
    number of directions per pixel of angular area. The directions are split with the same bin width
    (in pixels) along both axes. The pixel sizes are those along the BornAgain horizontal and vertical
    axes, i.e. swapped for vertical samples.
    """
    if n_hit < 1:
        print("WARNING: No incident neutron hits the sample, the outgoing directions are chosen as for a single neutron.")
        n_hit = 1
    pixel_deg_h = np.degrees(detector.pixel_size_y_bornagain / sample_detector_distance)
    pixel_deg_v = np.degrees(detector.pixel_size_z_bornagain / sample_detector_distance)
    pixels_h = (angle_range[1] - angle_range[0]) / pixel_deg_h
    pixels_v = (angle_range[3] - angle_range[2]) / pixel_deg_v
    directions_per_pixel_1d = np.sqrt(rays_per_pixel / n_hit)
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


def set_outgoing_directions_from_sampling(args, particles, particle_type):
    """
    With a sampling target (--sampling/--rays_per_pixel), choose the outgoing-direction grid from the
    (preconditioned) particles and the final simulated angle range, and store it in
    args.outgoing_directions_horizontal/_vertical, which pack_parameters then uses. Does nothing otherwise.
    """
    rays_per_pixel = getattr(args, 'rays_per_pixel', None)
    if not rays_per_pixel:
        return
    instrument = build_instrument(args, particle_type)
    angle_range = get_simulated_angle_range(args, instrument)

    sample = Sample(args.sample_size_y, args.sample_size_x, args.model, None)
    columns = np.asarray(particles).T if len(particles) else np.zeros((7, 0))
    x, y, z, vz = columns[1], columns[2], columns[3], columns[6]  # p, x, y, z, vx, vy, vz, ... (BornAgain frame)
    n_hit = int(np.count_nonzero(~sample.sample_missed(x, y, z, vz)))

    n_h, n_v = outgoing_directions_for_sampling(angle_range, instrument.detector, instrument.sample_detector_distance, n_hit, rays_per_pixel)
    args.outgoing_directions = None
    args.outgoing_directions_horizontal, args.outgoing_directions_vertical = n_h, n_v

    # rays per pixel actually reached (above the target, since the direction numbers are rounded up)
    pixel_area_deg2 = np.degrees(instrument.detector.pixel_size_y_bornagain / instrument.sample_detector_distance) \
        * np.degrees(instrument.detector.pixel_size_z_bornagain / instrument.sample_detector_distance)
    angle_area_deg2 = (angle_range[1] - angle_range[0]) * (angle_range[3] - angle_range[2])
    rays = max(n_hit, 1) * n_h * n_v * pixel_area_deg2 / angle_area_deg2
    sampling = getattr(args, 'sampling', None)
    target = f"sampling '{sampling}'" if sampling else f"rays_per_pixel {rays_per_pixel:g}"
    print(f"Outgoing directions: {n_h} x {n_v} ({target}: about {rays:.0f} rays per detector pixel (target {rays_per_pixel:g}) "
          f"from {n_hit} neutrons hitting the sample, expected noise about {100 * 1.5 / np.sqrt(rays):.2g}% per pixel). "
          f"To reproduce: --outgoing_directions_horizontal {n_h} --outgoing_directions_vertical {n_v}")


def pack_parameters(args, particle_type):
    """Pack parameters necessary for processing in a single dictionary"""
    instr_params = get_instrument_parameters(args)
    no_gravity = args.no_gravity if particle_type != 'photon' else True
    instrument = Instrument(instr_params, args.alpha, args.wavelength_selected, args.sample_orientation, args.wfm, no_gravity)

    angle_range = get_simulated_angle_range(args, instrument)
    if getattr(args, 'verbose', False):
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