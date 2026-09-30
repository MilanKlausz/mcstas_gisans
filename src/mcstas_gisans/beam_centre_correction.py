"""
Find the detector centre offset from a measured direct-beam NeXus file.

The offset is the position of the detector centre relative to the undeflected nominal
beam axis through the sample (NeXus frame, metres). It is a property of the detector
position only, so it does not depend on the sample orientation. It is computed directly
in real space: the unscattered beam lands at the point where the undeflected beam axis
(tilted by the beam angle) hits the detector plane, lowered by the gravity drop at the
given wavelength; the offset places the measured intensity centroid on that point.
Together with the Q convention of Instrument.calculate_q_from_nexus_positions this puts
the measured direct beam exactly at Q = 0.
"""

import sys
import copy
import numpy as np

from .nexus_reader import read_nexus_raw
from .instrument import Instrument
from .instrument_defaults import instrument_defaults, default_detector

DEFAULT_BEAM_RADIUS = 0.1  # [m] radius around the beam used for the centroid (holds the whole D22 big beam)
SPECULAR_HALF_ALONG = 0.02  # [m] specular window half-width along the sample normal
SPECULAR_HALF_ACROSS = 0.08  # [m] specular window half-width across the sample normal: the whole spot


def find_required_centre_offset(filepath, beam_angle=None, wavelength=6.0, sample_orientation=1, instrument_name='d22', verbose=False, nxs_data_path=None, beam_radius=DEFAULT_BEAM_RADIUS):
    """
    Detector centre offset [x, y] (m, NeXus frame) that reproduces a measured direct beam.

    Parameters
    ----------
    filepath : str
        Path to the NeXus file of the direct-beam measurement.
    beam_angle : float, optional
        Beam angle in degrees (angle of the incident beam above the nominal axis in the
        plane of incidence, towards the sample surface normal). Default: the instrument's
        configured 'beam_angle', or 0.0.
    wavelength : float, optional
        Wavelength of the direct-beam measurement in Angstrom (default: 6.0). Only enters
        through the gravity drop.
    sample_orientation : int, optional
        Sample orientation (0, 1, 2). Only defines the plane in which the beam angle acts.
    instrument_name : str, optional
        The name of the instrument key in instrument_defaults (default: 'd22').
    verbose : bool, optional
        Print the intermediate quantities.
    nxs_data_path : str, optional
        Explicit HDF5 path to the detector data inside the NeXus file.
    beam_radius : float, optional
        Radius [m] around the beam used for the centroid (default 0.05; <= 0: whole detector).

    Returns
    -------
    centre_offset : np.ndarray
        The detector centre offset [x, y] in metres.
    """
    if instrument_name not in instrument_defaults:
        raise ValueError(f"Unknown instrument '{instrument_name}'. Available: {list(instrument_defaults)}")
    params = copy.deepcopy(instrument_defaults[instrument_name])
    params.setdefault('detector', copy.deepcopy(default_detector))
    params['detector']['direct_beam_centre_offset'] = [0.0, 0.0]
    if beam_angle is not None:
        params['beam_angle'] = beam_angle
    instrument = Instrument(params, 0.0, wavelength, sample_orientation)
    det = instrument.detector

    raw = read_nexus_raw(filepath, nxs_data_path)
    if raw.shape != (det.pixels_x_nexus, det.pixels_y_nexus):
        raise ValueError(f"Detector image shape {raw.shape} does not match the '{instrument_name}' detector "
                         f"({det.pixels_x_nexus}, {det.pixels_y_nexus}).")
    total = raw.sum()
    if total <= 0:
        raise ValueError(f"No counts in the detector image of {filepath}.")

    centroid, _, _, _ = image_statistics(raw, det, beam_radius)

    landing = instrument.direct_beam_landing_point_nexus(wavelength)
    offset = landing[:2] - centroid

    if verbose:
        print(f"Direct-beam centroid relative to the detector centre [m]: [{centroid[0]:.6f}, {centroid[1]:.6f}]")
        print(f"Predicted landing point of the unscattered beam [m]:      [{landing[0]:.6f}, {landing[1]:.6f}]"
              f" (beam angle {instrument.beam_angle} deg, wavelength {wavelength} Å)")
    return offset


def _instrument(instrument_name, offset, beam_angle, wavelength, sample_orientation, alpha=0.0):
    params = copy.deepcopy(instrument_defaults[instrument_name])
    params.setdefault('detector', copy.deepcopy(default_detector))
    params['detector']['direct_beam_centre_offset'] = list(offset)
    if beam_angle is not None:
        params['beam_angle'] = beam_angle
    return Instrument(params, alpha, wavelength, sample_orientation)


def image_statistics(image, detector, beam_radius=DEFAULT_BEAM_RADIUS):
    """
    Intensity centroid and RMS width [m] of the beam spot in a raw detector image, relative to
    the detector centre (NeXus frame). Only pixels within beam_radius of the centroid are used
    (iterated from the brightest pixel, as in Mantid FindCenterOfMassPosition), so scattered
    background elsewhere on the detector does not bias the result; beam_radius <= 0 uses the
    whole detector.
    """
    x_rel = (np.arange(detector.pixels_x_nexus) + 0.5) * detector.pixel_size_x_nexus - 0.5 * detector.size_x_nexus
    y_rel = (np.arange(detector.pixels_y_nexus) + 0.5) * detector.pixel_size_y_nexus - 0.5 * detector.size_y_nexus
    X, Y = np.meshgrid(x_rel, y_rel, indexing='ij')

    def moments(weights):
        total = weights.sum()
        c = np.array([(weights * X).sum(), (weights * Y).sum()]) / total
        w = np.sqrt(np.array([(weights * (X - c[0]) ** 2).sum(), (weights * (Y - c[1]) ** 2).sum()]) / total)
        return c, w

    if beam_radius is None or beam_radius <= 0:
        centre, width = moments(image)
    else:
        i, j = np.unravel_index(np.argmax(image), image.shape)
        centre = np.array([x_rel[i], y_rel[j]])
        for _ in range(50):
            window = np.where((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 <= beam_radius ** 2, image, 0.0)
            new_centre, width = moments(window)
            converged = np.allclose(new_centre, centre, rtol=0, atol=1e-9)
            centre = new_centre
            if converged:
                break
    return centre, width, x_rel, y_rel


def simulate_direct_beam(mcpl_path, offset, beam_angle=None, wavelength=6.0, sample_orientation=1, instrument_name='d22', seed=0):
    """
    Ray-trace the neutrons of a McStas MCPL file (at the sample position) straight to the detector,
    exactly as mg_run does for a direct-beam simulation (no sample, gravity included), and return
    the raw detector image (sum of weights, i.e. rate per second) and some beam properties.
    """
    from .input_output import get_particles
    from .preconditioning import transform_to_bornagain_coordinate_system, calculate_beam_angle
    instrument = _instrument(instrument_name, offset, beam_angle, wavelength, sample_orientation)
    particles, _, _ = get_particles(mcpl_path, 1.0, [float('-inf'), float('inf')], 0.0, use_polarization=False)
    p, _, _, _, vx, vy, vz, lam = particles.T[:8]
    info = {
        'mcpl_beam_angle': calculate_beam_angle(vx, vy, vz, p, sample_orientation),
        'mcpl_mean_wavelength': float(np.average(lam, weights=p)),
        'mcpl_total_rate': float(p.sum()),
    }
    particles_ba, _ = transform_to_bornagain_coordinate_system(particles, 0.0, sample_orientation, instrument.beam_angle)
    weight, x, y, z, vx, vy, vz, _, t = particles_ba.T[:9]
    np.random.seed(seed)  # detector resolution smearing
    ix, iy, valid, _ = instrument.calculate_pixel_hit(x, y, z, t, vx, vy, vz)
    image = np.zeros((instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus))
    np.add.at(image, (ix[valid], iy[valid]), weight[valid])
    return image, info, instrument


def compare_with_simulated_direct_beam(filepath, mcpl_path, offset, beam_angle=None, wavelength=6.0, sample_orientation=1,
                                       instrument_name='d22', nxs_data_path=None, experiment_time=None, figure=None,
                                       savename='beam_centre_check', residual_warning_pixels=0.5, beam_radius=DEFAULT_BEAM_RADIUS):
    """
    Cross-check the detector offset with a direct-beam simulation of the McStas beam: report the
    centroid residual (simulated - measured), the spot widths, the MCPL beam angle and wavelength,
    and the intensity factor (if experiment_time is given). The offset itself always comes from
    the measurement; a residual points at the McStas model (beam direction/position, slits),
    the beam angle, or the wavelength. Returns a dict of the results.
    """
    measured = read_nexus_raw(filepath, nxs_data_path)
    simulated, info, instrument = simulate_direct_beam(mcpl_path, offset, beam_angle, wavelength, sample_orientation, instrument_name)
    det = instrument.detector
    c_meas, w_meas, x_rel, y_rel = image_statistics(measured, det, beam_radius)
    c_sim, w_sim, _, _ = image_statistics(simulated, det, beam_radius)
    residual = c_sim - c_meas
    pixel = np.array([det.pixel_size_x_nexus, det.pixel_size_y_nexus])

    results = dict(info, residual=residual, residual_pixels=residual / pixel, width_measured=w_meas, width_simulated=w_sim,
                   simulation_matched_offset=np.asarray(offset) + residual)
    print("\n--- Direct-beam simulation cross-check ---")
    print(f"Centroid residual simulated - measured [mm]: x={residual[0]*1e3:+.3f}, y={residual[1]*1e3:+.3f} "
          f"({residual[0]/pixel[0]:+.2f} px, {residual[1]/pixel[1]:+.2f} px)")
    print(f"RMS spot width measured / simulated [mm]: x={w_meas[0]*1e3:.2f} / {w_sim[0]*1e3:.2f}, y={w_meas[1]*1e3:.2f} / {w_sim[1]*1e3:.2f}")
    print(f"Beam angle: used {instrument.beam_angle:.4f} deg, MCPL estimate {info['mcpl_beam_angle']:.4f} deg")
    print(f"Wavelength: given {wavelength:.4f} Å, MCPL mean {info['mcpl_mean_wavelength']:.4f} Å")
    if np.any(np.abs(residual / pixel) > residual_warning_pixels):
        print(f"WARNING: the simulated direct beam is more than {residual_warning_pixels} pixel off the measured one. The offset "
              f"above describes the measurement; the difference points at the McStas beam (direction, position, slits), "
              f"the beam angle or the wavelength. If the McStas beam is known to be right, the offset that makes the "
              f"simulation match is [{results['simulation_matched_offset'][0]:.6f}, {results['simulation_matched_offset'][1]:.6f}].")
    if abs(info['mcpl_mean_wavelength'] - wavelength) > 0.02 * wavelength:
        print("WARNING: the mean wavelength of the MCPL file differs from --wavelength by more than 2%: wrong MCPL file or wavelength?")
    if experiment_time:
        measured_rate = measured.sum() / experiment_time
        results['intensity_factor'] = measured_rate / simulated.sum()
        print(f"Intensity factor = measured counts / experiment_time / simulated rate on the detector = "
              f"{measured.sum():.0f} / {experiment_time} / {simulated.sum():.2f} = {results['intensity_factor']:.4f}")
        from .nexus_reader import warn_if_duration_mismatch
        warn_if_duration_mismatch([filepath], experiment_time, label=filepath)
    if figure:
        _direct_beam_figure(measured, simulated, x_rel + offset[0], y_rel + offset[1], c_meas + offset, c_sim + offset, figure, savename,
                            beam_radius)
    return results


def _direct_beam_figure(measured, simulated, x_lab, y_lab, c_meas, c_sim, figure, savename, beam_radius=DEFAULT_BEAM_RADIUS):
    import matplotlib
    if figure != 'show':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    scaled = simulated * measured.sum() / simulated.sum()
    from matplotlib.patches import Circle
    ix, iy = np.unravel_index(np.argmax(measured), measured.shape)
    pitch_x, pitch_y = abs(x_lab[1] - x_lab[0]), abs(y_lab[1] - y_lab[0])
    reach = max(beam_radius or 0.0, 0.06) * 1.15  # show the whole centroid window
    hx, hy = int(np.ceil(reach / pitch_x)), int(np.ceil(reach / pitch_y))
    xs, ys = slice(max(ix - hx, 0), ix + hx + 1), slice(max(iy - hy, 0), iy + hy + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    extent = [x_lab[xs][0] * 1e3, x_lab[xs][-1] * 1e3, y_lab[ys][0] * 1e3, y_lab[ys][-1] * 1e3]
    vmax = max(measured[xs, ys].max(), scaled[xs, ys].max())
    for ax, image, title, centre in ((axes[0, 0], measured, 'Measured direct beam', c_meas),
                                     (axes[0, 1], scaled, 'Simulated direct beam (scaled to measured total)', c_sim)):
        mesh = ax.imshow(np.maximum(image[xs, ys].T, 0.1), origin='lower', extent=extent, aspect='auto', norm=LogNorm(vmin=1, vmax=vmax), cmap='jet')
        ax.plot(centre[0] * 1e3, centre[1] * 1e3, 'x', color='red', markersize=12, mew=2.5, label='centroid')
        if beam_radius and beam_radius > 0:
            ax.add_patch(Circle((centre[0] * 1e3, centre[1] * 1e3), beam_radius * 1e3, fill=False, ec='red', ls='--', lw=1.5,
                                label=f'centroid window ({beam_radius * 1e3:.0f} mm)'))
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(title)
        ax.set_xlabel('x (NeXus, from beam axis) [mm]')
        ax.set_ylabel('y (NeXus, from beam axis) [mm]')
        fig.colorbar(mesh, ax=ax)
    for ax, axis, coords, label in ((axes[1, 0], 1, x_lab, 'x'), (axes[1, 1], 0, y_lab, 'y')):
        ax.plot(coords * 1e3, measured.sum(axis=axis), 'b.-', label='measured')
        ax.plot(coords * 1e3, scaled.sum(axis=axis), 'g.-', label='simulated (scaled)')
        ax.axvline(c_meas[1 - axis] * 1e3, color='b', ls='--')
        ax.axvline(c_sim[1 - axis] * 1e3, color='g', ls='--')
        lo, hi = (xs if axis == 1 else ys).start, (xs if axis == 1 else ys).stop
        ax.set_xlim(coords[lo] * 1e3, coords[min(hi, len(coords) - 1)] * 1e3)
        ax.set_yscale('log')
        ax.set_xlabel(f'{label} (NeXus, from beam axis) [mm]')
        ax.set_ylabel('counts (summed)')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    if figure == 'show':
        plt.show()
    else:
        path = f"{savename}.{figure}"
        plt.savefig(path)
        print(f"Created {path}")


def measure_incident_angle(sample_filepath, direct_beam_filepath, instrument_name='d22', sample_orientation=1,
                           nxs_data_path=None, min_separation_pixels=3, beam_radius=DEFAULT_BEAM_RADIUS,
                           half_along=SPECULAR_HALF_ALONG, half_across=SPECULAR_HALF_ACROSS, figure=None,
                           savename='incident_angle_check'):
    """
    Measure the real incident angle from a sample measurement: the specular spot lies 2*alpha from
    the direct beam (both at the same wavelength, so the gravity drop cancels). Returns alpha [deg].
    The direct beam is the windowed centroid of image_statistics. The specular spot starts at the
    brightest pixel displaced from the direct beam along the sample normal; its position is the
    intensity centroid in a window of +-half_along along the normal and +-half_across across it,
    re-centred on its own centroid until it settles. The window has to hold the whole spot: the
    specular of a wide beam is flat-topped, and a window around the brightest pixel alone is biased.
    """
    instrument = _instrument(instrument_name, [0.0, 0.0], None, 6.0, sample_orientation)
    det = instrument.detector
    direct = read_nexus_raw(direct_beam_filepath, nxs_data_path)
    sample = read_nexus_raw(sample_filepath, nxs_data_path)
    c_direct, _, x_rel, y_rel = image_statistics(direct, det, beam_radius)
    normal = np.array(det.coords.bornagain_to_nexus(0.0, 0.0, 1.0)[:2])  # sample normal in the detector plane
    normal = normal / np.linalg.norm(normal)
    tangent = np.array([normal[1], -normal[0]])
    X, Y = np.meshgrid(x_rel, y_rel, indexing='ij')
    along = (X - c_direct[0]) * normal[0] + (Y - c_direct[1]) * normal[1]
    across = (X - c_direct[0]) * tangent[0] + (Y - c_direct[1]) * tangent[1]
    pixel = max(det.pixel_size_x_nexus, det.pixel_size_y_nexus)
    candidates = np.where(along > min_separation_pixels * pixel, sample, 0.0)
    if candidates.max() <= 0:
        raise ValueError("No specular reflection found above the direct beam in the sample measurement.")
    i, j = np.unravel_index(np.argmax(candidates), candidates.shape)
    s, c = along[i, j], across[i, j]
    for _ in range(100):
        w = np.where((np.abs(along - s) <= half_along) & (np.abs(across - c) <= half_across), sample, 0.0)
        s_new, c = (w * along).sum() / w.sum(), (w * across).sum() / w.sum()
        converged = abs(s_new - s) < 1e-9
        s = s_new
        if converged:
            break
    alpha = float(np.rad2deg(0.5 * np.arctan(s / instrument.sample_detector_distance)))
    if figure:
        spot = c_direct + s * normal + c * tangent
        _incident_angle_figure(direct, sample, x_rel, y_rel, c_direct, spot, normal, tangent, beam_radius, half_along,
                               half_across, alpha, figure, savename)
    return alpha


def _incident_angle_figure(direct, sample, x_rel, y_rel, c_direct, spot, normal, tangent, beam_radius, half_along,
                           half_across, alpha, figure, savename):
    """Direct beam and sample images with the found centres (x) and the windows used to find them."""
    import matplotlib
    if figure != 'show':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle, Polygon
    extent = np.array([x_rel[0] - 0.5 * (x_rel[1] - x_rel[0]), x_rel[-1] + 0.5 * (x_rel[1] - x_rel[0]),
                       y_rel[0] - 0.5 * (y_rel[1] - y_rel[0]), y_rel[-1] + 0.5 * (y_rel[1] - y_rel[0])]) * 1e3
    corners = [(spot + a * half_along * normal + b * half_across * tangent) * 1e3 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    panels = ((axes[0], direct, 'Direct beam', c_direct, max(beam_radius or 0.0, 0.06) * 1.2, None),
              (axes[1], sample, f'Sample: specular {np.dot(spot - c_direct, normal) * 1e3:.1f} mm from the direct beam, '
                                f'alpha = {alpha:.4f} deg', 0.5 * (c_direct + spot), None, None),
              (axes[2], sample, 'Specular close-up (linear scale)', spot, None, 'linear'))
    for ax, image, title, centre, reach, scale in panels:
        if reach is None:
            reach = (0.5 * np.dot(spot - c_direct, normal) + 0.06) if ax is axes[1] else max(half_along, half_across) * 1.2
        norm = None if scale == 'linear' else LogNorm(vmin=max(image.max() * 1e-5, 0.5), vmax=image.max())
        ax.imshow(np.maximum(image.T, 0.5), origin='lower', extent=extent, norm=norm, cmap='viridis', interpolation='nearest', aspect='equal')
        ax.set_xlim((centre[0] - reach) * 1e3, (centre[0] + reach) * 1e3)
        ax.set_ylim((centre[1] - reach) * 1e3, (centre[1] + reach) * 1e3)
        if ax is not axes[2]:
            ax.plot(c_direct[0] * 1e3, c_direct[1] * 1e3, 'x', color='red', ms=12, mew=2.5, label='direct-beam centre')
        if ax is axes[0] and beam_radius and beam_radius > 0:
            ax.add_patch(Circle(c_direct * 1e3, beam_radius * 1e3, fill=False, ec='red', ls='--', lw=1.5,
                                label=f'direct-beam window ({beam_radius * 1e3:.0f} mm)'))
        if ax is not axes[0]:
            ax.plot(spot[0] * 1e3, spot[1] * 1e3, 'x', color='magenta', ms=12, mew=2.5, label='specular centre')
            ax.add_patch(Polygon(corners, closed=True, fill=False, ec='magenta', ls='--', lw=1.5,
                                 label=f'specular window (+-{half_along * 1e3:.0f} mm along the normal, +-{half_across * 1e3:.0f} mm across)'))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (NeXus, detector centre) [mm]')
        ax.set_ylabel('y (NeXus, detector centre) [mm]')
        ax.legend(loc='best', fontsize=7)
    plt.tight_layout()
    if figure == 'show':
        plt.show()
    else:
        path = f"{savename}.{figure}"
        plt.savefig(path)
        print(f"Created {path}")


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Find required detector centre_offset for a given NeXus data file.")
    parser.add_argument('filepath', type=str, help="Path to the NeXus data file.")
    parser.add_argument('--wavelength', type=float, default=6.0, help="Wavelength of the direct-beam measurement in Angstrom; enters through the gravity drop (default: 6.0).")
    parser.add_argument('--sample_orientation', type=int, default=1, choices=[0, 1, 2], help="Sample orientation (0, 1, 2); defines the plane in which the beam angle acts (default: 1).")
    parser.add_argument('--instrument', type=str.lower, default='d22', choices=list(instrument_defaults.keys()), help="Instrument name in instrument_defaults (default: 'd22').")
    parser.add_argument('--beam_angle', type=float, default=None, help="Beam angle in degrees: angle of the incident beam above the nominal beam axis, in the plane of incidence, positive towards the sample surface normal (default: the instrument's configured value, or 0). Note: opposite sign to the former --beam_declination.")
    parser.add_argument('--beam_declination', type=float, default=None, help=argparse.SUPPRESS)  # removed, see main
    parser.add_argument('--nxs_data_path', type=str, default=None, help='Explicit HDF5 path to the detector data inside the NeXus file, e.g. "entry0/data1/MultiDetector1_data". Overrides the default paths that are otherwise tried automatically.')
    parser.add_argument('--beam_radius', type=float, default=DEFAULT_BEAM_RADIUS, help="Radius [m] around the beam within which the intensity centroid is computed (iterated from the brightest pixel), so that background elsewhere on the detector does not bias it; <= 0 uses the whole detector (default: 0.1).")
    parser.add_argument('--verbose', action='store_true', help="Print the direct-beam centroid and the predicted landing point.")
    check = parser.add_argument_group('Cross-checks (optional)')
    check.add_argument('--mcpl', type=str, default=None, help="McStas MCPL file of the direct-beam configuration: ray-trace it with the found offset and compare the simulated with the measured direct beam (centroid residual, widths, beam angle, wavelength, intensity factor).")
    check.add_argument('--experiment_time', type=float, default=None, help="Duration of the direct-beam measurement [s], for the intensity factor (with --mcpl).")
    check.add_argument('--figure', choices=['png', 'pdf', 'show'], default=None, help="Figures with the found centres and the windows used to find them: the measured and simulated direct beam (with --mcpl) and the incident-angle measurement (with --sample_nxs). Save as png/pdf, or show them.")
    check.add_argument('--savename', type=str, default='beam_centre_check', help="Output file name (without extension) for --figure png/pdf (default: beam_centre_check; the incident-angle figure gets the suffix _incident_angle).")
    check.add_argument('--sample_nxs', type=str, default=None, help="Sample measurement (same wavelength and detector position): measure the real incident angle from the distance of the specular spot to the direct beam, and compare it with --alpha.")
    check.add_argument('--alpha', type=float, default=None, help="Intended incident angle [deg], compared with the angle measured from --sample_nxs.")
    return parser

def main():
    parser = create_argparser()
    args = parser.parse_args()
    if args.experiment_time and not args.mcpl:
        parser.error("--experiment_time requires --mcpl.")
    if args.figure and not (args.mcpl or args.sample_nxs):
        parser.error("--figure requires --mcpl or --sample_nxs.")
    if args.beam_declination is not None:
        parser.error("--beam_declination was renamed to --beam_angle, with the OPPOSITE sign "
                     f"(positive = beam rising towards the sample normal): use --beam_angle {-args.beam_declination}")

    try:
        offset = find_required_centre_offset(
            args.filepath,
            beam_angle=args.beam_angle,
            wavelength=args.wavelength,
            sample_orientation=args.sample_orientation,
            instrument_name=args.instrument,
            verbose=args.verbose,
            nxs_data_path=args.nxs_data_path,
            beam_radius=args.beam_radius
        )
        print(f"Calculated centre_offset [m]:  [{offset[0]:.6f}, {offset[1]:.6f}]")
        if args.mcpl:
            compare_with_simulated_direct_beam(args.filepath, args.mcpl, offset, beam_angle=args.beam_angle, wavelength=args.wavelength,
                                               sample_orientation=args.sample_orientation, instrument_name=args.instrument,
                                               nxs_data_path=args.nxs_data_path, experiment_time=args.experiment_time,
                                               figure=args.figure, savename=args.savename, beam_radius=args.beam_radius)
        if args.sample_nxs:
            alpha_measured = measure_incident_angle(args.sample_nxs, args.filepath, args.instrument, args.sample_orientation, args.nxs_data_path,
                                                    beam_radius=args.beam_radius, figure=args.figure,
                                                    savename=f"{args.savename}_incident_angle")
            print(f"\nIncident angle measured from the specular spot of {args.sample_nxs}: {alpha_measured:.4f} deg")
            if args.alpha is not None:
                print(f"Given --alpha: {args.alpha:.4f} deg (difference {alpha_measured - args.alpha:+.4f} deg)")
                if abs(alpha_measured - args.alpha) > max(0.05 * abs(args.alpha), 0.01):
                    print("WARNING: the measured incident angle differs from --alpha by more than 5% (or 0.01 deg): check the "
                          "incident angle, the sample orientation, and the sample alignment.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
