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

BEAM_WINDOW_WIDTHS = 3.0  # automatic direct-beam window: half-sizes of this many RMS widths of the beam, per axis
SPECULAR_ALONG_FACTOR = 1.5  # automatic specular window: the direct-beam half-size across the normal, this times it along
INITIAL_WINDOW_PIXELS = 5  # half-size [pixels] of the first window around the starting point
EDGE_WARNING_FRACTION = 0.01  # warn if this fraction of a window's counts lies in detector pixels at the window's cut edge


def find_required_centre_offset(filepath, beam_angle=None, wavelength=6.0, sample_orientation=1, instrument_name='d22', verbose=False, nxs_data_path=None, beam_window=None):
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
    beam_window : sequence of 2 floats, optional
        Half-sizes [m] of the rectangular centroid window along the detector x and y axes
        (default: BEAM_WINDOW_WIDTHS RMS widths of the beam, see beam_spot).

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

    spot = beam_spot(raw, det, beam_window, label=f"direct beam of {filepath}")
    centroid = spot['centre']
    how = "as given" if beam_window is not None else f"{BEAM_WINDOW_WIDTHS:g} RMS widths of the beam"
    print(f"Direct-beam window: +-{spot['half'][0] * 1e3:.1f} mm (x) x +-{spot['half'][1] * 1e3:.1f} mm (y), {how}. "
          f"To set it: --beam_window {spot['half'][0]:.4f} {spot['half'][1]:.4f}")

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


def _pixel_coordinates(detector):
    x_rel = (np.arange(detector.pixels_x_nexus) + 0.5) * detector.pixel_size_x_nexus - 0.5 * detector.size_x_nexus
    y_rel = (np.arange(detector.pixels_y_nexus) + 0.5) * detector.pixel_size_y_nexus - 0.5 * detector.size_y_nexus
    return x_rel, y_rel


def _smoothed_maximum(image, allowed=None):
    """Pixel of the maximum of the 3x3-median-filtered image, optionally among allowed pixels: an isolated hot
    pixel is removed by the median, however many counts it has, while an extended spot keeps its level."""
    padded = np.pad(image, 1)
    neighbours = np.stack([padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dy:padded.shape[1] - 1 + dy]
                           for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    filtered = np.median(neighbours, axis=0)
    if allowed is not None:
        filtered = np.where(allowed, filtered, -np.inf)
    return np.unravel_index(np.argmax(filtered), filtered.shape)


def _window_centroid(image, U, V, centre, half, max_iterations=100):
    """Intensity centroid and RMS width in the rectangle |U - cu| <= half[0], |V - cv| <= half[1], re-centred until it settles."""
    centre = np.asarray(centre, dtype=float)
    for _ in range(max_iterations):
        inside = (np.abs(U - centre[0]) <= half[0]) & (np.abs(V - centre[1]) <= half[1])
        w = np.where(inside, image, 0.0)
        total = w.sum()
        if total <= 0:
            raise ValueError("No counts in the centroid window.")
        new_centre = np.array([(w * U).sum(), (w * V).sum()]) / total
        converged = np.allclose(new_centre, centre, rtol=0, atol=1e-9)
        centre = new_centre
        if converged:
            break
    width = np.sqrt(np.array([(w * (U - centre[0]) ** 2).sum(), (w * (V - centre[1]) ** 2).sum()]) / total)
    return centre, width, inside


def _warn_if_cut(image, inside, label):
    """Warn if the window reaches the detector edge and a noticeable part of its counts lies in the edge pixels."""
    edge = np.zeros(image.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    total = image[inside].sum()
    if total > 0 and image[inside & edge].sum() > EDGE_WARNING_FRACTION * total:
        print(f"WARNING: the {label} reaches the edge of the detector ({100 * image[inside & edge].sum() / total:.1f}% of its "
              f"counts are in the outermost pixels): the spot may be cut off and its centre biased.")


def beam_spot(image, detector, window=None, label="direct beam"):
    """
    Centre, RMS width and window of the beam spot in a raw detector image [ix, iy] (NeXus frame, m,
    relative to the detector centre). The search starts at the maximum of the 3x3-median-filtered image. The
    centroid is taken in a rectangle aligned with the detector axes and re-centred until it settles;
    by default (window=None) the half-sizes of the rectangle are BEAM_WINDOW_WIDTHS RMS widths of the
    spot along each axis, measured and updated together with the centre, starting from
    INITIAL_WINDOW_PIXELS pixels. This works for any beam and pixel size: a rectangular beam of full
    width W has an RMS width of W/sqrt(12), so 3 RMS widths give a half-size of 0.87 W. An explicit
    window [half_x, half_y] (m) is used as given. Background elsewhere on the detector does not bias
    the result. Returns a dict with 'centre', 'width', 'half' (window half-sizes) and 'inside' (mask).
    """
    x_rel, y_rel = _pixel_coordinates(detector)
    X, Y = np.meshgrid(x_rel, y_rel, indexing='ij')
    i, j = _smoothed_maximum(image)
    centre = np.array([x_rel[i], y_rel[j]])
    pixel = np.array([detector.pixel_size_x_nexus, detector.pixel_size_y_nexus])
    if window is not None:
        half = np.asarray(window, dtype=float)
        centre, width, inside = _window_centroid(image, X, Y, centre, half)
    else:
        half = INITIAL_WINDOW_PIXELS * pixel
        for _ in range(50):
            centre, width, inside = _window_centroid(image, X, Y, centre, half)
            new_half = np.maximum(BEAM_WINDOW_WIDTHS * width, pixel)
            converged = np.allclose(new_half, half, rtol=1e-4, atol=0)
            half = new_half
            if converged:
                break
        centre, width, inside = _window_centroid(image, X, Y, centre, half)
    _warn_if_cut(image, inside, f"window of the {label}")
    return {'centre': centre, 'width': width, 'half': half, 'inside': inside}


def image_statistics(image, detector, window=None):
    """Intensity centroid and RMS width [m] of the beam spot (see beam_spot), and the pixel coordinates."""
    spot = beam_spot(image, detector, window)
    x_rel, y_rel = _pixel_coordinates(detector)
    return spot['centre'], spot['width'], x_rel, y_rel


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
                                       savename='beam_centre_check', residual_warning_pixels=0.5, beam_window=None):
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
    spot_meas = beam_spot(measured, det, beam_window, label="measured direct beam")
    spot_sim = beam_spot(simulated, det, spot_meas['half'], label="simulated direct beam")  # the same window size
    c_meas, w_meas, c_sim, w_sim = spot_meas['centre'], spot_meas['width'], spot_sim['centre'], spot_sim['width']
    x_rel, y_rel = _pixel_coordinates(det)
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
                            spot_meas['half'])
    return results


def _direct_beam_figure(measured, simulated, x_lab, y_lab, c_meas, c_sim, figure, savename, half):
    import matplotlib
    if figure != 'show':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    scaled = simulated * measured.sum() / simulated.sum()
    from matplotlib.patches import Rectangle
    ix, iy = np.unravel_index(np.argmax(measured), measured.shape)
    pitch_x, pitch_y = abs(x_lab[1] - x_lab[0]), abs(y_lab[1] - y_lab[0])
    hx, hy = int(np.ceil(1.3 * half[0] / pitch_x)) + 2, int(np.ceil(1.3 * half[1] / pitch_y)) + 2  # show the whole window
    xs, ys = slice(max(ix - hx, 0), ix + hx + 1), slice(max(iy - hy, 0), iy + hy + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    extent = [x_lab[xs][0] * 1e3, x_lab[xs][-1] * 1e3, y_lab[ys][0] * 1e3, y_lab[ys][-1] * 1e3]
    vmax = max(measured[xs, ys].max(), scaled[xs, ys].max())
    for ax, image, title, centre in ((axes[0, 0], measured, 'Measured direct beam', c_meas),
                                     (axes[0, 1], scaled, 'Simulated direct beam (scaled to measured total)', c_sim)):
        mesh = ax.imshow(np.maximum(image[xs, ys].T, 0.1), origin='lower', extent=extent, aspect='auto', norm=LogNorm(vmin=1, vmax=vmax), cmap='jet')
        ax.plot(centre[0] * 1e3, centre[1] * 1e3, 'x', color='red', markersize=12, mew=2.5, label='centroid')
        ax.add_patch(Rectangle(((centre[0] - half[0]) * 1e3, (centre[1] - half[1]) * 1e3), 2 * half[0] * 1e3, 2 * half[1] * 1e3,
                               fill=False, ec='red', ls='--', lw=1.5,
                               label=f'centroid window (+-{half[0] * 1e3:.0f} x +-{half[1] * 1e3:.0f} mm)'))
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
                           nxs_data_path=None, min_separation_pixels=3, beam_window=None, specular_window=None,
                           alpha=None, figure=None, savename='incident_angle_check', return_details=False):
    """
    Measure the real incident angle from a sample measurement: the specular spot lies 2*alpha from
    the direct beam (both at the same wavelength, so the gravity drop cancels). Returns alpha [deg]
    (with return_details, a dict with the positions and windows as well).
    The direct beam is found with beam_spot. The specular window is a rectangle aligned with the
    sample normal; by default its half-size across the normal is that of the direct-beam window
    and SPECULAR_ALONG_FACTOR times it along the normal (a flat sample images the direct beam,
    slightly broadened), or specular_window = [half_along, half_across] (m). The search starts at
    the maximum of the 3x3-median-filtered sample image near the expected position L*tan(2*alpha) if alpha
    is given, else anywhere at least min_separation_pixels beyond the direct beam along the normal;
    the window is then re-centred on its centroid until it settles. The window has to hold the
    whole spot: the specular of a wide beam is flat-topped, and a window around its brightest
    pixels alone is biased.
    """
    instrument = _instrument(instrument_name, [0.0, 0.0], None, 6.0, sample_orientation)
    det = instrument.detector
    L = instrument.sample_detector_distance
    direct = read_nexus_raw(direct_beam_filepath, nxs_data_path)
    sample = read_nexus_raw(sample_filepath, nxs_data_path)
    beam = beam_spot(direct, det, beam_window, label=f"direct beam of {direct_beam_filepath}")
    c_direct = beam['centre']
    normal = np.array(det.coords.bornagain_to_nexus(0.0, 0.0, 1.0)[:2])  # sample normal in the detector plane
    normal = normal / np.linalg.norm(normal)
    tangent = np.array([normal[1], -normal[0]])
    x_rel, y_rel = _pixel_coordinates(det)
    X, Y = np.meshgrid(x_rel, y_rel, indexing='ij')
    along = (X - c_direct[0]) * normal[0] + (Y - c_direct[1]) * normal[1]
    across = (X - c_direct[0]) * tangent[0] + (Y - c_direct[1]) * tangent[1]
    pixel_along = abs(normal[0]) * det.pixel_size_x_nexus + abs(normal[1]) * det.pixel_size_y_nexus
    if specular_window is None:
        beam_along = abs(normal[0]) * beam['half'][0] + abs(normal[1]) * beam['half'][1]
        beam_across = abs(tangent[0]) * beam['half'][0] + abs(tangent[1]) * beam['half'][1]
        half = np.array([SPECULAR_ALONG_FACTOR * beam_along, beam_across])
        how = f"from the direct-beam window (x{SPECULAR_ALONG_FACTOR:g} along the normal)"
    else:
        half = np.asarray(specular_window, dtype=float)
        how = "as given"
    if alpha is not None:
        s_expected = L * np.tan(2 * np.radians(alpha))
        allowed = (np.abs(along - s_expected) <= max(3 * half[0], 5 * pixel_along)) & (np.abs(across) <= half[1])
    else:
        allowed = along > min_separation_pixels * max(det.pixel_size_x_nexus, det.pixel_size_y_nexus)
    if not np.any(allowed & (sample > 0)):
        raise ValueError("No specular reflection found in the sample measurement (check --alpha and --sample_orientation).")
    i, j = _smoothed_maximum(sample, allowed)
    (s, c), _, inside = _window_centroid(sample, along, across, (along[i, j], across[i, j]), half)
    _warn_if_cut(sample, inside, f"specular window of {sample_filepath}")
    print(f"Specular window: +-{half[0] * 1e3:.1f} mm along the sample normal x +-{half[1] * 1e3:.1f} mm across it, {how}. "
          f"To set it: --specular_window {half[0]:.4f} {half[1]:.4f}")
    measured_alpha = float(np.rad2deg(0.5 * np.arctan(s / L)))
    spot = c_direct + s * normal + c * tangent
    if figure:
        _incident_angle_figure(direct, sample, x_rel, y_rel, c_direct, spot, normal, tangent, beam['half'], half,
                               measured_alpha, figure, savename)
    if return_details:
        return {'alpha': measured_alpha, 'separation': float(s), 'direct_beam': c_direct, 'specular': spot,
                'beam_half': beam['half'], 'specular_half': half, 'normal': normal, 'sample_detector_distance': L}
    return measured_alpha


def _incident_angle_figure(direct, sample, x_rel, y_rel, c_direct, spot, normal, tangent, beam_half, specular_half,
                           alpha, figure, savename):
    """Direct beam and sample images with the found centres (x) and the windows used to find them."""
    import matplotlib
    if figure != 'show':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon, Rectangle
    extent = np.array([x_rel[0] - 0.5 * (x_rel[1] - x_rel[0]), x_rel[-1] + 0.5 * (x_rel[1] - x_rel[0]),
                       y_rel[0] - 0.5 * (y_rel[1] - y_rel[0]), y_rel[-1] + 0.5 * (y_rel[1] - y_rel[0])]) * 1e3
    half_along, half_across = specular_half
    corners = [(spot + a * half_along * normal + b * half_across * tangent) * 1e3 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    separation = np.dot(spot - c_direct, normal)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    panels = ((axes[0], direct, 'Direct beam', c_direct, 1.25 * max(beam_half), None),
              (axes[1], sample, f'Sample: specular {separation * 1e3:.1f} mm from the direct beam, alpha = {alpha:.4f} deg',
               0.5 * (c_direct + spot), 0.5 * separation + 1.3 * max(specular_half), None),
              (axes[2], sample, 'Specular close-up (linear scale)', spot, 1.25 * max(specular_half), 'linear'))
    for ax, image, title, centre, reach, scale in panels:
        norm = None if scale == 'linear' else LogNorm(vmin=max(image.max() * 1e-5, 0.5), vmax=image.max())
        ax.imshow(np.maximum(image.T, 0.5), origin='lower', extent=extent, norm=norm, cmap='viridis', interpolation='nearest', aspect='equal')
        ax.set_xlim((centre[0] - reach) * 1e3, (centre[0] + reach) * 1e3)
        ax.set_ylim((centre[1] - reach) * 1e3, (centre[1] + reach) * 1e3)
        if ax is not axes[2]:
            ax.plot(c_direct[0] * 1e3, c_direct[1] * 1e3, 'x', color='red', ms=12, mew=2.5, label='direct-beam centre')
        if ax is axes[0]:
            ax.add_patch(Rectangle(((c_direct[0] - beam_half[0]) * 1e3, (c_direct[1] - beam_half[1]) * 1e3), 2 * beam_half[0] * 1e3,
                                   2 * beam_half[1] * 1e3, fill=False, ec='red', ls='--', lw=1.5,
                                   label=f'direct-beam window (+-{beam_half[0] * 1e3:.0f} x +-{beam_half[1] * 1e3:.0f} mm)'))
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
    parser.add_argument('--beam_angle', type=float, default=None, help="Beam angle in degrees: angle of the incident beam above the nominal beam axis, in the plane of incidence, positive towards the sample surface normal (default: the instrument's configured value, or 0).")
    parser.add_argument('--beam_declination', type=float, default=None, help=argparse.SUPPRESS)  # removed, see main
    parser.add_argument('--nxs_data_path', type=str, default=None, help='Explicit HDF5 path to the detector data inside the NeXus file, e.g. "entry0/data1/MultiDetector1_data". Overrides the default paths that are otherwise tried automatically.')
    parser.add_argument('--beam_window', type=float, nargs=2, metavar=('HALF_X', 'HALF_Y'), default=None, help=f"Half-sizes [m] of the rectangular window (along the detector x and y axes) in which the direct-beam centroid is computed. Default: {BEAM_WINDOW_WIDTHS:g} RMS widths of the beam on each axis, determined from the image (printed with the option that reproduces it).")
    parser.add_argument('--specular_window', type=float, nargs=2, metavar=('HALF_ALONG', 'HALF_ACROSS'), default=None, help=f"Half-sizes [m] of the rectangular window (along and across the sample normal) in which the specular centroid is computed (with --sample_nxs). Default: the direct-beam window size across the normal and {SPECULAR_ALONG_FACTOR:g} times it along the normal (printed with the option that reproduces it).")
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
            beam_window=args.beam_window
        )
        print(f"Calculated centre_offset [m]:  [{offset[0]:.6f}, {offset[1]:.6f}]")
        if args.mcpl:
            compare_with_simulated_direct_beam(args.filepath, args.mcpl, offset, beam_angle=args.beam_angle, wavelength=args.wavelength,
                                               sample_orientation=args.sample_orientation, instrument_name=args.instrument,
                                               nxs_data_path=args.nxs_data_path, experiment_time=args.experiment_time,
                                               figure=args.figure, savename=args.savename, beam_window=args.beam_window)
        if args.sample_nxs:
            alpha_measured = measure_incident_angle(args.sample_nxs, args.filepath, args.instrument, args.sample_orientation, args.nxs_data_path,
                                                    beam_window=args.beam_window, specular_window=args.specular_window,
                                                    alpha=args.alpha, figure=args.figure,
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
