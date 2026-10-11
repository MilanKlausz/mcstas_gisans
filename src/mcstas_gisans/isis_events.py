"""
Measured ISIS SANS2D events in the mcstas_gisans TOF output format.

The detection events of a raw ISIS SANS2D NeXus file (raw_data_1/detector_1_events: the spectrum of each event and
its TOF from the proton pulse) are written as a Scipp DataGroup like the TOF output of mg_run: events with
detector_id (= ix * 120 + iy of the rear tube detector, see nexus_reader.read_sans2d_raw) and tof [s], weight 1,
binned per pixel. The instrument, sample and pixel-position metadata are taken from a simulation output file of the
same instrument geometry (TEMPLATE), so the measurement goes through exactly the TOF reduction of mg_plot (Q per
event, --wavelength_slice) and can be compared with that simulation (e.g. with --split_view).

Usage: python -m mcstas_gisans.isis_events RAW.nxs TEMPLATE_SIMULATION.h5 OUTPUT.h5
"""
import argparse

import h5py
import numpy as np

from .nexus_reader import SANS2D_MONITOR_SPECTRA, sans2d_rear_pixel_row


def read_sans2d_events(filepath):
    """
    Read the rear-detector events of a raw ISIS SANS2D NeXus file.

    Returns
    -------
    dict
        ``pixel`` (along the tubes, +x), ``row`` (tube row from the bottom, +y) and ``tof`` [s] of every
        rear-detector event, the total number of events in the file (``n_events``), ``title`` and ``duration`` [s].
    """
    with h5py.File(filepath, 'r') as file:
        raw = file['raw_data_1']
        spectra = np.asarray(raw['detector_1/spectrum_index'][()])
        detector_ids = np.asarray(raw['isis_vms_compat/UDET'][SANS2D_MONITOR_SPECTRA:SANS2D_MONITOR_SPECTRA + len(spectra)])
        event_spectra = np.asarray(raw['detector_1_events/event_id'][()])
        tof = np.asarray(raw['detector_1_events/event_time_offset'][()], dtype=np.float64) * 1e-6  # us -> s
        title = raw['title'][0].decode() if 'title' in raw else ''
        duration = float(raw['duration'][0]) if 'duration' in raw else None
    order = np.argsort(spectra)
    index = order[np.searchsorted(spectra[order], event_spectra)]
    if np.any(spectra[index] != event_spectra):
        raise ValueError(f"{filepath}: events of spectra that are not in detector_1/spectrum_index.")
    ids = detector_ids[index]
    rear = ids // 1000000 == 1
    pixel, row = sans2d_rear_pixel_row(ids[rear])
    return {'pixel': pixel, 'row': row, 'tof': tof[rear], 'n_events': len(event_spectra), 'title': title,
            'duration': duration}


def events_to_datagroup(events, template):
    """The events (read_sans2d_events) as a Scipp DataGroup with the metadata of the template DataGroup."""
    import scipp as sc
    positions = template['data'].coords['position']
    n_y = 120
    detector_id = (events['pixel'] * n_y + events['row']).astype(np.int32)
    n = len(detector_id)
    data = sc.DataArray(
        data=sc.array(dims=['event'], values=np.ones(n), variances=np.ones(n), unit='counts'),
        coords={'detector_id': sc.array(dims=['event'], values=detector_id, unit=None),
                'tof': sc.array(dims=['event'], values=events['tof'], unit='s')})
    binned = data.bin(detector_id=sc.arange('detector_id', 0, positions.sizes['detector_id'] + 1, unit=None))
    binned.coords['position'] = positions
    content = {'data': binned}
    for key in ('instrument', 'sample'):
        if key in template:
            content[key] = template[key]
    content['provenance'] = sc.DataGroup({'measured_title': sc.scalar(events['title']),
                                          'measured_duration_s': sc.scalar(float(events['duration'] or 0.0))})
    return sc.DataGroup(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('raw', help='Raw ISIS SANS2D NeXus file (with raw_data_1/detector_1_events).')
    parser.add_argument('template', help='mcstas_gisans TOF simulation output (.h5) of the same instrument geometry '
                                         '(the source of the instrument metadata and pixel positions).')
    parser.add_argument('output', help='Output file (.h5).')
    args = parser.parse_args()
    import scipp as sc
    template = sc.io.load_hdf5(args.template)
    events = read_sans2d_events(args.raw)
    if template['data'].coords['position'].sizes['detector_id'] != 512 * 120:
        parser.error('The template simulation does not have the 512 x 120 pixel SANS2D rear detector.')
    events_to_datagroup(events, template).save_hdf5(args.output)
    print(f"{args.raw} ('{events['title']}'): {len(events['tof'])} rear-detector events of {events['n_events']} "
          f"-> {args.output}")


if __name__ == '__main__':
    main()
