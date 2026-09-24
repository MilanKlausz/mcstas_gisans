
"""
Instrument parameters used for data reduction
"""

# 't0_monitor_name' is required to enable t0 correction based on McStas TOFLambda monitor
# Wavelength Frame Multiplication requires the following options
#   'wfm_t0_monitor_name'
#   'wfm_virtual_source_distance'

# 'detector' property is not required, but if added, it is expected to have
# all properties listed in the default_detector object below

instrument_defaults = {
    'saga': {
        'nominal_source_sample_distance' : 55, #[m]
        'sample_detector_distance' : 10, #[m] along the y axis

        'tof_instrument' : True,
        'mcpl_monitor_name' : 'Mcpl_TOF_Lambda',
        't0_monitor_name' : 'Source_TOF_Lambda',
        'wfm_t0_monitor_name' : 'toflambdawfmc',
        'wfm_virtual_source_distance': 8.2, #real source to virtual source distance for WFM mode
    },
    'loki': {
        'nominal_source_sample_distance' : 23.6,
        'sample_detector_distance' : 10, #can be 5-10m
        'tof_instrument' : True,
        'mcpl_monitor_name' : 'Mcpl_TOF_Lambda',
        't0_monitor_name' : 'Source_TOF_Lambda',
    },
    'skadi': {
        'nominal_source_sample_distance' : 38.43,
        'sample_detector_distance' : 12, #can be 4-20m
        'tof_instrument' : True,
        'detector': {
            'size': [1.024, 1.024], #[m]
            'direct_beam_centre_offset': [0.0, 0.0],
            'pixels': [128, 256],
            'resolution': [0.0, 0.0] #fwhm[m]
        }
    },
    'd22': { #ILL
        'nominal_source_sample_distance' : 61.28, #approximate value, but it is not really used
        'sample_detector_distance' : 17.6,

        'tof_instrument' : False,
        't0_monitor_name' : 'Source_TOF_Lambda',
        'detector': {
            'size': [1.024, 1.024], #[m]
            'direct_beam_centre_offset': [0.0, 0.0], #[m]
            'pixels': [128, 256],
            'resolution': [0.0, 0.004] #fwhm[m]
        },
    }
}

default_detector = {
    'size': [1.024, 1.024], #[m]
    'direct_beam_centre_offset': [0.0, 0.0],
    'pixels': [256, 256],
    'resolution': [0.0, 0.0] #fwhm[m]
}

#required keys in the instrument_defaults to enable WFM(wavelength frame multiplication) mode
required_keys_for_wfm = ['wfm_t0_monitor_name', 'wfm_virtual_source_distance']

# temporary hard-coded sub-pulse tof limits for the SAGA instrument
saga_subpulse_tof_limits = [
    [10200, 12000],
    [12000, 14300],
    [14300, 16100],
    [16100, 18000]
]

def get_saga_subpulse_tof_limits(wavelength):
    """
    Get hard-coded TOF limits of a WFM sub-pulse in between the WFM choppers
    for the SAGA instrument, depending on the wavelength
    """
    if wavelength < 5.15:
        subpulse_id = 0
    elif wavelength < 6.15:
        subpulse_id = 1
    elif wavelength < 7.1:
        subpulse_id = 2
    else:
        subpulse_id = 3

    return saga_subpulse_tof_limits[subpulse_id]

import copy

_initial_instrument_defaults = copy.deepcopy(instrument_defaults)

def reset_instrument_defaults():
    """Reset instrument_defaults dictionary to original initial values."""
    global instrument_defaults
    instrument_defaults.clear()
    for k, v in copy.deepcopy(_initial_instrument_defaults).items():
        instrument_defaults[k] = v

# CLI override flags (without prefix) -> instrument parameter keys. Detector keys live in params['detector'].
_OVERRIDE_KEYS = {
    'nominal_source_sample_distance': 'nominal_source_sample_distance',
    'sample_detector_distance': 'sample_detector_distance',
    't0_monitor_name': 't0_monitor_name',
    'wfm_t0_monitor_name': 'wfm_t0_monitor_name',
    'wfm_virtual_source_distance': 'wfm_virtual_source_distance',
    'beam_angle': 'beam_angle',
}
_DETECTOR_OVERRIDE_KEYS = {
    'detector_size': 'size',
    'detector_centre_offset': 'direct_beam_centre_offset',
    'detector_pixels': 'pixels',
    'detector_resolution': 'resolution',
}


def resolve_instrument_parameters(instrument_name, args=None, prefix='instrument_', base=None):
    """
    Return a NEW, fully resolved instrument parameter dict: the built-in defaults of
    `instrument_name` (or a copy of `base`, e.g. parameters stored in a simulation
    output file) with any command line overrides `--<prefix><key>` applied.
    The module-level instrument_defaults are never modified.
    """
    if base is None:
        if instrument_name not in instrument_defaults:
            raise KeyError(f"Unknown instrument '{instrument_name}'. Available: {list(instrument_defaults)}")
        base = instrument_defaults[instrument_name]
    params = copy.deepcopy(base)
    params.setdefault('detector', copy.deepcopy(default_detector))
    if args is not None:
        for flag, key in _OVERRIDE_KEYS.items():
            value = getattr(args, prefix + flag, None)
            if value is not None:
                params[key] = value
        tof = getattr(args, prefix + 'tof_instrument', None)
        if tof is not None:
            params['tof_instrument'] = (tof == 'true')
        for flag, key in _DETECTOR_OVERRIDE_KEYS.items():
            value = getattr(args, prefix + flag, None)
            if value is not None:
                params['detector'][key] = list(value)
    params['name'] = instrument_name
    return params


def set_instrument_parameters(args, instrument_name=None):
    """
    Resolve the instrument parameters of the selected instrument with the command line
    overrides, store them as args.instrument_params (the single source of instrument
    parameters for the rest of the run) and return them.
    """
    instr_name = (getattr(args, 'instrument', None) if args is not None else None) or instrument_name or 'd22'
    params = resolve_instrument_parameters(instr_name, args)
    if args is not None:
        args.instrument_params = params
    return params


def get_instrument_parameters(args):
    """Resolved instrument parameters of a run (args.instrument_params), resolving them if needed."""
    params = getattr(args, 'instrument_params', None)
    return params if params is not None else set_instrument_parameters(args)


def get_nxs_instrument_parameters(args, default_instr_name='d22', base=None):
    """
    Instrument parameters for interpreting measured NeXus data: `base` (e.g. the
    parameters of the simulation being compared, if it is the same instrument) or the
    built-in defaults, with the --nxs_instrument_* overrides applied. Returns a new dict.
    """
    instr_name = getattr(args, 'nxs_instrument_name', None) if args is not None else None
    if instr_name is None:
        instr_name = default_instr_name
    elif base is not None and base.get('name') != instr_name:
        base = None
    return resolve_instrument_parameters(instr_name, args, prefix='nxs_instrument_', base=base)
