====================
Scipp Output Format
====================

The ``mcstas_gisans`` simulation output is standardized using a deeply nested **Scipp DataGroup** (``.h5`` format). 
This architecture ensures that regardless of whether the simulation is Time-of-Flight (TOF) or monochromatic (non-TOF), the output file shape, coordinate systems, and metadata extraction paths remain identical and adhere to FAIR data principles.

Below is an overview of the output hierarchy and the purpose of each data block.

.. code-block:: python

    dataset = sc.DataGroup({
        'data': sc.DataArray(...),
        'instrument': sc.DataGroup(...),
        'sample': sc.DataGroup(...),
        'provenance': sc.DataGroup(...),
        'mcpl': sc.DataGroup(...) # Only present if MCPL input was used
    })

1. The ``data`` Block
---------------------
The core physics arrays—the binned intensity counts and automatically tracked statistical variances—are stored in a standard ``sc.DataArray``. Coordinates required natively for dimension alignment, slicing, or immediate plotting are bound directly to the ``DataArray``'s coords.

* ``data`` (array): The intensity counts, with built-in (Poisson) variances.

  * For **non-TOF** instruments, this is a flat ``sc.DataArray`` with one bin per detector pixel along a ``detector_id`` dimension (reshape using the detector's pixel grid, e.g. via ``coords['position']``, for 2D plotting).
  * For **TOF** instruments, this is *binned* (event-mode) data: still one outer bin per ``detector_id``, but each bin holds the individual weighted events (with their own ``tof`` coordinate) that landed on that pixel, rather than a single summed count. No TOF-axis histogramming is performed at save time — the exact TOF of every event is preserved for flexible slicing later (e.g. with ``mg_plot --wavelength_slice``), at the cost of output file size scaling with the number of simulated events rather than a fixed number of TOF bins.

* ``coords['position']`` (vectors): The physical 3D position of each detector pixel, indexed by ``detector_id``.
* ``coords['detector_id']``: The integer pixel index each event/bin belongs to.
* ``coords['tof']`` (array, TOF only, inside the per-pixel event bins): The individual Time-of-Flight event timestamps in seconds — not pre-binned TOF edges.

2. The ``instrument`` Block
---------------------------
A ``sc.DataGroup`` containing all relevant, standardized scalar metadata describing the physical layout and state of the instrument during the simulation. 

* ``name`` (scalar): The identifier of the instrument (e.g., 'd22', 'skadi').
* ``is_tof_instrument`` (scalar bool): Indicates if the instrument operates in TOF mode.
* ``detector_centre_offset_x`` / ``detector_centre_offset_y`` (scalar, meters): The physical misalignment corrections applied to map the beam center to the Nexus grid.
* ``alpha_inc_deg`` (scalar, degrees): The incident angle of the beam.
* ``beam_angle`` (scalar, degrees): The beam declination angle relative to the nominal beam axis (either provided via ``--instrument_beam_angle``, or otherwise calculated automatically from the average particle velocities in the input file).
* ``sample_orientation`` / ``sample_position`` / ``source_position``: Geometric alignment constants.
* ``wavelength_selected`` (scalar, Angstroms): The fixed monochromatic wavelength (Present **only** for non-TOF instruments; for TOF, wavelength is calculated dynamically per-bin or stored in the provenance CLI args).

3. The ``sample`` Block
-----------------------
A complete encapsulation of the simulated physical sample to guarantee 100% reproducibility of the physics setup without needing to rely on external files.

* ``name`` (scalar): The name of the built-in or custom sample Python module.
* ``arguments_json`` (scalar string): The exact keyword arguments passed to the sample builder via the CLI (serialized as JSON).
* ``script_content`` (scalar string): The full, literal Python source code of the sample module used to build the BornAgain layout.

4. The ``mcpl`` Block
---------------------
Preserves upstream metadata injected by the McStas ray-tracer. This block is conditionally added *only* if the input particles were read from an MCPL file.

* ``filename`` (scalar): The path to the MCPL file.
* ``sourcename`` (scalar): The upstream component name that generated the MCPL file.
* ``nparticles`` (scalar): The total number of particles stored in the MCPL file.
* ``comments`` (scalar string): Any descriptive comments embedded in the MCPL header.

5. The ``provenance`` Block
---------------------------
The exact software versions and command-line execution state used to generate this file.

* ``cli_command`` (scalar string): The literal shell command executed.
* ``cli_args_json`` (scalar string): The fully parsed argparse namespace (including all default values), serialized as JSON.
* ``bornagain_version`` (scalar string): The version of the BornAgain library used for the DWBA calculations.
* ``mcstas_gisans_version`` (scalar string): The version of the ``mcstas_gisans`` Python package.
* ``mcpl_version`` (scalar string): The version of the underlying MCPL library.
* ``timestamp`` (scalar string): The ISO 8601 timestamp of when the simulation was executed.
