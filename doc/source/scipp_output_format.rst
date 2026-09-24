====================
Scipp Output Format
====================

The ``mcstas_gisans`` simulation output is standardized using a deeply nested **Scipp DataGroup** (``.h5`` format). 
This architecture ensures that regardless of whether the simulation is Time-of-Flight (TOF) or monochromatic (non-TOF), the metadata layout is the same, and the file records everything needed to interpret and reproduce the simulation.

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
The simulated detector image is stored in a ``sc.DataArray`` with one entry per detector pixel along the ``detector_id`` dimension.

* ``data``: the sum of the statistical weights of the simulated rays that hit the pixel. The MCPL weights are intensities (neutrons per second), so the values are **rates** (the unit is labelled ``counts``); ``mg_plot``/``mg_fit`` multiply them by ``--experiment_time`` to get expected counts. The variances are the Monte Carlo variances (sum of the squared weights), **not** Poisson counting variances.

  * **Non-TOF** instruments: one value per pixel.
  * **TOF** instruments: *binned* (event-mode) data, one bin per pixel holding the individual weighted events with their ``tof`` coordinate (seconds, T0-corrected time of flight from the source to the detector). Nothing is histogrammed in TOF at save time, so the wavelength can be sliced later (``mg_plot --wavelength_slice``); the file size scales with the number of events (particles × outgoing directions that hit the detector).

* ``coords['position']`` (vectors, m): the NeXus-frame position of each pixel centre relative to the sample, including the detector offset. The pixels are ordered with x (NeXus) as the slow and y as the fast index: ``data.values.reshape(pixels_x, pixels_y)``.
* ``detector_id`` is a dimension without a coordinate for non-TOF data; for TOF data it is the bin-edge coordinate of the event binning.

2. The ``instrument`` Block
---------------------------
A ``sc.DataGroup`` containing all relevant, standardized scalar metadata describing the physical layout and state of the instrument during the simulation. 

* ``name`` (scalar): The identifier of the instrument (e.g., 'd22', 'skadi').
* ``is_tof_instrument`` (scalar bool): Indicates if the instrument operates in TOF mode.
* ``detector_centre_offset_x`` / ``detector_centre_offset_y`` (scalar, meters): The detector centre position relative to the undeflected beam axis through the sample (NeXus frame).
* ``alpha_inc_deg`` (scalar, degrees): The incident angle of the beam.
* ``beam_angle`` (scalar, degrees): The beam angle used for the simulation (``--instrument_beam_angle`` or the instrument default, otherwise 0): the angle of the incident beam above the nominal beam axis, towards the sample surface normal.
* ``sample_orientation`` / ``sample_position`` / ``source_position``: Geometric alignment constants.
* ``parameters_json`` (scalar string): The complete, resolved instrument parameter dictionary used for the simulation (defaults plus command line overrides, including the detector size/pixels/resolution/offset and the beam angle actually used), as JSON. ``mg_plot`` rebuilds the instrument from it.
* ``no_gravity`` / ``wfm`` (scalar bool): Whether the simulation ignored gravity, and whether Wavelength Frame Multiplication mode was used.
* ``wavelength_selected`` (scalar, Angstroms): the selected wavelength (present **only** for non-TOF instruments; for TOF instruments the wavelength of each event is calculated from its TOF and the flight path).

3. The ``sample`` Block
-----------------------
A complete encapsulation of the simulated physical sample to guarantee 100% reproducibility of the physics setup without needing to rely on external files.

* ``name`` (scalar): The name of the built-in or custom sample Python module.
* ``arguments_json`` (scalar string): The exact keyword arguments passed to the sample builder via the CLI (serialized as JSON).
* ``script_content`` (scalar string): The full, literal Python source code of the sample module used to build the BornAgain layout.

4. The ``mcpl`` Block
---------------------
Preserves upstream metadata injected by the McStas ray-tracer. This block is conditionally added *only* if the input particles were read from an MCPL file.

* ``filename`` (scalar): The absolute path of the MCPL file.
* ``file_size_bytes`` (scalar): Its size, to recognise a changed file.
* ``sourcename`` (scalar): The upstream component name that generated the MCPL file.
* ``nparticles`` (scalar): The total number of particles stored in the MCPL file.
* ``comments`` (scalar string): Any descriptive comments embedded in the MCPL header.

5. The ``provenance`` Block
---------------------------
The exact software versions and command-line execution state used to generate this file.

* ``cli_command`` (scalar string): The literal shell command executed.
* ``cli_args_json`` (scalar string): The fully parsed argparse namespace (including all default values), serialized as JSON.
* ``random_seed`` (scalar): The seed of the Monte Carlo sampling (``--seed``); rerunning with it reproduces the file exactly.
* ``bornagain_version`` (scalar string): The version of the BornAgain library used for the DWBA calculations.
* ``mcstas_gisans_version`` (scalar string): The version of the ``mcstas_gisans`` Python package.
* ``mcstas_gisans_git_commit`` (scalar string): ``git describe`` of the source tree when run from a git checkout (``-dirty`` if it had uncommitted changes).
* ``mcpl_version`` (scalar string): The version of the underlying MCPL library.
* ``timestamp`` (scalar string): The ISO 8601 timestamp of when the simulation was executed.
