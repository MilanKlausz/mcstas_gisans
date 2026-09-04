===============================
Architecture & Workflow Details
===============================

This document provides in-depth technical details about the architecture of `mcstas_gisans`, the simulation workflows, and the data structures used.

1. Workflows: TOF vs. Non-TOF
-----------------------------

The simulation workflow diverges depending on whether the experiment utilizes Time-of-Flight (TOF) data or non-TOF data (this is controlled by the ``tof_instrument`` key for the selected instrument in ``instrument_defaults.py``, see :ref:`instrument_defaults_schema` below):

* **Non-TOF Workflow**: The ``mg_run`` (via ``run.py``) script simulates the particles and bins them into a single count (with variance) per detector pixel, stored as a flat ``sc.DataArray`` indexed by ``detector_id`` (see :doc:`scipp_output_format`). Downstream tools reshape this using the detector's pixel grid for 2D plotting.
* **TOF Workflow**: The simulation produces intermediate event-based datasets. Events are preserved with their weight and time-of-flight information. These events are saved into the same Scipp HDF5 (``.h5``) output file. Currently, the data is exclusively saved in **binned (event-mode)** format, preserving the exact TOF of every particle rather than pre-histogramming it — this keeps output files precise but means their size scales with the number of simulated events, not with a fixed number of TOF bins. The binning into final Q-space or detector space is performed flexibly afterwards (e.g. via ``mg_plot --wavelength_slice``), allowing for dynamic slicing and filtering without needing to rerun the BornAgain simulation.

TOF-specific pre-processing (``mg_run``/``mg_fit``, TOF instruments only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two optional, McStas-monitor-driven corrections are applied to the input MCPL particles before the BornAgain simulation, both handled by ``tof_filtering.py``/``preconditioning.py`` and controlled by the *MCPL filtering* and *T0 correction* argument groups of ``mg_run``/``mg_fit`` (see :doc:`cli_reference`):

* **MCPL TOF filtering** (``get_tof_filtering_limits``): if a central ``--wavelength`` is given (and filtering isn't disabled with ``--no_mcpl_filtering``), the TOFLambda-vs-wavelength spectrum from the instrument's ``mcpl_monitor_name`` McStas monitor (found next to the input MCPL file) is sliced at that wavelength, a Gaussian is fitted to the resulting 1D TOF distribution, and only particles within one FWHM of the fitted mean are kept. Explicit TOF limits can be supplied directly with ``--input_tof_limits`` to skip the monitor fit. ``--tof_filtering_figure`` plots the fit and exits without simulating.
* **T0 correction** (``apply_t0_correction``): shifts every particle's TOF by a constant offset, either a fixed value (``--t0_fixed``) or one derived by fitting the same kind of Gaussian to the instrument's ``t0_monitor_name`` (or, in WFM mode, ``wfm_t0_monitor_name``) monitor. This compensates for the time the wavelength-defining chopper system needs before neutrons of the selected wavelength actually leave the source. Disable with ``--no_t0_correction``; visualize with ``--t0_correction_figure``.

2. Coordinate Systems & Transformations
---------------------------------------

The framework is careful to keep two distinct coordinate systems consistent, implemented in ``coordinates.py`` (``CoordinateTransform``):

* **NeXus system** (used for MCPL input, experimental NeXus data, and detector output): X = horizontal (left), Y = vertical (up), Z = longitudinal (beam direction, forward).
* **BornAgain system** (used only internally, around the sample interaction): X = longitudinal (forward), Y = horizontal (left), Z = vertical (up, i.e. the sample-normal direction).

Incoming particles are transformed NeXus → BornAgain before the DWBA calculation (``preconditioning.py``), and outgoing scattered rays are transformed back BornAgain → NeXus before detector-hit projection. Two effects are folded into this transform:

* **Sample orientation** (``--sample_orientation``, one of ``0``/``1``/``2``): a rotation of the sample around the beam axis, letting the same physical sample model represent a vertical scattering geometry (beam hitting from the left or right) as well as the default horizontal geometry, by remapping which NeXus axis becomes "horizontal" and which becomes "vertical" in the BornAgain frame.
* **Sample inclination** (``-a``/``--alpha``, the incident grazing angle) plus the **beam declination angle** (``beam_angle``, either given explicitly or computed automatically from the mean particle velocities, see :doc:`scipp_output_format`): a rotation in the vertical-longitudinal plane so that the incoming beam meets the sample at exactly the intended grazing angle in the BornAgain frame, even if the physical beam in the McStas/NeXus frame is not perfectly horizontal.

**Plotting convention**: independent of the above, ``mg_plot`` always draws :math:`Q_y` on the horizontal plot axis and :math:`Q_z` on the vertical plot axis.

**Limitation**: the detector is currently assumed to be a flat, vertical surface in the NeXus frame (fixed distance and orientation relative to the sample). Curved or tilted detectors are not supported without further coordinate work.

.. _instrument_defaults_schema:

3. Instrument Configuration Reference (``instrument_defaults.py``)
--------------------------------------------------------------------

Every instrument known to ``mcstas_gisans`` (selected via ``-i``/``--instrument`` on ``mg_run``/``mg_plot``/``mg_fit``, or ``--instrument`` on ``mg_beam_centre_correction``) is a plain Python dict entry in the ``instrument_defaults`` dictionary at the top of ``src/mcstas_gisans/instrument_defaults.py``. Adding support for a new instrument means adding a new key there (and, optionally, McStas monitors so the automated TOF filtering/T0 correction described above can work).

Required keys:

* ``nominal_source_sample_distance`` (float, m): distance from the McStas source to the sample position. Used for T0/TOF-related calculations; can be overridden per run with ``--instrument_nominal_source_sample_distance``.
* ``sample_detector_distance`` (float, m): distance from the sample to the detector, along the beam axis. Overridable with ``--instrument_sample_detector_distance``.
* ``tof_instrument`` (bool): whether the instrument is Time-of-Flight (``True``, e.g. SAGA/LOKI/SKADI) or monochromatic (``False``, e.g. D22). This selects which of ``--wavelength``/``--wavelength_selected`` is required, and whether the TOF workflow described above applies. Overridable with ``--instrument_tof_instrument true|false``.

Optional keys:

* ``detector`` (dict): overrides the module-level ``default_detector`` for this instrument. If given, must define all of:

  * ``size`` — ``[size_x, size_y]`` in meters.
  * ``direct_beam_centre_offset`` — ``[offset_x, offset_y]`` in meters; the detector's physical misalignment relative to where the direct beam would nominally hit, as computed by ``mg_beam_centre_correction``.
  * ``pixels`` — ``[pixels_x, pixels_y]`` pixel counts.
  * ``resolution`` — ``[res_x, res_y]`` detector position resolution, FWHM in meters (``0.0`` disables resolution smearing on that axis).

  All of the above are individually overridable per run with ``--instrument_detector_size``, ``--instrument_detector_centre_offset``, ``--instrument_detector_pixels``, ``--instrument_detector_resolution`` (and the ``--nxs_instrument_*`` equivalents on ``mg_plot``/``mg_fit`` for the separately-configurable NeXus/experimental instrument).
* ``mcpl_monitor_name`` (str): name of a McStas *TOFLambda_monitor* placed at the sample position, used for MCPL TOF filtering (see :ref:`sample_position_monitors`).
* ``t0_monitor_name`` (str): name of a McStas *TOFLambda_monitor* placed at the source position, used for T0 correction (see :ref:`source_position_monitors`). Overridable with ``--instrument_t0_monitor_name``.
* ``wfm_t0_monitor_name`` and ``wfm_virtual_source_distance`` (str, float [m]): together enable Wavelength Frame Multiplication (``--wfm``) mode; see :ref:`virtual_source_position_monitors`. Both are required for ``--wfm`` to be accepted for an instrument (checked against ``required_keys_for_wfm``). Overridable with ``--instrument_wfm_t0_monitor_name``/``--instrument_wfm_virtual_source_distance``.
* ``beam_angle`` (float, deg): a fixed default beam declination angle for the instrument; normally left unset so it is calculated automatically per simulation (see above). Overridable with ``--instrument_beam_angle``.

See :doc:`mcstas_preparation` for how to instrument a McStas model with the monitors these keys refer to.

4. Core Tools and Modules
-------------------------

With the modularization of the codebase, specific tasks are handled by dedicated scripts under ``src/mcstas_gisans/``:

* **nexus_reader.py**: Converts experimental NeXus detector pixel data into Q-space using the same ``Instrument`` object parameters as the simulation, for direct comparison. By default the detector data is looked up at one of two known ILL D22 HDF5 layouts (``entry0/D22/Detector 1/data1`` or ``entry0/data1/MultiDetector1_data``); ``--nxs_data_path`` overrides this with an explicit HDF5 path for NeXus files from other facilities or layouts. It also provides ``read_nexus_duration()``, used to weakly cross-check ``--experiment_time`` against the measurement duration reported in the NeXus file itself (see :doc:`main_workflow`).
* **instrument.py**: The ``Instrument`` class — wraps an instrument's ``instrument_defaults`` entry together with the run's alpha/wavelength/orientation to expose detector geometry, pixel positions, and Q-space conversions.
* **coordinates.py** / **preconditioning.py**: Coordinate transformations and MCPL particle preconditioning (T0 correction, coordinate frame conversion, beam-angle calculation), see above.
* **tof_filtering.py**: Derives MCPL TOF acceptance limits from a McStas monitor fit, see above.
* **masking.py**: Builds the boolean Q-space masks used by ``mg_fit`` (``--mask_*`` options) and the ``--mask_view`` visualization.
* **fit.py (mg_fit)**: Runs parameter scans and automated fitting/optimization; see :doc:`main_workflow`.
* **beam_centre_correction.py**: Solves for the detector ``direct_beam_centre_offset`` that centres a direct-beam NeXus measurement at :math:`(Q_y, Q_z) = (0, 0)`.

5. Testing
----------

The codebase includes a suite of regression tests (e.g. ``test_d22_regression.py``) that run full simulations and assert that the computed ``reduced_chi2`` against reference datasets remains stable, alongside unit tests for individual modules. (``test_d22_microgel_regression_local.py`` runs a similar check but depends on data files that are not committed to the repository, so it only runs on machines that have them locally.)

6. BornAgain Compatibility
---------------------------

The core ``mcstas_gisans`` framework is compatible with **BornAgain 21.2, 22.2, 23.0 and 24.1**.
*(Note: individual custom sample models defined in the ``bornagain_samples/`` directory may require minor syntax adjustments depending on the specific BornAgain version being used, due to deprecations in BornAgain's Python API across these versions — see the version-specific ``silica_100nm_air_ba22_ba23``/``silica_100nm_air_ba24`` built-in models for examples. See :doc:`installation_and_usage` for how to select a BornAgain version.)*
