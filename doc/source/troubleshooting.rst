===============
Troubleshooting
===============

This page covers common issues and resolutions when running ``mcstas_gisans``.

Common Issues
-------------

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~
If you see ``ModuleNotFoundError: No module named 'bornagain'`` or ``'scipp'``, ensure that you have correctly activated the conda environment.

.. code-block:: bash

   conda activate mcstas_gisans

NeXus Alignment Fails (``mg_beam_centre_correction``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the optimization fails to find a center for the NeXus direct beam:

- **Saturated Detectors:** The NeXus file might have saturated pixels at the direct beam center. Check the NeXus data visually.
- **Initial Guess:** The default center of mass calculation might be thrown off by background noise. ``mg_beam_centre_correction`` does not currently expose an ``--initial_guess`` CLI option; if the optimizer converges to the wrong feature, inspect and, if needed, pre-mask the NeXus data before running the correction.

Fit Does Not Converge (``mg_fit``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the automated fitting stops early or returns an unphysical result:

- **Masking:** Ensure that the specular reflection and the direct beam are correctly masked using ``--mask_exclude_q_box`` (or ``--mask_qy_min_cut``/``--mask_qy_max_cut``/``--mask_qz_min_cut``/``--mask_qz_max_cut``). The optimizer will often fail if it tries to fit the overwhelmingly bright specular peak instead of the GISANS scattering features.
- **Fit Bounds:** Check your ``--fit`` bounds. If they are too tight, the optimizer cannot explore the space (e.g. ``--fit radius 5 20`` allows radius to vary between 5 and 20).
- **Poisson Sampling:** Do NOT use ``--poisson_sampling`` during ``mg_fit``. Random noise prevents the objective function from converging smoothly.

Multiple ``--nxs`` Files Don't Match (``mg_fit``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``mg_fit --nxs`` accepts more than one NeXus file (e.g. for a measurement recorded across several segments); their counts are summed before fitting. If you see ``Incompatible NeXus data shapes across files``, the given files were read out into detector histograms of different shapes (for example, because they were recorded with a different ``--instrument``/``--nxs_instrument_name`` or detector configuration than the others) and cannot be summed. Also remember to set ``--experiment_time`` to the *cumulative* time across all given files, not the time of a single one.

Could Not Find Detector Data in NeXus File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, ``--nxs`` files are expected to hold their detector data at one of two known ILL D22 HDF5 paths. If you see ``Could not find detector data in NeXus file ... at any of the default paths``, your file uses a different facility/instrument layout: inspect it (e.g. with ``h5py`` or ``h5dump -n``) to find the correct dataset path, then pass it explicitly with ``--nxs_data_path``, e.g. ``--nxs_data_path "entry0/instrument/detector/data"``.

Experiment Time / Measurement Duration Mismatch Warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If your NeXus file(s) report their own measurement duration (checked via the standard ``NXentry/duration`` field, falling back to the ILL-specific ``NXentry/time`` alias), it is compared against ``--experiment_time`` on ``mg_plot``/``mg_fit`` (the tools that read both ``--nxs`` and ``--experiment_time``). A message like ``WARNING: ... report a total measurement duration of 1800.0s ... but --experiment_time was set to 3600.0s`` does **not** stop execution — it's a hint that ``--experiment_time`` may not match the actual measurement. This is common when reusing a placeholder value across a whole segmented series (e.g. always passing the nominal per-segment duration instead of the true cumulative sum, which can differ if a segment was cut short). If the value is intentional (e.g. deliberately simulating a different exposure time than the real measurement), the warning can be ignored.

Wavelength Parse Errors
~~~~~~~~~~~~~~~~~~~~~~~

- If you see ``argparse.ArgumentError: argument --wavelength_selected: not allowed with argument --wavelength``:
  - Use ``--wavelength_selected`` for monochromatic/continuous sources (e.g., D22).
  - Use ``--wavelength`` for TOF sources (e.g., SAGA, LOKI).

No Particles Hitting the Sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you get a warning that 0 particles hit the sample surface:

- Verify that your McStas ``MCPL_output`` component is positioned exactly at the sample center (origin). In your ``.instr`` file, look for a snippet like this:

  .. code-block:: c
      
      COMPONENT MCPL_out = MCPL_output(filename="output_dir/test_events")
      AT (0, 0, 0) RELATIVE sample_position

- If the beam is slightly misaligned vertically in the physical instrument, try using the ``--nexus_y_shift`` flag in ``mg_run``. Even though the input is an MCPL file, this flag will mathematically shift the particles along the NeXus Y-axis prior to BornAgain simulation, ensuring they actually intersect the sample plane.
