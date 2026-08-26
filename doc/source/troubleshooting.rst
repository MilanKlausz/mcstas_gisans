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
- **Initial Guess:** The default center of mass calculation might be thrown off by background noise. Provide an explicit initial guess using ``--initial_guess X Y``.

Fit Does Not Converge (``mg_fit``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the automated fitting stops early or returns an unphysical result:

- **Masking:** Ensure that the specular reflection and the direct beam are correctly masked using ``--mask_rect`` or ``--mask_radius``. The optimizer will often fail if it tries to fit the overwhelmingly bright specular peak instead of the GISANS scattering features.
- **Step Bounds:** Check your ``--fit_parameters`` step sizes. If the bounds are too tight, the optimizer cannot explore the space. (e.g. ``radius=5:1:20`` allows radius to vary between 5 and 20).
- **Poisson Sampling:** Do NOT use ``--poisson_sampling`` during ``mg_fit``. Random noise prevents the objective function from converging smoothly.

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
