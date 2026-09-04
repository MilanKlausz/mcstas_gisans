=============
CLI Reference
=============

This section documents the primary command-line interfaces for the `mcstas_gisans` package.

mg_run
------

.. argparse::
   :module: mcstas_gisans.run_cli
   :func: create_argparser
   :prog: mg_run

**Example Usage:**

.. code-block:: bash

   mg_run mcstas_output.mcpl.gz -i d22 --wavelength_selected 6.0 --model silica_100nm_D2O -n 100 --specular include_specular

mg_plot
-------

.. argparse::
   :module: mcstas_gisans.plot_cli
   :func: create_argparser
   :prog: mg_plot

**Example Usage:**

.. code-block:: bash

   mg_plot -f test_q.h5 -t 3600 --background 0.001

mg_fit
------

.. argparse::
   :module: mcstas_gisans.fit_cli
   :func: create_fit_parser
   :prog: mg_fit

**Example Usage:**

.. code-block:: bash

   mg_fit mcstas_output.mcpl.gz --nxs d22_experiment.nxs --instrument d22 --model my_custom_sample --wavelength_selected 6.0 --fit radius 5 1 20 --fit height 10 5 50

.. note::
   For more details on the ``--model`` argument and how to construct it, see the :doc:`custom_sample` tutorial.

mg_fit_monitor
--------------

.. argparse::
   :module: mcstas_gisans.fit_monitor_cli
   :func: create_argparser
   :prog: mg_fit_monitor

mg_beam_centre_correction
-------------------------

.. argparse::
   :module: mcstas_gisans.beam_centre_correction
   :func: create_argparser
   :prog: mg_beam_centre_correction

**Example Usage:**

.. code-block:: bash

   mg_beam_centre_correction d22_direct_beam.nxs --instrument d22 --wavelength 6.0
