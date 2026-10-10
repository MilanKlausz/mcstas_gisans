=============
CLI Reference
=============

This section documents the primary command-line interfaces for the ``mcstas_gisans`` package. The examples use the D22 data included in the repository (``data/paper``), as in :doc:`quickstart`.

mg_run
------

.. argparse::
   :module: mcstas_gisans.run_cli
   :func: create_argparser
   :prog: mg_run

**Example Usage:**

.. code-block:: bash

   mg_run data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz \
     --instrument d22 --wavelength_selected 6.0 --intensity_factor 0.2107 \
     --model silica_100nm_air --sample_arguments "radius=51;interferenceRange=5;latticeParameter=114" \
     --sample_size_y 0.06 --sample_size_x 0.08 --allow_sample_miss \
     --alpha 0.24 --sample_orientation 2 --instrument_detector_centre_offset 0.290838 -0.016061 \
     --specular specular_simulation --savename paper_sim

mg_plot
-------

.. argparse::
   :module: mcstas_gisans.plot_cli
   :func: create_argparser
   :prog: mg_plot

**Example Usage:**

.. code-block:: bash

   mg_plot --filename paper_sim.h5 --nxs data/paper/d22_measurement/073174.nxs \
     --experiment_time 10800 --background 1.6 --overlay --q_min 0.072 --q_max 0.102

mg_fit
------

.. argparse::
   :module: mcstas_gisans.fit_cli
   :func: create_fit_parser
   :prog: mg_fit

**Example Usage:**

.. code-block:: bash

   mg_fit data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz \
     --nxs data/paper/d22_measurement/073174.nxs --experiment_time 10800 --fit_background \
     --instrument d22 --wavelength_selected 6.0 --intensity_factor 0.2107 \
     --model silica_100nm_air --sample_arguments "interferenceRange=5" \
     --sample_size_y 0.06 --sample_size_x 0.08 --allow_sample_miss \
     --alpha 0.24 --sample_orientation 2 --instrument_detector_centre_offset 0.290838 -0.016061 \
     --specular specular_simulation --mask_qz_min_cut 0.14 \
     --fit radius 51 45 57 --fit latticeParameter 114 100 130 \
     --max_evals 20 --output_dir fit_example

.. note::
   For more details on the ``--model`` argument and how to construct it, see the :doc:`custom_sample` tutorial. For ``--scan``, ``--nxs`` (including passing multiple files for segmented measurements), masking, joint/dual-sample fitting, and everything else about how ``mg_fit`` works, see :doc:`main_workflow`.

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

   mg_beam_centre_correction data/paper/d22_measurement/073162.nxs --instrument d22 --wavelength 6.0 --sample_orientation 2
