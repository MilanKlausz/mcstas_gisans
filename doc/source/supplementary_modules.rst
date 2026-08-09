=====================
Supplementary Modules
=====================

.. _instrument_defaults_module:

instrument_defaults.py
----------------------

This module defines parameters for different neutron scattering instruments, enabling them to be used for simulations.

The list of available instruments for the :ref:`mg_run <run_simulations_with_mg_run>` script (input: ``--instrument, -i``) is generated from the keys of the *instrument_defaults* dictionary in this module. In order to add a new instrument to the list, copy-paste and edit one of the existing instruments already defined in the *instrument_defaults* dictionary. Some parameters are required for all instruments, while some are optional. Optional parameters enable simulation options like: TOF filtering in the :ref:`Loading neutrons from an MCPL file <loading_neutrons_from_mcpl>` step; :ref:`T0 correction <t0_correction_section>`; :ref:`T0 correction <t0_correction_section>` when using :ref:`Wavelength Frame Multiplication (WFM) mode <wfm_mode_section>`. The full list of required and optional parameters are the following:

Required parameters:
~~~~~~~~~~~~~~~~~~~~

- **nominal_source_sample_distance**: nominal source sample distance in metres unit.
- **sample_detector_distance**: nominal sample to detector surface distance in metres unit.
- **tof_instrument**: boolean value describing whether the instrument is a TOF instrument.

Optional parameters:
~~~~~~~~~~~~~~~~~~~~

- **detector:** description of the detector. Described in detail in :ref:`Detector definition <detector_definition_section>`.
- **beam_declination_angle:** declination angle of the beam at the end of the guide system (before the sample), used for propagation to the vertical detector surface.
- **mcpl_monitor_name**: name of the *TOFLambda_monitor* component at the :ref:`Sample position <sample_position_monitors>`, intended to describe the content of the :ref:`MCPL output <mcpl_output_section>` for TOF filtering in the :ref:`Loading neutrons from an MCPL file <loading_neutrons_from_mcpl>` step.
- **t0_monitor_name**: name of the *TOFLambda_monitor* component at the :ref:`Source position <source_position_monitors>`, intended to describe the neutrons at the source for :ref:`T0 correction <t0_correction_section>`.
- **wfm_t0_monitor_name**: name of the *TOFLambda_monitor* component at the :ref:`Virtual source position <virtual_source_position_monitors>`, intended to describe the neutrons at the virtual source for :ref:`T0 correction <t0_correction_section>` when using :ref:`Wavelength Frame Multiplication (WFM) mode <wfm_mode_section>`.
- **wfm_virtual_source_distance**: real source to virtual source distance in metres unit, used for calculating the distance travelled by the neutron from the virtual source to detector surface when calculating the wavelength of a neutron from its :ref:`T0 corrected <t0_correction_section>` TOF when using :ref:`Wavelength Frame Multiplication (WFM) mode <wfm_mode_section>`.

Example instrument definition:

.. code-block:: python

   instrumentParameters = {
       'saga': {
           'nominal_source_sample_distance': 55, #[m]
           'sample_detector_distance': 10, #[m] along the y axis
           'beam_declination_angle': 0.4, #[deg]
           'tof_instrument': True,
           'mcpl_monitor_name': 'Mcpl_TOF_Lambda',
           't0_monitor_name': 'Source_TOF_Lambda',
           'wfm_t0_monitor_name': 'toflambdawfmc',
           'wfm_virtual_source_distance': 8.2, #real source to virtual source distance for WFM mode
       },
   }

.. _samples_directory_module:

samples directory
-----------------

This directory stores the python scripts defining different sample models for the BornAgain simulations. It can be extended simply by adding a new file here -- possibly exported from the BornAgain GUI --, which will automatically be available for the :ref:`simulation script <run_simulations_with_mg_run>` through the ``--model`` input option. This is not necessary though, as any sample model can be used by providing the path to it. The models in this directory are referred to as the built-in samples, as they are always available for a simulation, and can be used by providing only the name of the sample models, not the path to them. Note that local files in the working directory with the same name have precedence over the built-in models.

It is important to know that the sample definitions of different BornAgain versions are not necessarily compatible with each other. The *get_sample* function of the sample model will be called with the arguments passed in through the ``--sample_arguments`` input option.

.. _fit_monitor_module:

fit_monitor.py
--------------

A utility script to fit a Gaussian function to a 1D TOF spectrum from 2D `TOFLambda_monitor <https://www.mcstas.org/download/components/3.7.9/monitors/TOFLambda_monitor.html>`__ result, as described in :ref:`Loading neutrons from an MCPL file <loading_neutrons_from_mcpl>` and :ref:`T0 correction <t0_correction_section>`. This module is used internally by the :ref:`mg_run <run_simulations_with_mg_run>` script, but it also has a command line interface, so it can be used as a standalone tool by invoking the ``mg_fit_monitor`` command to visualise the fitted Gaussian function on a certain McStas monitor file.

.. _fit_monitor_cli_module:

fit_monitor_cli.py
------------------

Command line interface for the ``fit_monitor`` module.

.. _run_cli_module:

run_cli.py
----------

Command line interface for the *run.py* main simulation module.

.. _plot_cli_module:

plot_cli.py
-----------

Command line interface for the *plot.py* main plotting module.

.. _detector_module:

detector.py
-----------

Module defining the Detector class, which facilitates the determination of detection coordinates.

.. _instrument_module:

instrument.py
-------------

Module defining the Instrument class, which is mainly intended for scattering vector (Q) related calculations.

.. _sample_module:

sample.py
---------

Module defining the Sample class, which encapsulates sample related parameters and methods (e.g., parsing the ``--sample_arguments`` input string).

.. _mcstas_reader_module:

mcstas_reader.py
----------------

A python module for reading McStas monitor output files. As the header line states, it is not an original file: *"Simple support library for reading, analyzing and plotting of McStas results from the Estia instrument simulations."* Note that this is a third party file.

.. _input_output_module:

input_output.py
---------------

A utility module holding functions for reading particles from MCPL files (or .dat file) and writing output Q histogram files (or raw Q list files) in the :ref:`mg_run <run_simulations_with_mg_run>` script, and functions to read the same output files in the :ref:`mg_plot <plot_simulation_results_section>` script.

.. _experiment_time_module:

experiment_time.py
------------------

Contains utilities for scaling simulated data in order to make them comparable with real measurements, as described in :ref:`Scaling to absolute measurement times <scaling_to_absolute_measurement_times_section>`. This module is used internally by the :ref:`mg_plot <plot_simulation_results_section>` script.

.. _read_d22_module:

read_d22.py
-----------

As described in the :ref:`Input files <input_files_section>` section, this is a module with a dedicated (hardcoded) function to retrieve 2D histogram data (with uncertainty and bin edges) from nexus files, that correspond to measurements carried out at the D22 instrument at ILL. This module is used internally by the :ref:`mg_plot <plot_simulation_results_section>` script.

.. _plotting_utils_module:

plotting_utils.py
-----------------

A small utility module that provides plotting functions. This module is used internally by the :ref:`mg_plot <plot_simulation_results_section>` script.

.. _particle_calculations_module:

particle_calculations.py
------------------------

A small utility module that provides conversion factors and functions for neutron-related (or photon-related) calculations. Functions to convert neutron properties from MCPL units to the units used in this framework are also present. This module is used internally by the :ref:`mg_run <run_simulations_with_mg_run>` script in multiple places.
