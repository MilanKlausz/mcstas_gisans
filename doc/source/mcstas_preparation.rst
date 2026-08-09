=============================
McStas Instrument Preparation
=============================

.. _mcpl_output_section:

MCPL Output
-----------

The only required modification for an existing McStas instrument model to be usable with this project is the addition of an `MCPL_output <https://mcstas.org/download/components/3.4_current/misc/MCPL_output.html>`__ (or `MCPL_output_noacc <https://mcstas.org/download/components/3.4_current/misc/MCPL_output_noacc.html>`__) component at the sample position:

.. code-block:: c

   COMPONENT mcpl_out = MCPL_output_noacc(filename="test_events")
   AT (0, 0, 0) RELATIVE SamplePos
   ROTATED (0,0,0) RELATIVE SamplePos

Assuming that there is an `Arm <https://www2.mcstas.org/download/components/3.4_current/optics/Arm.html>`__ component at the sample position with the name *SamplePos* aligned with the beam.

For correct propagation of the neutrons, this sample position should be at the centre of the top of the sample. As explained in https://mctools.github.io/mcpl/hooks_mcstas/ :

    *“The coordinates of the stored particle will be relative to the MCPL_output component itself.”*

Hence, by placing the *MCPL_output* component at the centre of the top of the sample, neutron parameters will be expressed with respect to that position, facilitating easy calculation of the incident angles and propagation to the sample surface. Note that neutrons are not propagated to the *MCPL_output* component, so the positions in the MCPL file will reflect the previous component to which the neutrons are propagated, relative to the MCPL_output component's position.

The BornAgain simulation script can handle both horizontal and vertical sample orientations – after a coordinate system rotation explained later. The only thing the user needs to take care of at the McStas simulation stage is that the beam would hit the sample at the sample position.

Note that McStas does not simulate individual physical neutrons. Instead, it simulates statistical representatives of them. This is reflected in the statistical weight assigned to each simulated neutron, which can be greater or less than one. For this reason, these simulated particles are often referred to as neutron rays or neutron events. Therefore, the neutrons stored in an MCPL file can represent either many physical neutrons sharing the same properties, or just a fractional contribution of a single neutron.

Note that in the current codebase, a slightly modified *MCPL_output* component is present with the name *MCPL_output_noacc_russian_roulette.instr*. Its additional – but currently unused – feature is the possibility to normalise the neutron weights to a certain number (e.g, have “real” neutrons with unit-weight) by the Splitting and the Russian Roulette Monte Carlo techniques:

.. code-block:: c

   COMPONENT mcpl_out = MCPL_output_noacc_russian_roulette(
       filename="test_events",
       intendedWeight=intendedWeight
   )
   AT (0, 0, 0) RELATIVE SamplePos
   ROTATED (0,0,0) RELATIVE SamplePos

Monitors
--------

Adding `TOFLambda_monitor <https://www.mcstas.org/download/components/3.7.9/monitors/TOFLambda_monitor.html>`__ components to certain parts of the instrument is not strictly necessary, but they can enable simulation options that are based on the output of these monitors. The names of the McStas monitors don’t tend to change often, so instead of providing these monitors as input for all simulations, the names of the monitors have to be defined in the :ref:`instrument_defaults.py <instrument_defaults_module>` module (for each instrument separately), so that they can be loaded using the :ref:`mcstas_reader.py <mcstas_reader_module>` module by finding the *mccode.sim* file in the same directory as the provided MCPL input file.

.. _sample_position_monitors:

Sample position
~~~~~~~~~~~~~~~

Add a *TOFLambda_monitor* component to the sample position to describe the TOF–wavelength distribution of neutrons in the input MCPL file necessary for TOF filtering in the :ref:`Loading neutrons from an MCPL file <loading_neutrons_from_mcpl>` step.

Provide its name with an **mcpl_monitor_name** field in the :ref:`instrument_defaults.py <instrument_defaults_module>` module for the instrument.

.. _source_position_monitors:

Source position
~~~~~~~~~~~~~~~

Add a *TOFLambda_monitor* component to the source position to describe the TOF–wavelength distribution of neutrons for :ref:`T0 correction <t0_correction_section>`.

Provide its name with a **t0_monitor_name** field in the :ref:`instrument_defaults.py <instrument_defaults_module>` module for the instrument.

.. _virtual_source_position_monitors:

Virtual source position
~~~~~~~~~~~~~~~~~~~~~~~

Add a *TOFLambda_monitor* component to the virtual source position (between the two relevant choppers) to describe the TOF–wavelength distribution of neutrons for :ref:`T0 correction <t0_correction_section>` when using :ref:`Wavelength Frame Multiplication (WFM) mode <wfm_mode_section>`.

Provide its name with a **wfm_t0_monitor_name** field in the :ref:`instrument_defaults.py <instrument_defaults_module>` module for the instrument, and also add a **wfm_virtual_source_distance** field to provide the real source to virtual source distance.
