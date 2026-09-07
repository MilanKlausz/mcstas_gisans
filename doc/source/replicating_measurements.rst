=====================================================
Replicating a Measurement: End-to-End Calibration
=====================================================

This guide walks through the practical, step-by-step procedure for reproducing a
real GISANS measurement with this framework: calibrating the simulated detector
geometry and intensity scale against a **direct beam measurement**, then reusing
that calibration for the actual sample simulation (or fit). It ties together the
individual tools already described in :doc:`mcstas_preparation` and
:doc:`main_workflow` into a single recommended workflow.

The procedure assumes you have (or can obtain) a **direct beam NeXus measurement**
taken with the same instrument configuration (slits, wavelength, detector distance)
as your sample measurement, but without a sample in the beam.

0. McStas simulation matching the measurement
----------------------------------------------

Run (or reuse) a McStas simulation of the instrument configured with the same
settings used during the real measurement — most importantly the **slit
openings**, since these determine the beam divergence and footprint. The
instrument model must end in an ``MCPL_output``/``MCPL_output_noacc`` component
placed at the sample position, so its output can be used as the neutron source
for the BornAgain simulation. See :doc:`mcstas_preparation` for the details and
requirements of this component.

You will typically want two McStas runs (or one run reused for both): one
matching the direct beam measurement's configuration, and one matching the
sample measurement's configuration — they only need to differ if the
instrument settings themselves differ between the two measurements (e.g. some
facilities close down slits for a direct beam measurement to avoid saturating
the detector).

1. Finding the detector offset (``mg_beam_centre_correction``)
----------------------------------------------------------------

Real instruments are rarely perfectly aligned: the direct beam does not
necessarily hit the pixel that the nominal instrument geometry predicts. The
``mg_beam_centre_correction`` utility computes the rigid detector offset needed
to correct for this, from the direct beam **NeXus measurement** alone (no
simulation is required for this step):

.. code-block:: bash

   mg_beam_centre_correction path/to/direct_beam.nxs --instrument d22 --sample_orientation 2

This requires:

- the target instrument (``--instrument``, default ``d22``) to already be
  described in ``instrument_defaults.py``,
- the correct ``--sample_orientation`` for how the measurement was taken (see
  :doc:`main_workflow` for the orientation codes), and
- the beam's declination angle, if the incident beam is not horizontal — pass
  it explicitly with ``--beam_angle`` (in degrees). Unlike ``mg_run`` (see step
  4 below), this tool works from detector pixel counts alone and has no
  velocity information from which to derive this angle automatically, so it
  must be known and supplied if non-zero.

The tool prints the required offset, e.g.:

.. code-block:: text

   Calculated centre_offset [m]:  [X, Y]

Keep this ``[X, Y]`` value — it is reused, unchanged, in steps 3 and 4 below as
``--instrument_detector_centre_offset X Y`` to ``mg_run``. It corrects the
*simulated* detector's assumed geometry to match where the real, physically
misaligned detector actually is, so that Q-space labeling agrees between
simulation and measurement without needing any further correction on the
measurement side.

2. Direct beam simulation and the intensity factor
-----------------------------------------------------

McStas and BornAgain simulations are normalized to an absolute neutron rate
(neutrons/second), which generally does not match the real measured count
rate — real detectors have finite efficiency, and the McStas instrument model
is itself only an approximation of the real instrument. This mismatch is
compensated with a single flat scaling factor, ``--intensity_factor``, applied
to ``mg_run``.

First, run the direct beam simulation itself. Since there is no sample to
scatter off in a direct beam measurement, use ``--sample_size_x 0.0
--sample_size_y 0.0`` together with ``--allow_sample_miss`` so the neutrons
propagate straight through to the detector, and apply the detector offset
found in step 1:

.. code-block:: bash

   mg_run direct_beam.mcpl.gz --instrument d22 --wavelength_selected 6.0 \
     --sample_size_x 0.0 --sample_size_y 0.0 --allow_sample_miss \
     --instrument_detector_centre_offset X Y --sample_orientation 2 \
     --savename direct_beam_sim

This should already produce a beam spot at the correct position (thanks to the
step 1 offset) when overlaid on the measurement, but at the wrong (McStas-normalized)
intensity. Compare the two directly with ``--verbose``, which prints the summed
intensity of both datasets:

.. code-block:: bash

   mg_plot --filename direct_beam_sim.h5 --nxs direct_beam.nxs --overlay --verbose --sample_orientation 2

The intensity factor can then be calculated directly, without any further
simulation, from three numbers:

.. code-block:: text

   intensity_factor = (NXS sum intensity / NXS measurement time [s]) / MCPL sum intensity

- **NXS sum intensity** and **NXS measurement time**: the total measured counts
  and real duration of the direct beam measurement. The sum is printed by the
  ``mg_plot --verbose`` command above (or read directly from the NeXus file);
  the duration is the real measurement time in seconds (from your experiment
  log, or the NeXus file's own ``duration``/``time`` field — ``mg_plot``/``mg_fit``
  will print a warning if the ``--experiment_time`` you give them later
  disagrees with this field by more than 1%).
- **MCPL sum intensity**: the simulated rate (neutrons/second, since McStas
  normalizes source output to one second) at the sample position, i.e. the
  ``sum(weights)`` reported by:

  .. code-block:: bash

     pymcpltool --stats direct_beam.mcpl.gz

  This equals the simulated intensity at the detector, since (absent detector
  efficiency modeling) every neutron reaching the sample position in a direct
  beam simulation also reaches the detector.

**Worked example** (from ``examples/paper``, reproducing the paper's D22 comparison):
a 60 second direct beam measurement recorded 120538 total counts, and
``pymcpltool --stats`` on the corresponding MCPL file reported ``sum(weights):
9533.86``, giving:

.. code-block:: text

   intensity_factor = 120538 / 60 / 9533.86 = 0.2107

See ``examples/paper/README.md`` for the full worked commands.

.. note::
   For **TOF instruments**, ``mg_run`` filters the MCPL file by TOF before
   applying ``--intensity_factor`` (see ``--tof_min``/``--tof_max``), so
   ``pymcpltool``'s whole-file ``sum(weights)`` is only an approximation of the
   quantity actually normalized. The formula above is exact for non-TOF
   instruments (e.g. D22) where no such filtering applies; for TOF instruments,
   treat the result as a starting guess and refine it visually in step 3.

3. Confirming the intensity factor
-------------------------------------

Re-run the direct beam simulation with the calculated ``--intensity_factor``,
and this time upscale the plot comparison to the real measurement time with
``--experiment_time``:

.. code-block:: bash

   mg_run direct_beam.mcpl.gz --instrument d22 --wavelength_selected 6.0 \
     --sample_size_x 0.0 --sample_size_y 0.0 --allow_sample_miss \
     --instrument_detector_centre_offset X Y --intensity_factor 0.2107 \
     --sample_orientation 2 --savename direct_beam_sim

   mg_plot --filename direct_beam_sim.h5 --nxs direct_beam.nxs --overlay \
     --experiment_time 60 --sample_orientation 2

This should now show good agreement in both beam position and total intensity.
If it does not, revisit the offset (step 1) and/or the intensity factor
inputs (step 2) before proceeding.

4. Simulating the actual sample
-----------------------------------

With the detector offset and intensity factor calibrated, run the sample
simulation itself, adding the sample model and its incident angle
(``--alpha``):

.. code-block:: bash

   mg_run sample.mcpl.gz --instrument d22 --wavelength_selected 6.0 \
     --model my_sample_model --alpha 0.4 \
     --instrument_detector_centre_offset X Y --intensity_factor 0.2107 \
     --sample_orientation 2 --savename sample_sim

   mg_plot --filename sample_sim.h5 --nxs sample.nxs --overlay \
     --experiment_time <sample measurement time>

Unlike ``mg_beam_centre_correction``, ``mg_run`` does not need the beam's
declination angle to be known in advance: it estimates it automatically from
the average transverse velocity of the loaded MCPL particles (printed to the
console as "Calculated beam angle"), and uses that unless overridden with
``--instrument_beam_angle``. If you have independently determined a more
reliable beam angle (e.g. from a set of direct beam measurements), pass it
explicitly instead.

5. Unknown sample parameters: fitting instead of a single comparison
-------------------------------------------------------------------------

If the sample's parameters (particle size, lattice spacing, etc.) are not
already known well enough for a single simulation to match the measurement,
use ``mg_fit`` in place of the manual ``mg_run``/``mg_plot`` comparison in
step 4, passing through the same offset/intensity-factor/orientation options:

5a. Build and check a mask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regions such as the specular peak and direct beam typically need to be excluded
from the fit, since they would otherwise dominate the loss and drown out the
diffuse scattering signal of interest. Build a mask with the ``--mask_*``
options and confirm it visually with ``--mask_view`` before committing to a
full fit — this renders the raw vs. masked measurement and exits immediately,
without running any simulation:

.. code-block:: bash

   mg_fit sample.mcpl.gz --nxs sample.nxs --instrument d22 \
     --mask_exclude_q_box -0.05 0.05 -0.02 0.02 --mask_view

5b. Run the fit
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mg_fit sample.mcpl.gz --nxs sample.nxs --instrument d22 \
     --model my_sample_model --alpha 0.4 \
     --instrument_detector_centre_offset X Y --intensity_factor 0.2107 \
     --sample_orientation 2 --experiment_time <sample measurement time> \
     --mask_exclude_q_box -0.05 0.05 -0.02 0.02 \
     --fit radius 50 100 \
     --fit height 20 50

See :doc:`main_workflow` (section 4) for the available optimizers, joint/dual-sample
fitting, and other ``mg_fit`` options.
