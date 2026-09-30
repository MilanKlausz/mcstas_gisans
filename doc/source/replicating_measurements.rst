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
``mg_beam_centre_correction`` utility computes the detector offset from the
direct beam **NeXus measurement** alone (no simulation is required for this
step):

.. code-block:: bash

   mg_beam_centre_correction path/to/direct_beam.nxs --instrument d22 --wavelength 6.0

The offset is the position of the detector centre relative to the undeflected
beam axis through the sample. It is computed in real space: the tool takes the
intensity centroid of the measured direct beam and places it where the
unscattered beam must land — the beam axis (tilted by ``--beam_angle``)
lowered by the gravity drop at ``--wavelength``. It therefore requires:

- the target instrument (``--instrument``, default ``d22``) to already be
  described in ``instrument_defaults.py``,
- the wavelength of the direct-beam measurement (``--wavelength``, default
  6.0 Å; it enters only through the gravity drop), and
- the beam angle, if the incident beam is not along the nominal axis
  (``--beam_angle`` in degrees, positive towards the sample surface normal, in
  the plane of incidence of the ``--sample_orientation`` used; opposite sign to
  the former ``--beam_declination``). Use the same value in ``mg_run``.

The offset does not depend on the sample orientation, so one direct-beam
measurement serves horizontal and vertical samples alike. The tool prints it,
e.g.:

.. code-block:: text

   Calculated centre_offset [m]:  [X, Y]

(for the D22 paper direct beam ``073162.nxs`` at 6 Å: ``[0.290855, -0.016063]``;
the 0.29 m matches the recorded 300 mm sideways detector translation).

Keep this ``[X, Y]`` value — it is reused, unchanged, in steps 3 and 4 below as
``--instrument_detector_centre_offset X Y``. With it, the measured direct beam
is exactly at Q = 0 and a simulated direct beam lands on the measured one (see
:ref:`q_convention` in :doc:`technical_details`). The centroid is computed
within ``--beam_radius`` (default 100 mm, which holds the whole D22 big beam)
of the beam, so that scattered background elsewhere on the detector does not
bias it.

**Cross-checks in the same command (recommended).** Given the MCPL file of the
direct-beam McStas simulation (``--mcpl``), the tool ray-traces the McStas beam
with the found offset — exactly like a ``mg_run`` direct-beam simulation, but
within seconds — and reports the centroid residual between the simulated and
the measured beam, the spot widths, the MCPL beam angle and mean wavelength
(against ``--beam_angle``/``--wavelength``), and, with ``--experiment_time``,
the intensity factor of step 2. ``--figure png|pdf|show`` draws the two beams
and their profiles. A residual above half a pixel is reported with the offset
that would make the simulation match; it points at the McStas beam (direction,
position, slits), the beam angle or the wavelength — the offset printed first
always describes the measurement. Given a sample measurement at the same
wavelength (``--sample_nxs``), the real incident angle is measured from the
distance between the specular spot and the direct beam (:math:`2\alpha`; the
gravity drop cancels) and compared with ``--alpha``. The specular position is
the intensity centroid in a window of ±20 mm along the sample normal and
±80 mm across it, re-centred on its own centroid: the window has to hold the
whole spot, because the specular of a wide beam is flat-topped and a window
around its brightest pixels is biased (by up to 1 pixel for the D22 big beam).
With ``--figure``, a second figure (``<savename>_incident_angle``) shows the
direct beam and the sample image with the found centres and both windows:

.. code-block:: bash

   mg_beam_centre_correction direct_beam.nxs --instrument d22 --wavelength 6.0 --sample_orientation 2 \
     --mcpl direct_beam.mcpl.gz --experiment_time 60 --figure png \
     --sample_nxs sample.nxs --alpha 0.24

For the D22 paper data this gives a residual of 0.01 pixel, an intensity
factor of 0.2107, and a measured incident angle of 0.235° for the nominal 0.24°.

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
   applying ``--intensity_factor`` (``--wavelength`` / ``--input_tof_limits``), so
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

Like ``mg_beam_centre_correction``, ``mg_run`` defaults the beam angle to
``0.0`` (or whatever is configured for the instrument in
``instrument_defaults.py``) unless overridden with ``--instrument_beam_angle``
— use the same value here as in step 1, for consistency with the detector
offset calculated there. The beam angle must describe the mean direction of the
simulated (MCPL) beam at the sample. ``mg_run`` prints an independent estimate
of it from the MCPL particle velocities, purely as a sanity check: it is never
used as the actual value, but a printed ``WARNING`` (disagreement with the value
actually used by more than 0.05°) is worth investigating — it usually means a
forgotten/wrong ``--instrument_beam_angle``, or an MCPL file that does not
match the intended measurement.

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

   mg_fit --nxs sample.nxs --instrument d22 --wavelength_selected 6.0 \
     --alpha 0.4 --sample_orientation 2 --instrument_detector_centre_offset X Y \
     --mask_exclude_q_box -0.05 0.05 -0.02 0.02 --mask_view

The mask is defined in Q, so the preview needs the same wavelength, incident
angle, orientation and detector offset as the fit itself.

5b. Run the fit
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mg_fit sample.mcpl.gz --nxs sample.nxs --instrument d22 --wavelength_selected 6.0 \
     --model my_sample_model --alpha 0.4 \
     --instrument_detector_centre_offset X Y --intensity_factor 0.2107 \
     --sample_orientation 2 --experiment_time <sample measurement time> \
     --mask_exclude_q_box -0.05 0.05 -0.02 0.02 \
     --fit radius 51 40 60 \
     --fit height 30 20 50

See :doc:`fitting_guide` for choosing masks, optimizers and loss functions, judging
the result, and a worked example on the D22 data, and :doc:`main_workflow`
(section 4) for joint/dual-sample fitting.
