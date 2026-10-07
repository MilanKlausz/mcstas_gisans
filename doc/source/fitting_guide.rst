=============
Fitting guide
=============

``mg_fit`` compares simulations with a measured NeXus detector image and adjusts
parameters of the sample model to improve the agreement. Every evaluation is a
complete ``mg_run`` simulation, so a fit costs (number of evaluations) × (one
simulation). This page describes how to prepare, run and judge a fit; the
option reference is in :doc:`cli_reference`.

1. Before fitting
-----------------

A fit can only adjust sample parameters. Everything else must already match the
measurement, otherwise the fit compensates instrument errors with wrong sample
parameters. Establish these first, as described in :doc:`replicating_measurements`:

- **Detector offset** (``--instrument_detector_centre_offset X Y``) from the
  direct beam measurement with ``mg_beam_centre_correction``.
- **Intensity factor** (``--intensity_factor``), the ratio of the measured and
  the simulated beam intensity (``mg_beam_centre_correction --mcpl ... --experiment_time ...``).
- **Incident angle** (``--alpha``) and **sample orientation**
  (``--sample_orientation``). The real incident angle can be measured from the
  specular spot with ``mg_beam_centre_correction --sample_nxs ... --alpha ...``.
- **Measurement time** (``--experiment_time``, required): the simulated rates are
  scaled to expected counts over this time. For several ``--nxs`` files (summed)
  it is their total time.
- **Background**: a flat number of counts per pixel added to the simulation,
  either fixed (``--background``) or fitted (``--fit_background``); see
  section 3.
- **Sample model**: a built-in model or a Python file with a
  ``get_sample(**kwargs)`` function (see :doc:`custom_sample`). Every fitted or
  scanned name must be a keyword argument of ``get_sample``; fixed values are
  given with ``--sample_arguments "name=value;name=value"``.

A single ``mg_run``/``mg_plot`` comparison at the initial parameters is a cheap
way to check all of this before starting a fit.

2. Masking
----------

The loss is calculated only over the unmasked pixels. Regions that the sample
model does not describe must be masked, typically the specular reflection and
the transmitted or direct beam: they are orders of magnitude brighter than the
diffuse scattering and would dominate any loss.

- ``--mask_qy_min_cut``/``--mask_qy_max_cut``/``--mask_qz_min_cut``/``--mask_qz_max_cut``:
  disregard everything below/above the given Q value.
- ``--mask_exclude_q_box qy_min qy_max qz_min qz_max`` (repeatable): exclude a
  rectangle; ``--mask_include_q_box`` (repeatable) adds a rectangle back. Exclusions
  are applied first.

The mask is defined in Q, so it depends on the wavelength, incident angle,
orientation and detector offset. Check it without simulating with ``--mask_view``
(pass the same instrument options as for the fit): it shows the measurement with
and without the mask and exits. ``mg_fit`` prints the number of unmasked pixels.

``--simulate_mask_angle_range`` restricts the simulated outgoing angles to those
that can reach unmasked pixels, which makes each evaluation faster when a large
part of the detector is masked. By default (``--simulate_mask_angle_range_factor
auto``) every neutron hitting the sample gets its own window of outgoing
directions: the range enclosing the unmasked pixels, as seen from the sample
centre, shifted by the neutron's incident horizontal direction, its hit point on
the sample and its gravity drop, and widened by 2 sigma of the detector
resolution on every side. The windows of all neutrons have the size of that
range (plus the resolution margins), smaller than a common range containing all
of them (which the beam divergence, the beam spot and the wavelength spread
widen), so the same number of outgoing directions is denser; only rays smeared
into the region from beyond 2 sigma are lost (about 0.01% for the D22
examples). The size of the windows and their union
are printed. With a number instead of ``auto`` the range enclosing the unmasked
pixels is scaled by it about its centre and simulated for every neutron; a
warning is printed if it does not contain the union of the windows.

3. Background
-------------

The measured counts contain a background that the sample model does not
describe: room and instrument background, detector noise, incoherent scattering
(e.g. from a liquid subphase). In a GISANS measurement much of the unmasked
region is at, or close to, the background level, so the background has a large
effect on the loss. If it is set too low, the fit compensates with a pattern
that is too intense. In the D22 microgel measurement used in the example below,
the flat floor is 6.1 counts per pixel (the same far from the pattern, at high
:math:`Q_z` and below the sample horizon, with a Poisson-like spread), while an
earlier fit used 1.0: re-scored with 6.1, its loss dropped from 6.14 to 4.68
(reduced :math:`\chi^2` from 8.67 to 3.55), and its pattern turned out to be
about 1.7 times too intense.

The background is an expected value, like the simulated counts: it is not
Poisson-sampled, because the counting noise of the measurement is already part
of the loss. Two ways to set it (they cannot be combined):

- ``--background B``: a fixed number of counts per detector pixel over
  ``--experiment_time`` (default 0). It can be estimated as the mean measured
  counts in a region without scattering from the sample, for example below the
  sample horizon or far from the pattern (away from the direct and transmitted
  beam).
- ``--fit_background``: the background is fitted for every evaluation. After
  each simulation, the flat level that minimises ``--loss_function`` over the
  unmasked pixels is found by a one-dimensional minimisation between 0 and the
  mean measured counts. This needs no extra simulation (milliseconds), and it
  leads to the same optimum as fitting the background as one more parameter.
  The value is printed with every evaluation, written to the ``background``
  column of ``fit_summary.csv``/``scan_summary.csv``, and reported for the best
  evaluation. In joint fits each measurement gets its own background.

The fitted value is conditional on the simulated pattern: where a pattern is
too intense, a lower background reduces the excess, so far from the optimum
the fitted background also absorbs errors of the model. In the microgel
example, the first evaluations of a fit gave 1.1 to 4.1 counts per pixel (the
best of them was the 1.7 times too intense pattern above), and it approaches
the floor of the data only as the pattern improves. Judge the value at the best
fit, and compare it with the measured floor. Two more points:

- Use it with the default ``poisson_deviance``: minimising ``reduced_chi2``
  (whose variance, the expected counts, is in the denominator) overestimates
  the background by about 0.5 counts per pixel.
- The background is flat. A background that varies across the detector (e.g.
  from the subphase, or from the sample environment) is not described by it;
  mask such regions or keep them in mind when judging the fit.

4. Scan first
-------------

A scan evaluates listed parameter values without optimisation:

.. code-block:: bash

   mg_fit ... --scan radius 48 50 52 54 --scan latticeParameter 110 114 118

Repeating ``--scan`` for several parameters evaluates all combinations. The loss
values in ``scan_summary.csv`` (in ``--output_dir``) show how sensitive the loss
is to each parameter and where a fit should start; ``--png`` saves a comparison
plot for every point.

5. Fitting
----------

Each ``--fit`` gives a parameter name followed by the initial value
(``--fit radius 51``), the bounds (``--fit radius 40 60``, starting from the
midpoint) or both (``--fit radius 51 40 60``). The initial value must be within
the bounds.

Optimizer (``--optimizer``):

- ``nelder-mead`` (default): local, derivative-free simplex search. Good for a
  few parameters and a reasonable starting point.
- ``powell``: local, searches along one direction at a time; can be more robust
  when parameters are weakly coupled.
- ``differential-evolution``: global search within the bounds (all parameters
  need bounds). It needs many evaluations: every generation is ``--popsize``
  × (number of parameters) simulations.

Nelder-Mead and Powell work in scaled parameters (each parameter divided by its
bound range, or by its initial value if unbounded), so parameters of very
different magnitude are handled alike. They stop when the parameters change by
less than ``--xatol`` (relative to the scale, default 1%) and the loss by less than
``--fatol`` (absolute, default 0.05), or after ``--max_evals`` evaluations
(default 10, respected by all optimizers). Differential Evolution has no
parameter criterion: it stops when the spread (standard deviation) of the losses
of its population is below ``--fatol``, or when ``--max_evals`` is used up, so
``--xatol`` has no effect there. ``--fit_integer name`` keeps a parameter
integer (e.g. a number of layers).

Joint fits of two measurements (``--nxs2``) with shared (``--fit_common``) and
separate (``--fit``/``--fit2``) parameters are described in :doc:`main_workflow`.

6. Loss function and Monte Carlo noise
--------------------------------------

The default loss, ``poisson_deviance``, is the per-pixel deviance of a Poisson
likelihood, with the Monte Carlo uncertainty of the simulation included. It is
about 1 per pixel for a model that describes the data within counting
statistics, and it gives unbiased parameters also at a few counts per pixel.
``reduced_chi2`` and ``log_residual`` are reported as well
(see :ref:`the loss definitions <loss-functions>`); compare loss values only
within one loss function. A value well above 1 means the model (or the mask, or
the instrument setup) does not describe the data; the fitted parameters are
then the best compromise, not necessarily the true values.

The simulation is deterministic within a fit: expected counts are compared (no
Poisson sampling), and every evaluation uses the same random seed (``--seed``,
printed and stored), so the loss does not jump randomly between evaluations.
The finite number of simulated neutrons still limits the precision: the fitted
parameters are those that best describe *this* Monte Carlo sample. To judge
this, repeat the evaluation at the best parameters (``--max_evals 1``) with a
few different ``--seed`` values, which sample the outgoing directions and the
detector resolution differently, and, if available, with an independent McStas
run of the same instrument (the seed does not change the MCPL particles). If the
loss changes as much as the differences between the best evaluations, more
statistics (more McStas neutrons, or more outgoing directions: ``-n``, or a
higher ``--sampling`` preset, see :ref:`sampling_presets`) are needed before
the parameters can be told apart.
The loss includes the Monte Carlo variance of the simulation, so a noisier
simulation scores a *lower* loss for the same pattern (for the D22 paper data
the Poisson deviance rose from 16 to 44 when the rays per pixel went from about
250 to 7000). Compare losses only between runs with the same MCPL file and the
same outgoing directions; with ``--sampling``, use the printed
``--outgoing_directions_horizontal``/``--outgoing_directions_vertical`` numbers
to repeat a run with exactly the same grid.
``mg_fit`` warns if the Monte Carlo variance exceeds the counting variance in
more than 5% of the pixels.

7. Outputs
----------

In ``--output_dir`` (default ``scan_results``):

- ``fit_summary.csv`` / ``scan_summary.csv``: every evaluation with its
  parameters, all loss values and, with ``--fit_background``, the fitted
  background, sorted by the selected loss.
- With ``--png``: a comparison plot of the measurement and the simulation for
  every evaluation (``fit_eval_<n>_<parameters>.png`` in fits,
  ``sim_<parameters>.png`` in scans). ``--gif`` turns the fit plots into an
  animation. With ``--split_view`` the two 2D maps become one, with the
  measurement for Qy < 0 and the simulation for Qy > 0 on a common colour
  scale.

The summary with the best parameters, the optimizer's status and the run time
is also printed. "Maximum number of function evaluations has been exceeded"
means the budget ran out before the tolerances were met; continue from the best
parameters with a new fit if the loss was still decreasing.

8. Run time
-----------

The cost of one evaluation grows with the number of MCPL particles and with the
square of ``-n`` (the outgoing directions per axis). ``--sampling`` chooses the
directions for a target noise per pixel (``quick``, the default, ``standard``,
``long``) and takes the (effective) number of neutrons into account, so a larger
MCPL file automatically gets fewer directions per neutron. ``-p`` sets the number of
parallel processes; with many processes use ``--bornagain_number_of_threads 1``.
``--simulate_mask_angle_range`` saves time when much of the detector is masked.
A scan with a few points is usually worth more than a long fit from a poor
starting point.

9. Worked example (D22 paper data)
----------------------------------

The silica nanoparticle measurement of the paper (``073174.nxs``, 3 hours at
0.24°, vertical sample with its normal pointing left) with the McStas beam of
the paper included in the repository. The detector offset and the intensity
factor come from the direct beam measurement ``073162.nxs``
(see :doc:`replicating_measurements`); the region below
:math:`Q_z = 0.14\,\mathrm{nm}^{-1}` (transmitted beam, specular reflection and
Yoneda region) is masked:

.. code-block:: bash

   mg_fit data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz \
     --nxs data/paper/d22_measurement/073174.nxs --experiment_time 10800 --fit_background \
     --instrument d22 --wavelength_selected 6.0 --intensity_factor 0.2107 \
     --alpha 0.24 --sample_orientation 2 --instrument_detector_centre_offset 0.290855 -0.016063 \
     --model silica_100nm_air --sample_arguments "interferenceRange=5" \
     --sample_size_y 0.10 --sample_size_x 0.10 --allow_sample_miss \
     --specular specular_simulation --use_avg_materials \
     --mask_qz_min_cut 0.14 \
     --fit radius 51 45 57 --fit latticeParameter 114 100 130 \
     --max_evals 20 --seed 1 --output_dir fit_example

``mg_fit`` prints the number of unmasked pixels, the chosen outgoing directions
(here the ``quick`` preset) and, for every evaluation, the parameters, the loss
and the fitted background; ``fit_summary.csv`` lists them sorted by the loss.
If the best evaluations differ by less than the Monte Carlo noise of the loss
(repeat the best one with a few ``--seed`` values, section 6), more simulated
statistics (``--sampling standard``) are needed to pin the parameters down more
precisely.
