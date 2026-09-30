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
- **Background** (``--background``): a flat number of counts per pixel added to
  the simulation, e.g. estimated from a region without scattering.
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
part of the detector is masked. The range is widened by the beam divergence,
the sample size, the detector resolution and gravity, so no intensity inside the
unmasked region is lost.

3. Scan first
-------------

A scan evaluates listed parameter values without optimisation:

.. code-block:: bash

   mg_fit ... --scan radius 48 50 52 54 --scan latticeParameter 110 114 118

Repeating ``--scan`` for several parameters evaluates all combinations. The loss
values in ``scan_summary.csv`` (in ``--output_dir``) show how sensitive the loss
is to each parameter and where a fit should start; ``--png`` saves a comparison
plot for every point.

4. Fitting
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

5. Loss function and Monte Carlo noise
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
statistics (more McStas neutrons, or more outgoing directions ``-n``) are needed
before the parameters can be told apart.
``mg_fit`` warns if the Monte Carlo variance exceeds the counting variance in
more than 5% of the pixels.

6. Outputs
----------

In ``--output_dir`` (default ``scan_results``):

- ``fit_summary.csv`` / ``scan_summary.csv``: every evaluation with its
  parameters and all loss values, sorted by the selected loss.
- With ``--png``: a comparison plot of the measurement and the simulation for
  every evaluation (``fit_eval_<n>_<parameters>.png`` in fits,
  ``sim_<parameters>.png`` in scans). ``--gif`` turns the fit plots into an
  animation.

The summary with the best parameters, the optimizer's status and the run time
is also printed. "Maximum number of function evaluations has been exceeded"
means the budget ran out before the tolerances were met; continue from the best
parameters with a new fit if the loss was still decreasing.

7. Run time
-----------

The cost of one evaluation grows with the number of MCPL particles and with the
square of ``-n`` (the outgoing directions per axis). ``-p`` sets the number of
parallel processes; with many processes use ``--bornagain_number_of_threads 1``.
``--simulate_mask_angle_range`` saves time when much of the detector is masked.
A scan with a few points is usually worth more than a long fit from a poor
starting point.

8. Worked example (D22 paper data)
----------------------------------

The silica nanoparticle measurement of the paper (``073174.nxs``, 3 hours at
0.24°, vertical sample with its normal pointing left) with the low-statistics
McStas beam included in the repository. The detector offset and the intensity
factor come from the direct beam measurement ``073162.nxs``
(see :doc:`replicating_measurements`); the region below
:math:`Q_z = 0.14\,\mathrm{nm}^{-1}` (transmitted beam, specular reflection and
Yoneda region) is masked:

.. code-block:: bash

   mg_fit data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz \
     --nxs data/paper/d22_measurement/073174.nxs --experiment_time 10800 --background 1.6 \
     --instrument d22 --wavelength_selected 6.0 --intensity_factor 0.2084 \
     --alpha 0.24 --sample_orientation 2 --instrument_detector_centre_offset 0.290855 -0.016063 \
     --model silica_100nm_air --sample_arguments "interferenceRange=5" \
     --sample_size_y 0.10 --sample_size_x 0.10 --allow_sample_miss \
     --specular include_specular --use_avg_materials \
     --mask_qz_min_cut 0.14 \
     --fit radius 51 45 57 --fit latticeParameter 114 100 130 \
     --max_evals 20 --seed 1 --output_dir fit_example

With 18176 unmasked pixels, 20 evaluations took 5 minutes on a laptop and
lowered the deviance from 1.650 (radius 51 nm, lattice parameter 114 nm) to
1.637 (52.6 nm, 117.5 nm). The small differences between the best evaluations
show that more simulated statistics would be needed to pin the parameters down
more precisely (section 5).
