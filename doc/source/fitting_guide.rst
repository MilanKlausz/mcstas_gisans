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
into the region from beyond 2 sigma are lost (about 0.01% in the D22 example
of section 9). The size of the windows and their union
are printed. With a number instead of ``auto`` the range enclosing the unmasked
pixels is scaled by it about its centre and simulated for every neutron; a
warning is printed if it does not contain the union of the windows.

A window (or the scaled range) is a rectangle of directions, but the fit region
often is not: the corners of a T-shaped region, or the hole of an excluded
specular box, are aimed at masked pixels only. These directions are skipped:
before each neutron's BornAgain simulation its rays are traced to the detector
(without the resolution smearing), and the directions whose hit point, widened
by 2 sigma of the detector resolution along each detector axis, touches no
unmasked pixel are masked in the BornAgain detector, which does not compute
them. The unmasked pixels get the same counts as without skipping (only rays
smeared by more than 2 sigma from a skipped direction are missing, as for the
windows), and the masked pixels of the simulated image stay mostly empty. The
skipped fraction is printed after the first evaluation; the time saved depends on
the case: for the band with an excluded specular box of section 9 (a small grid)
30% of the directions are skipped and an evaluation is 1.2 times faster; for a
T-shaped region of a liquid-surface fit (46 x 36 directions) 56% are skipped and
an evaluation is 3.5 times faster (the losses agree to 0.03%).
``--keep_masked_directions`` simulates all directions of the windows, as before.

3. Background
-------------

The measured counts contain a background that the sample model does not
describe: room and instrument background, detector noise, incoherent scattering
(e.g. from a liquid subphase). In a GISANS measurement much of the unmasked
region is at, or close to, the background level, so the background has a large
effect on the loss. If it is set too low, the fit compensates with a pattern
that is too intense. For example, in a D22 measurement of pNIPAM microgels at
the air/D2O interface (not included in the repository), the flat floor was 6.1
counts per pixel (the same far from the pattern, at high :math:`Q_z` and below
the sample horizon, with a Poisson-like spread). A fit with a fixed background
of 1.0 counts per pixel compensated with a pattern about 1.7 times too intense:
re-scored with 6.1, its loss dropped from 6.14 to 4.68 (reduced
:math:`\chi^2` from 8.67 to 3.55).

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
  unmasked pixels (with ``--fit_objective 1d`` over the Qy profile of the
  :math:`Q_z` band, section 6.3) is found by a one-dimensional minimisation between 0 and the
  mean measured counts. This needs no extra simulation (milliseconds), and it
  leads to the same optimum as fitting the background as one more parameter.
  The value is printed with every evaluation, written to the ``background``
  column of ``fit_summary.csv``/``scan_summary.csv``, and reported for the best
  evaluation. In joint fits each measurement gets its own background.

The fitted value is conditional on the simulated pattern: where a pattern is
too intense, a lower background reduces the excess, so far from the optimum
the fitted background also absorbs errors of the model. In a fit of the
microgel measurement above, the first evaluations gave 1.1 to 4.1 counts per pixel (the
best of them was the 1.7 times too intense pattern above), and it approaches
the floor of the data only as the pattern improves. Judge the value at the best
fit, and compare it with the measured floor. Two more points:

- Use it with the default ``poisson_deviance``: minimising ``reduced_chi2``
  (whose variance, the expected counts, is in the denominator) overestimates
  the background by about 0.5 counts per pixel (see :ref:`loss-functions`).
- The background is flat. A background that varies across the detector (e.g.
  from the subphase, or from the sample environment) is not described by it;
  mask such regions or keep them in mind when judging the fit.

4. Scan first
-------------

A scan evaluates listed parameter values without optimisation:

.. code-block:: bash

   mg_fit ... --scan radius 48 50 52 54 --scan latticeParameter 110 114 118

(``radius`` and ``latticeParameter`` are parameters of the built-in
``silica_100nm_air`` model, used in the example of section 9.)

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

5.1 Global search, then local refinement (``--polish``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a fit whose loss may have several minima, the recommended procedure is a
global search followed by a local refinement of its best point:

1. ``--optimizer differential-evolution`` with bounds wide enough to contain
   the optimum. Differential Evolution finds the right minimum, but it stops
   when the losses of its population agree within ``--fatol``. Its best point
   can then still be a few standard errors from the bottom of that minimum,
   especially in narrow valleys of correlated parameters: one standard error
   corresponds to a change of the total deviance by 1, i.e. of the loss (per
   pixel) by 1/(number of pixels), far below a useful ``--fatol``.
2. ``--polish nelder-mead`` (or ``powell``) refines the best point of the
   Differential Evolution in the same fit, with the integer parameters fixed at
   their values there. It stops when the parameters change by less than
   ``--polish_xatol`` (relative to the bound ranges, default 0.002) and the loss
   by less than ``--polish_fatol`` (default 0.002), or after
   ``--polish_max_evals`` evaluations (default 100). Its evaluations have
   ``polish`` in the ``phase`` column of ``fit_summary.csv``.
3. The uncertainties of the fitted parameters follow from the curvature of the
   loss at the refined minimum. That point should not lie at a bound: a result
   at a bound is not a minimum; widen the bound and continue.

The local optimisers are derivative-free, because the loss comes from a Monte
Carlo simulation: with a fixed ``--seed`` it is a smooth function of the
parameters on the scale of physical parameter changes, but not on the scale of
the tiny steps of finite-difference gradients (scipy's own polish of a
Differential Evolution uses L-BFGS-B with such gradients and is therefore
switched off). A polish costs some tens of evaluations, typically 15–30% of the
Differential Evolution.

5.2 Continuing a fit, or starting it again (``--overwrite``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``--output_dir`` holds one fit. Besides ``fit_summary.csv``, ``mg_fit``
writes ``fit_state.json`` (the settings, the status and one entry per run of the
fit) and ``fit_evaluations.jsonl`` (every evaluation in the order made), after
every evaluation. When ``mg_fit`` starts, it looks at ``--output_dir`` and
prints what it found and decided:

- **no fit there**: a new fit starts;
- **an unfinished fit with the same settings** (stopped by a job time limit,
  ``scancel``, a node failure or an error): it is **continued** automatically.
  The optimiser is started again exactly as before; the evaluations it requests
  in the same order with the same parameters are answered from
  ``fit_evaluations.jsonl`` instead of being simulated, and then the fit goes on
  with new evaluations. The scipy optimisers are deterministic for given losses
  (Differential Evolution with its seeded generator), so the result is the same
  as that of an uninterrupted fit. The simulation itself need not be
  deterministic for this: the stored losses are used. At most the evaluation
  that was running when the fit stopped is lost;
- **a finished fit**, **an unfinished fit with other settings**, or a
  ``fit_summary.csv`` without ``fit_state.json`` (an older ``mg_fit``): ``mg_fit``
  stops with a message listing the reason (and the differing settings), and
  nothing is changed.

Between the runs of one fit, only the budgets (``--max_evals``,
``--polish_max_evals``; a larger budget lets a fit that ran out of evaluations
go on), the parallel options and the plot options may change. File options
(``--nxs``, the MCPL file, ``--model``) are compared by file name, so a fit can
be continued on another machine. Without ``--seed``, the seed of the fit is
used. If the optimiser nevertheless requests another point than the stored one
(e.g. after an update of scipy), ``mg_fit`` prints a warning, discards the
remaining stored evaluations and goes on with simulations from that point.

``--overwrite`` starts the fit from scratch: the three files are removed (the
comparison plots are overwritten as they are made). ``--de_continue`` starts a
new Differential Evolution fit, in another ``--output_dir``, from the population
of an existing fit (e.g. one of an older ``mg_fit``, whose evaluations are
replayed with the update rules of scipy).

5.3 Batch jobs that continue themselves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because a fit continues automatically in its ``--output_dir``, a fit longer than
the time limit of a batch queue runs as a chain of jobs of one script. Each job
submits its successor when it starts, with a dependency that is met only if the
job ends abnormally (time limit, node failure); if the job finishes normally,
the dependency can never be met and slurm removes the successor. A cap on the
number of jobs ends the chain if the fit keeps failing. In a slurm script:

.. code-block:: bash

   #SBATCH --time=24:00:00
   #SBATCH --open-mode=append        # the log of every job of the chain in one file
   SEGMENT=${SEGMENT:-1}; MAX_SEGMENTS=${MAX_SEGMENTS:-5}
   if [ "$SEGMENT" -lt "$MAX_SEGMENTS" ]; then
     SCRIPT=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=//p')
     sbatch --dependency=afternotok:$SLURM_JOB_ID --kill-on-invalid-dep=yes \
            --export=ALL,SEGMENT=$((SEGMENT + 1)) "$SCRIPT"
   fi
   python -m mcstas_gisans.fit ... --output_dir <the same directory in every job>

slurm sends ``SIGTERM`` at the time limit; ``mg_fit`` then records the status
``interrupted`` in ``fit_state.json`` and exits. The successor uses the script
file as it is when the successor is submitted (at the start of the job).

Joint fits of two measurements (``--nxs2``) with shared (``--fit_common``) and
separate (``--fit``/``--fit2``) parameters are described in :doc:`main_workflow`.

6. Loss function and Monte Carlo noise
--------------------------------------

.. _loss-functions:

6.1 Choosing the loss function (``--loss_function``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loss is the single number that the optimizer minimises: it measures how
well the simulation describes the measurement over the :math:`n` unmasked
pixels. Below, :math:`N_i` is the measured number of counts in pixel :math:`i`,
:math:`m_i` the simulated *expected* counts over ``--experiment_time``
(including the flat background, ``--background`` or ``--fit_background``) and
:math:`\sigma_i^2` the Monte Carlo variance of :math:`m_i`, which comes from the
finite number of simulated rays (section 6.2). All three losses are calculated
for every evaluation and written to ``fit_summary.csv``/``scan_summary.csv``;
``--loss_function`` selects the one that is minimised (and by which the
summaries are sorted). The same losses of the Qy profile of a :math:`Q_z` band
are described in section 6.3.

**poisson_deviance** (default). The measured counts are Poisson distributed: a
pixel with expected counts :math:`m` records :math:`N` counts with probability
:math:`P(N|m) = m^N e^{-m}/N!`. The Poisson deviance compares the probability of
the measurement under the model with that under a *saturated* model, which
predicts every pixel exactly (:math:`m_i = N_i`), the best any model can do:

.. math::

   D = \frac{2}{n}\sum_i \left[\ln P(N_i|N_i) - \ln P(N_i|m_i)\right]
     = \frac{2}{n}\sum_i \left[m_i - N_i + N_i \ln\frac{N_i}{m_i}\right]

(a pixel with :math:`N_i = 0` contributes :math:`2m_i`). Each term is zero where
the model predicts the measured counts and positive otherwise. Minimising
:math:`D` is the same as maximising the Poisson likelihood of the measurement,
i.e. it is the maximum-likelihood fit for counting data. Unlike the reduced
:math:`\chi^2` below, it does not push the model towards too high counts in pixels
with few counts (also at 0, 1 or 2 counts per pixel).
For many counts each term approaches :math:`(N_i - m_i)^2/m_i`, so :math:`D`
becomes the reduced :math:`\chi^2` below. This is a standard statistic under
several names: the (Poisson) deviance of generalised linear models [1]_, the
Cash statistic or C-statistic of X-ray astronomy [2]_ and the likelihood
:math:`\chi^2` of Baker and Cousins in particle physics [3]_. For a model that
describes the data within counting statistics, :math:`D` is about 1 per pixel:
its expectation is 1.00 at 20 counts per pixel, 1.03 at 8 and about 1.15 at 1
to 2 counts, and it drops below 1 for less than one count per pixel (0.47 at
0.1 counts). A value well above 1 means that the model does not describe the
data.

``mg_fit`` includes the Monte Carlo uncertainty of the simulation in this
likelihood: the true expectation of a pixel is taken as gamma distributed with
mean :math:`m_i` and variance :math:`\sigma_i^2`, which makes :math:`N_i`
negative-binomially distributed with mean :math:`m_i` and variance
:math:`m_i + \sigma_i^2` [4]_:

.. math::

   P(N|m,\sigma^2) = \frac{\Gamma(N+\alpha)}{\Gamma(\alpha)\,N!}
     \left(\frac{m}{m+\sigma^2}\right)^{\alpha}
     \left(\frac{\sigma^2}{m+\sigma^2}\right)^{N},
   \qquad \alpha = \frac{m^2}{\sigma^2} .

It replaces :math:`P(N_i|m_i)` in :math:`D` (the saturated term stays Poisson)
and becomes the Poisson probability for :math:`\sigma \to 0`. At high counts
each term approaches :math:`(N-m)^2/(m+\sigma^2) + \ln(1+\sigma^2/m)`, so a
perfect model gives about :math:`1 + \ln(1+\sigma^2/m)` per pixel.

**reduced_chi2**. Pearson's :math:`\chi^2` per pixel, the squared residual in
units of its expected standard deviation (the Poisson variance of the model
plus the Monte Carlo variance):

.. math::

   \chi^2_\mathrm{red} = \frac{1}{n}\sum_i \frac{(N_i - m_i)^2}{m_i + \sigma_i^2}

(divided by the number of pixels, not by the number of pixels minus the number
of fitted parameters, a negligible difference for thousands of pixels). For the
true model its expectation is 1 (without Monte Carlo variance) at any number of
counts, which makes it a familiar measure of the goodness of fit. Its *minimum*,
however, is biased when the pixels have few counts: the model is in the
denominator, so a higher :math:`m_i` lowers the penalty of the pixels with more
counts than predicted. For example, the flat level that minimises
:math:`\chi^2_\mathrm{red}` for Poisson counts of mean :math:`\mu` is
:math:`\sqrt{\langle N^2\rangle} \approx \mu + 1/2`, while :math:`D` gives the
mean, :math:`\mu`. This bias of about half a count per pixel is negligible at
hundreds of counts, but not at a few. In fits of the microgel measurement of
section 3 (median unmasked pixel about 8 counts), ``reduced_chi2`` fitted a flat
background about 0.5 counts per pixel higher than ``poisson_deviance``, while
the fitted sample parameters agreed.

**log_residual**. The mean of :math:`(\log_{10} N_i - \log_{10} m_i)^2` over the
pixels where both are positive. It compares relative deviations, so weak
regions weigh as much as bright ones, as on a logarithmic colour scale. It
ignores the counting statistics (a pixel with 3 counts weighs as much as one
with 3000, and pixels without counts are left out), does not include the Monte
Carlo variance, and has no reference value for a perfect model.

.. list-table::
   :header-rows: 1
   :widths: 20 26 27 27

   * - Loss
     - Perfect model
     - Few counts per pixel
     - Monte Carlo variance
   * - ``poisson_deviance``
     - about 1 per pixel
     - unbiased
     - included
   * - ``reduced_chi2``
     - 1 per pixel
     - biased (about +0.5 counts per pixel)
     - included
   * - ``log_residual``
     - no reference value
     - empty pixels left out, noisy
     - not included

Which one to use:

- ``poisson_deviance`` (default): the recommended choice for all fits. It is
  correct for counts at any level, which matters in GISANS, where much of the
  unmasked detector has only a few counts per pixel, and it is the loss to use
  with ``--fit_background``.
- ``reduced_chi2``: when a familiar :math:`\chi^2` is preferred, e.g. to compare
  with other analyses. With hundreds of counts in most pixels it gives the same
  parameters as ``poisson_deviance`` and a similar value; with few counts its
  fit is biased towards too high expected counts. Since every loss is written
  to the summary, a fit can minimise ``poisson_deviance`` and still report the
  reduced :math:`\chi^2` of the best evaluation.
- ``log_residual``: for a first look at the shape of the pattern over orders of
  magnitude (e.g. in a scan), not for final parameters.

Compare loss values only within one loss function, and only between runs with
the same simulation statistics (section 6.2). A value well above 1 means that
the model (or the mask, or the instrument setup) does not describe the data;
the fitted parameters are then the best compromise, not necessarily the true
values.

.. [1] P. McCullagh and J. A. Nelder, *Generalized Linear Models*, 2nd ed.,
   Chapman & Hall, London (1989).
.. [2] W. Cash, Parameter estimation in astronomy through application of the
   likelihood ratio, ApJ 228 (1979) 939,
   `doi:10.1086/156922 <https://doi.org/10.1086/156922>`__.
.. [3] S. Baker and R. D. Cousins, Clarification of the use of chi-square and
   likelihood functions in fits to histograms, Nucl. Instrum. Meth. 221 (1984)
   437, `doi:10.1016/0167-5087(84)90016-4 <https://doi.org/10.1016/0167-5087(84)90016-4>`__.
.. [4] C. A. Argüelles, A. Schneider and T. Yuan, A binned likelihood for
   stochastic models, JHEP 06 (2019) 030,
   `doi:10.1007/JHEP06(2019)030 <https://doi.org/10.1007/JHEP06(2019)030>`__.

.. _fit-monte-carlo-noise:

6.2 Monte Carlo noise
~~~~~~~~~~~~~~~~~~~~~

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

``poisson_deviance`` and ``reduced_chi2`` include the Monte Carlo variance of
the simulation, so a noisier simulation scores a *lower* loss for the same
pattern (in the example of section 9 the Poisson deviance rose from 16 to 44 when
the rays per pixel went from about 250 to 7000). Compare losses only between
runs with the same MCPL file and the same outgoing directions; with
``--sampling``, use the printed
``--outgoing_directions_horizontal``/``--outgoing_directions_vertical`` numbers
to repeat a run with exactly the same grid.

``mg_fit`` warns if the Monte Carlo variance exceeds the counting variance in
more than 5% of the pixels, or if it lowers the loss of the fit by more than
10% compared with the loss without it (:math:`\sigma_i = 0`). The second case
happens also when only a few pixels are affected: where the pattern is steep
compared with the outgoing-direction grid, the simulated counts are noisy, and
a model that misses the data there (e.g. overshoots it 2-3x) is hardly
penalised. In a fit of the microgel measurement of section 3 (41 x 33 and
123 x 99 outgoing directions; about 6% of the pixels affected with 41 x 33),
the losses near the best parameters were (without the Monte Carlo term in
brackets):

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Outgoing directions
     - ``poisson_deviance``
     - ``reduced_chi2``
   * - 41 x 33
     - 2.53-2.80 (3.9-4.4)
     - 2.73-3.15 (4.0-4.6)
   * - 123 x 99
     - 3.06-3.24 (3.5-3.8)
     - 3.33-3.45 (3.7-4.0)

The coarse grid scores lower only because its simulation is noisier. The
warning states the loss with and without the Monte Carlo term; increase
``--outgoing_directions`` (especially along the direction in which the pattern
is steep) or the number of simulated neutrons until the two are close.

.. _fit-1d-profile:

6.3 Qy profile of a Qz band (``--fit_objective``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below the 2D maps, the comparison plots (``--png``) show the Qy profile of the
:math:`Q_z` band given by ``--q_min`` and ``--q_max``: for every Qy bin (detector
column), the counts of the unmasked pixels within the band are summed. The band
consists of whole :math:`Q_z` bins, from the one containing ``--q_min`` to the
one containing ``--q_max``; its exact limits are in the title of the 1D panel,
and ``mg_fit`` prints them with the number of unmasked pixels in the band.

If the band is given (``--q_min`` < ``--q_max``), the profiles of the
measurement and the simulation are compared as well, from the same unmasked
pixels as the 2D loss. For Qy bin :math:`j` with :math:`n_j` unmasked pixels
:math:`i` in the band,

.. math::

   N_j = \sum_i N_i, \qquad m_j = \sum_i m_i, \qquad \sigma_j^2 = \sum_i \sigma_i^2 ,

where :math:`m_j` includes :math:`n_j b` of a flat background :math:`b` per
pixel. A sum of Poisson counts is Poisson distributed, so the losses of section
6.1 apply unchanged, with :math:`n` the number of Qy bins that contain at least
one unmasked pixel of the band (bins without are left out). They are printed
with every evaluation and written to the summaries as ``poisson_deviance_1d``,
``reduced_chi2_1d`` and ``log_residual_1d``, next to the losses of the detector
image (``poisson_deviance``, ``reduced_chi2``, ``log_residual``); in joint fits, like the 2D losses,
per measurement (``..._1d_sample1``, ``..._1d_sample2``) and summed.

``--fit_objective`` selects the comparison whose ``--loss_function`` is
minimised and sorts the summaries:

- ``2d`` (default): the unmasked detector pixels. The 1D losses are only
  reported.
- ``1d``: the Qy profile of the band (requires ``--q_min`` < ``--q_max`` or
  ``--band_exit_angle``, and unmasked pixels in the band). ``--fit_background`` then fits :math:`b` to the
  profile.

Instead of a :math:`Q_z` band, ``--band_exit_angle LOW HIGH`` [deg] defines the
band by the exit angle relative to the specular direction: the pixels whose
exit angle :math:`\alpha_f` above the sample surface has
:math:`\text{LOW} \le \alpha_f - \alpha \le \text{HIGH}` (:math:`\alpha` = ``--alpha``,
the specular exit angle). The exit angle of a pixel centre is that of the Q
conversion, :math:`Q_z = k (\sin\alpha_f + \sin\alpha)`, including the gravity
correction at the wavelength. For monochromatic data this is a fixed set of
pixels; its use is with time-of-flight data (section 10), where the specular
:math:`Q_z` moves with the wavelength. The profile then sums the unmasked
pixels of this band per Qy bin, and the 1D panel of the plots shows it.

The 2D loss weighs every unmasked pixel alike, so a feature that occupies a
small part of the unmasked region, such as the intensity drop before the first
peak in a narrow band, contributes little to it. The profile concentrates on
the band: a misfit common to the pixels of a Qy bin adds up in their sum,
while deviations that change sign along :math:`Q_z` within the band cancel. It
ignores everything else: the :math:`Q_z` dependence within the band and all
pixels outside it, so a model can match the profile and miss the rest of the
pattern. The two values are therefore worth comparing: a fit to one comparison
that worsens the other is a hint that the model or the setup describes the data
only partly. The 1D losses are normalised per Qy bin, not per pixel, and with
many counts per bin the same relative misfit gives a larger value than in 2D, so
compare 1D values only with 1D values.

Summing the pixels also adds their Monte Carlo variances, and in the bright
bins of the profile they can exceed the counting variance even where they do
not in the single pixels. In a fit of the microgel measurement of section 3 (band
0.149-0.170 nm\ :sup:`-1`, 40 Qy bins of 9 pixels, 41 x 33 outgoing
directions), the Monte Carlo term lowered the 2D Poisson deviance from 4.79 to
2.61, but the 1D one from 26.1 to 5.64: the peak bins, where the model was
about 1.5 times too intense, were hardly penalised. The warning of section 6.2
checks the minimised loss (with ``--fit_objective 1d`` the 1D one) and states
it with and without the Monte Carlo term; before relying on a fit of the
profile, increase the simulated statistics until the two are close.

7. Outputs
----------

In ``--output_dir`` (default ``scan_results``):

- ``fit_summary.csv`` / ``scan_summary.csv``: every evaluation with its
  parameters, all loss values (with ``--q_min`` < ``--q_max`` also those of the
  Qy profile, ``..._1d``, section 6.3) and, with ``--fit_background``, the
  fitted background, sorted by the selected loss. In fits it is written after
  every evaluation (in the order made; sorted at the end) and has a ``phase``
  column (``de``, ``polish``, or the local optimiser).
- Fits: ``fit_state.json`` (settings, status ``running``, ``interrupted``,
  ``failed`` or ``finished``, one entry per run of the fit with host, slurm job
  and the numbers of retraced and new evaluations, and the result) and
  ``fit_evaluations.jsonl`` (every evaluation with the exact parameter vector of
  the optimiser), used to continue the fit (section 5.2).
- With ``--png``: a comparison plot of the measurement and the simulation for
  every evaluation (``fit_eval_<n>_<parameters>.png`` in fits,
  ``sim_<parameters>.png`` in scans). ``--gif`` turns the fit plots into an
  animation. With ``--split_view`` the two 2D maps become one, with the
  measurement for Qy < 0 and the simulation for Qy > 0 on a common colour
  scale.

The summary with the best parameters, the optimizer's status and the run time
is also printed. "Maximum number of function evaluations has been exceeded"
means the budget ran out before the tolerances were met; if the loss was still
decreasing, run the same command again with a larger ``--max_evals`` (and
``--polish_max_evals``) and the fit goes on where it stopped (section 5.2).

8. Run time
-----------

The cost of one evaluation grows with the number of MCPL particles and with the
square of ``-n`` (the outgoing directions per axis). ``--sampling`` chooses the
directions for a target noise per pixel (``quick``, the default, ``standard``,
``long``) and takes the (effective) number of neutrons into account, so a larger
MCPL file automatically gets fewer directions per neutron. ``-p`` sets the number of
parallel processes; with many processes use ``--bornagain_number_of_threads 1``.
``--simulate_mask_angle_range`` saves time when much of the detector is masked.
On a cluster, ``--cluster_nodes`` spreads every simulation over several nodes of a job with the same results
(about 6.5x faster on 8 nodes; see the DMSC cluster page). With Differential Evolution, ``--de_parallel``
simulates all members of a generation together, which keeps many nodes busy: on LUMI, 16 nodes were 11.7x faster
than one node with it and 7.6x without, and 64 nodes 33x faster (one evaluation at a time does not gain beyond
about 16 nodes). It changes the trial points (scipy's "deferred" instead of "immediate"
updating), so it gives a different, equally valid fit; a fit is continued with the same setting.
A scan with a few points is usually worth more than a long fit from a poor
starting point.

9. Worked example: silica nanoparticles on silicon (D22)
--------------------------------------------------------

The data of this example are included in the repository (``data/paper``): the
D22 measurement of silica nanoparticles on a silicon substrate in air published
in the mcstas_gisans paper (:doc:`cite`; ``073174.nxs``, 3 hours at 0.24°,
vertical sample with its normal pointing left), and the McStas simulation of
the beam of that measurement. The detector offset and the intensity
factor come from the direct beam measurement ``073162.nxs``
(see :doc:`replicating_measurements`); the region below
:math:`Q_z = 0.14\,\mathrm{nm}^{-1}` (transmitted beam, specular reflection and
Yoneda region) is masked:

.. code-block:: bash

   mg_fit data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz \
     --nxs data/paper/d22_measurement/073174.nxs --experiment_time 10800 --fit_background \
     --instrument d22 --wavelength_selected 6.0 --intensity_factor 0.2107 \
     --alpha 0.24 --sample_orientation 2 --instrument_detector_centre_offset 0.290838 -0.016061 \
     --model silica_100nm_air --sample_arguments "interferenceRange=5" \
     --sample_size_y 0.06 --sample_size_x 0.08 --allow_sample_miss \
     --specular specular_simulation --use_avg_materials \
     --mask_qz_min_cut 0.14 \
     --fit radius 51 45 57 --fit latticeParameter 114 100 130 \
     --max_evals 20 --seed 1 --output_dir fit_example

``mg_fit`` prints the number of unmasked pixels, the chosen outgoing directions
(here the ``quick`` preset) and, for every evaluation, the parameters, the loss
and the fitted background; ``fit_summary.csv`` lists them sorted by the loss.
If the best evaluations differ by less than the Monte Carlo noise of the loss
(repeat the best one with a few ``--seed`` values, section 6.2), more simulated
statistics (``--sampling standard``) are needed to pin the parameters down more
precisely.

10. Time-of-flight instruments (``--wavelength_slice``)
--------------------------------------------------------

For a TOF instrument the measurement and the simulation are compared in
wavelength slices, given with ``--wavelength_slice MIN MAX`` [Å] (required; it
can be given several times). The sample model is the same at all wavelengths,
so every slice is evaluated with the same parameter set from one simulation,
and the loss is the sum of the losses of the slices (each also reported, with
the suffix ``_slice<n>`` in the summaries, when there are several).

Both sides go through the same reduction:

- The wavelength of a TOF bin of a pixel is :math:`\lambda = h t / (m (L_1 + L_2))`
  at the bin centre :math:`t`, with :math:`L_1` the nominal source-sample
  distance of the instrument (``--instrument_nominal_source_sample_distance``)
  and :math:`L_2` the sample-pixel distance. The TOF of the simulated neutrons
  start at the McStas source and those of a raw measurement at the source pulse
  (e.g. the proton pulse at ISIS), so no T0 correction may be applied to the
  simulation: ``--no_t0_correction`` is required (or ``--t0_fixed`` for a known
  offset). The same :math:`L_1` is used for both sides (for SANS2D the McStas
  path 19.2625 m; the 19.281 m of the Mantid instrument definition would shift
  the measured wavelengths by 0.06%).
- The TOF bins whose centre wavelength is in the slice are converted to
  :math:`(Q_y, Q_z)` at the pixel centre with that wavelength (incident angle,
  sample orientation, detector offset and gravity as for monochromatic data)
  and summed into the Q grid of the slice: the pixel edges of the detector at
  the centre wavelength of the slice, so the ``--mask_*`` options, the
  :math:`Q_z` band and the plots work as for monochromatic data. Q bins that
  receive no (pixel, TOF bin) of the slice are left out.
- The simulation histograms the detected neutrons in the TOF bins of the
  measurement per pixel, and these histograms go through the same conversion.
  The MCPL input is filtered with the TOF range of the slices over
  :math:`L_1`, widened by 10% (or ``--input_tof_limits``/``--no_mcpl_filtering``),
  and after the preconditioning only the neutrons that can be detected in a
  TOF bin of a slice are simulated.
- The flat background (``--background``, ``--fit_background``) is in counts per
  detector pixel over the slice, spread over the TOF bins of the slice of each
  pixel in proportion to their widths (a background constant in time) and
  converted to Q in the same way. ``--fit_background`` fits it for every slice.

The ``--mask_*`` options in Q apply to every slice alike, although the specular
reflection moves in :math:`Q_z = 2 k \sin\alpha` with the wavelength. Regions that
follow it are given by the exit angle relative to the specular direction:
``--mask_exit_angle_band LOW HIGH`` [deg] keeps only the pixels whose exit angle
has :math:`\text{LOW} \le \alpha_f - \alpha \le \text{HIGH}`, and
``--band_exit_angle LOW HIGH`` defines the band of the 1D profile in the same
way (section 6.3). The band is a selection of detector pixels: in every slice,
the exit angle of each pixel centre is calculated at the centre wavelength of
the slice (the exit angle of the Q conversion, with its gravity correction), and
all TOF bins of the slice use the same pixels, for the measurement, the
simulation and the background alike. A Q bin of the slice can also receive
other pixels at other wavelengths of the slice; only the counts of the selected
pixels are compared. Without gravity the exit angle of a pixel does not depend
on the wavelength; with gravity it rises with the wavelength by the drop over
the sample-detector distance divided by that distance (0.03° at 12 Å over 12 m),
so the band moves down the detector from slice to slice by this much, and
within a wide slice the specular can move by a fraction of it (narrower slices
reduce this). For example, one tube (8.1 mm at 12 m: 0.0387°) below to six tubes
above the specular, without the specular core and its Qy wings:

.. code-block:: bash

   --mask_exit_angle_band -0.0387 0.232 --band_exit_angle -0.0387 0.232 --mask_qy_range -0.03 0.03

``--intensity_factor`` is one calibration for all wavelengths.
``--intensity_factor_spectrum FILE`` adds a wavelength-dependent one: a text
file with two columns, wavelength [Å] and factor (lines starting with ``#`` are
comments), covering the wavelength slices. The simulated count of every
(pixel, TOF bin) is multiplied by the factor interpolated linearly at its
wavelength (the squared Monte Carlo weights by its square), in addition to
``--intensity_factor``. It corrects the spectrum of the simulated beam with a
direct-beam measurement:

1. Measure the direct beam (no sample, same slits; attenuated if needed) and
   simulate it with the same McStas beam, i.e. the same instrument file and
   MCPL plane, and the sample replaced by nothing, so that the beam reaches the
   detector.
2. Histogram both in the same wavelength bins, the measured and the simulated
   wavelengths both from the TOF over :math:`L_1 + L_2` as in the slices (e.g.
   bins of 0.05 Å, summed over the detector or the beam spot).
3. Scale the simulation to the measurement time or proton charge of the direct-
   beam run (and the measured counts by any attenuator transmission), and write
   the ratio measured / simulated per bin, smoothed over a few bins where the
   statistics are poor, as the file.

The factor then absorbs the spectrum and normalisation differences between the
McStas beam and the real one; ``--intensity_factor`` stays the normalisation of
the McStas weights to the measurement (e.g. its proton charge).

``--png`` saves a
comparison plot per slice (the slices after the first with the suffix
``_slice<n>``), ``--mask_view`` a mask plot per slice, and the scans save the
simulated Q histogram of every slice (``.npz``, readable by ``mg_plot``).

Measured data: raw ISIS SANS2D NeXus files are read (the rear tube detector,
recognised by ``raw_data_1/beamline`` or with ``--nxs_format isis_sans2d``).
For example, run 60998 (100 nm silica on silicon, 0.328°) in the slice
6.5–7.5 Å, with a McStas beam of the measurement:

.. code-block:: bash

   mg_fit test_events.mcpl.gz --nxs SANS2D00060998.nxs --experiment_time 17257 --fit_background \
     --instrument sans2d --no_t0_correction --wavelength_slice 6.5 7.5 --intensity_factor 47.121 \
     --alpha 0.328 --sample_orientation 1 \
     --model silica_100nm_air --sample_arguments "latticeParameter=114;interferenceRange=5;positionVariance=20;defectAbundance=0.0" \
     --sample_size_x 0.08 --sample_size_y 0.06 --allow_sample_miss --specular specular_simulation \
     --mask_qy_min_cut -0.2 --mask_qy_max_cut 0.2 --mask_qz_min_cut 0.0937 --mask_qz_max_cut 0.1119 \
     --mask_exclude_q_box -0.025 0.025 0.0937 0.1119 --q_min 0.0937 --q_max 0.1119 \
     --simulate_mask_angle_range --scan radius 51

``mg_plot`` reads the same raw file as event data (the detection events, each with
its own TOF) and reduces it like the simulated TOF events, so a measured
wavelength slice can be shown next to a simulation of the same geometry:

.. code-block:: bash

   mg_plot --nxs SANS2D00060998.nxs --nxs_label measured --filename simulation.h5 --label simulated \
     --experiment_time 511931 --wavelength_slice 6.5 7.5 --split_view --png

(the simulation's McStas weights are rates per µA, so ``--experiment_time`` is
the proton charge in µA·s: 142.2 µAh = 511931 µA·s; the instrument
configuration of the measurement is that of the simulation file, or
``--instrument sans2d --alpha ... --sample_orientation 1`` without one).
