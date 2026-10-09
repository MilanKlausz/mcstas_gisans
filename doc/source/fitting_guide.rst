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
  unmasked pixels (with ``--fit_objective 1d`` over the Qy profile of the
  :math:`Q_z` band, section 6.3) is found by a one-dimensional minimisation between 0 and the
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
  the background by about 0.5 counts per pixel (see :ref:`loss-functions`).
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
hundreds of counts, but not at a few. In fits of the high-resolution microgel
data, where the median unmasked pixel has about 8 counts, ``reduced_chi2``
fitted a flat background about 0.5 counts per pixel higher than
``poisson_deviance``; both found the same range of lattice sizes.

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
pattern (for the D22 paper data the Poisson deviance rose from 16 to 44 when
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
penalised. In a fit of the high-resolution microgel data (about 6% of the
pixels affected on the 41 x 33 grid), the losses near the best parameters were
(without the Monte Carlo term in brackets):

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
image (whose column names do not change); in joint fits, like the 2D losses,
per measurement (``..._1d_sample1``, ``..._1d_sample2``) and summed.

``--fit_objective`` selects the comparison whose ``--loss_function`` is
minimised and sorts the summaries:

- ``2d`` (default): the unmasked detector pixels. The 1D losses are only
  reported.
- ``1d``: the Qy profile of the band (requires ``--q_min`` < ``--q_max`` and
  unmasked pixels in the band). ``--fit_background`` then fits :math:`b` to the
  profile.

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
not in the single pixels. In the high-resolution microgel example (band
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
  fitted background, sorted by the selected loss.
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
     --alpha 0.24 --sample_orientation 2 --instrument_detector_centre_offset 0.290838 -0.016061 \
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
(repeat the best one with a few ``--seed`` values, section 6.2), more simulated
statistics (``--sampling standard``) are needed to pin the parameters down more
precisely.
