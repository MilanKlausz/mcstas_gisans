=============================
Creating Custom Sample Models
=============================

One of the most powerful features of ``mcstas_gisans`` is the ability to fit custom sample models directly against experimental data using ``mg_fit`` (the same mechanism is also used by ``mg_run`` for a single simulation).

To do this, you must provide a Python script containing a ``get_sample(**kwargs)`` function. Pass its name (without ``.py``) or path via ``--model``. It is resolved in two steps:

1. **Local file:** if ``--model`` names an existing ``.py`` file (an absolute path, or a path relative to your current working directory), that file is loaded directly.
2. **Built-in model:** otherwise, ``--model`` is looked up by name among the built-in scripts in ``src/mcstas_gisans/bornagain_samples/`` (see ``--help`` for the current list, e.g. ``silica_100nm_air``, ``silica_100nm_D2O``, ``hexagonal_spheres``, ``lamellas_and_spheres``).

If neither resolves, ``mg_run``/``mg_fit`` exits with an error listing how to see the built-in options.

Basic Template
--------------

Here is a minimal, fully-commented template for a custom sample model.

.. code-block:: python

    import bornagain as ba
    from bornagain import deg, nm

    # helpers that build the same sample with BornAgain 22, 23 and 24 (see below)
    from mcstas_gisans.ba_compat import sld_material, hexagonal_lattice, infinite_lattice, add_particles

    def get_sample(**kwargs):
        """
        Dynamically constructs a BornAgain sample based on fitting parameters.

        Any keyword arguments passed here match the names given in the CLI
        `--fit` or `--sample_arguments` flags.
        """
        # 1. Extract parameters with default fallbacks
        radius = kwargs.get('radius', 5.0)     # Default 5.0 nm
        height = kwargs.get('height', 10.0)    # Default 10.0 nm

        # 2. Define materials (scattering length densities in 1/A^2)
        material_air = sld_material("Air", 0.0, 0.0)
        material_particle = sld_material("Particle", 4.0e-6, 0.0)
        material_substrate = sld_material("Substrate", 6.36e-6, 0.0)

        # 3. Create the particle shape
        particle = ba.Particle(material_particle, ba.Cylinder(radius*nm, height*nm))

        # 4. Arrange the particles on a hexagonal 2D lattice (with a decay of the order over 300 nm)
        order = infinite_lattice(hexagonal_lattice(20.0*nm, 0*deg),
                                 decay_function=ba.Profile2DCauchy(300*nm, 300*nm, 0))

        # 5. Define the layers; the particles sit on the substrate, i.e. on the bottom of the top (air) layer
        air_layer = ba.Layer(material_air)
        add_particles(air_layer, [(particle, 1.0)], order, top_layer=True)
        substrate_layer = ba.Layer(material_substrate)

        # 6. Assemble the sample
        sample = ba.Sample()
        sample.addLayer(air_layer)
        sample.addLayer(substrate_layer)

        return sample

BornAgain versions
------------------

BornAgain's Python API changes between major versions: BornAgain 22 replaced ``ba.MultiLayer`` with
``ba.Sample`` and moved the interface roughness into the layers, and BornAgain 24 changed the
materials (``ba.MaterialBySLD`` became ``ba.SLDMaterial`` with a colour argument) and replaced the
particle layouts (``ba.ParticleLayout`` with an interference function) by structures that contain
the particles (``ba.FiniteCrystal2D``, ``ba.Crystal2D``, ``ba.Paracrystal2D`` placed with
``layer.deposit2D``/``layer.suspend2D``). A model written directly for one version fails with the
other. The module ``mcstas_gisans.ba_compat`` builds the same sample with BornAgain 22, 23 and 24:

- ``sld_material(name, sld_real, sld_imag)``, ``refractive_material(name, delta, beta)``;
- ``roughness(sigma, hurst, lateral_corr_length)`` (the self-affine fractal roughness with a tanh
  profile, BornAgain 21's ``ba.LayerRoughness``), passed to ``ba.Layer(material, thickness, roughness)``;
- ``basic_lattice(...)``/``hexagonal_lattice(...)`` and the lateral order ``finite_lattice(lattice, n_1,
  n_2, integrate_xi, position_variance)``, ``infinite_lattice(lattice, decay_function, ...)`` or
  ``paracrystal(lattice, damping_length, domain_size_1, domain_size_2, pdf_1, pdf_2, ...)``, with the
  position variance in nm\ :sup:`2` (BornAgain 22/23 convention);
- ``add_particles(layer, [(particle, abundance), ...], order, top_layer=False)`` (one particle per unit cell):
  several particle types are an incoherent mixture with these relative abundances; the particle
  z positions are relative to the bottom of the layer for the top layer (``top_layer=True``) and
  relative to its top otherwise, as in BornAgain 22/23. The reference point of a particle must be its
  bottom (as for every BornAgain form factor): build a ``ba.Compound`` with its lowest point at its own
  origin and place it with ``compound.translate()``. (BornAgain 24 itself positions a particle suspended
  below an interface by its top; the helper converts.)

All built-in models in ``bornagain_samples/`` use these helpers. Particle shapes (``ba.Sphere``,
``ba.Cylinder``, ...), ``ba.Particle``, ``ba.Compound``, ``ba.Layer`` and ``ba.Sample`` are the same
in BornAgain 22, 23 and 24 and are used directly.

Using the Template with ``mg_fit``
----------------------------------

If you save the above script as ``my_custom_sample.py`` in the ``bornagain_samples`` directory (or your current working directory), you can run an automated fit. 

You must provide a real experimental NeXus file to fit against (e.g., from an ILL D22 measurement). For this example, we assume you have downloaded an experimental file named ``d22_experiment.nxs`` and that you generated an MCPL file named ``test_events.mcpl.gz`` during the Quickstart.

To fit the ``radius`` and ``height`` parameters, execute:

.. code-block:: bash

    mg_fit resources/mcstas_models/output_dir/test_events.mcpl.gz \
      --nxs d22_experiment.nxs \
      --instrument d22 \
      --model my_custom_sample \
      --wavelength_selected 6.0 \
      --fit radius 5 20 \
      --fit height 10 50 \
      --mask_exclude_q_box -0.05 0.05 -0.02 0.02

The ``--fit`` flag can be repeated once per parameter, and accepts either ``name x0`` (fixed
initial guess, unbounded), ``name min max`` (bounds, with the initial guess set to their
midpoint), or ``name x0 min max`` (explicit initial guess and bounds). In this case,
``--fit radius 5 20`` tells the optimizer to search for radii between 5 nm and 20 nm. The
framework automatically maps the fitted values to the ``**kwargs`` dictionary passed into your
``get_sample`` function. Any keyword your function does not declare (and does not catch via
``**kwargs``) is silently dropped, with a warning printed to the console.

Under the hood, ``mg_fit`` selects an optimizer with ``--optimizer``: ``nelder-mead`` (default),
``powell``, or ``differential-evolution`` (which requires finite bounds on every fitted
parameter). Each iteration re-runs the full McStas-particles-through-BornAgain simulation and
scores it against the experimental NeXus data over the *unmasked* detector region only, using a
loss function selected with ``--loss_function`` (``poisson_deviance`` by default, or
``reduced_chi2``/``log_residual``; see :ref:`loss-functions` for which to choose) — see
:doc:`main_workflow` for how ``--mask_*`` options control which region counts towards the loss,
and for the simpler grid-search alternative, ``--scan``.
