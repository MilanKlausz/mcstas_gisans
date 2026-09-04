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

    def get_sample(**kwargs):
        """
        Dynamically constructs a BornAgain sample based on fitting parameters.
        
        Any keyword arguments passed here match the names given in the CLI
        `--fit` or `--sample_arguments` flags.
        """
        # 1. Extract parameters with default fallbacks
        radius = kwargs.get('radius', 5.0)     # Default 5.0 nm
        height = kwargs.get('height', 10.0)    # Default 10.0 nm
        
        # 2. Define materials
        material_air = ba.MaterialBySLD("Air", 0.0, 0.0)
        material_particle = ba.MaterialBySLD("Particle", 4.0e-6, 0.0)
        material_substrate = ba.MaterialBySLD("Substrate", 6.36e-6, 0.0)
        
        # 3. Create the particle shape
        ff = ba.Cylinder(radius, height)
        particle = ba.Particle(material_particle, ff)
        
        # 4. Create an interference function (e.g., 2D lattice)
        interference = ba.Interference2DLattice(ba.HexagonalLattice2D(20.0, 0))
        
        # 5. Combine into a particle layout
        layout = ba.ParticleLayout()
        layout.addParticle(particle)
        layout.setInterference(interference)
        
        # 6. Define the layers
        air_layer = ba.Layer(material_air)
        air_layer.addLayout(layout)
        substrate_layer = ba.Layer(material_substrate)
        
        # 7. Assemble the multi-layer sample
        sample = ba.MultiLayer()
        sample.addLayer(air_layer)
        sample.addLayer(substrate_layer)
        
        return sample

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
loss function selected with ``--loss_function`` (``reduced_chi2`` by default, or
``log_residual``) — see :doc:`main_workflow` for how ``--mask_*`` options control which region
counts towards the loss, and for the simpler grid-search alternative, ``--scan``.
