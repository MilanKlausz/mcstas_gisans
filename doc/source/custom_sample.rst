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

    def get_sample(**kwargs):
        """
        Dynamically constructs a BornAgain sample based on fitting parameters
        (BornAgain 22/23; see "BornAgain versions" below for BornAgain 24).

        Any keyword arguments passed here match the names given in the CLI
        `--fit` or `--sample_arguments` flags.
        """
        # 1. Extract parameters with default fallbacks
        radius = kwargs.get('radius', 5.0)     # Default 5.0 nm
        height = kwargs.get('height', 10.0)    # Default 10.0 nm

        # 2. Define materials (scattering length densities in 1/A^2)
        material_air = ba.MaterialBySLD("Air", 0.0, 0.0)
        material_particle = ba.MaterialBySLD("Particle", 4.0e-6, 0.0)
        material_substrate = ba.MaterialBySLD("Substrate", 6.36e-6, 0.0)

        # 3. Create the particle shape
        particle = ba.Particle(material_particle, ba.Cylinder(radius*nm, height*nm))

        # 4. Create an interference function (a hexagonal 2D lattice, order decaying over 300 nm)
        interference = ba.Interference2DLattice(ba.HexagonalLattice2D(20.0*nm, 0*deg))
        interference.setDecayFunction(ba.Profile2DCauchy(300*nm, 300*nm, 0))

        # 5. Combine into a particle layout
        layout = ba.ParticleLayout()
        layout.addParticle(particle)
        layout.setInterference(interference)

        # 6. Define the layers: the particles sit on the substrate (bottom of the top layer)
        air_layer = ba.Layer(material_air)
        air_layer.addLayout(layout)
        substrate_layer = ba.Layer(material_substrate)

        # 7. Assemble the sample
        sample = ba.Sample()
        sample.addLayer(air_layer)
        sample.addLayer(substrate_layer)

        return sample

BornAgain versions
------------------

BornAgain's Python API changes between major versions: BornAgain 22 replaced ``ba.MultiLayer``
with ``ba.Sample`` and moved the interface roughness into the layers, and BornAgain 24 changed
the materials (``ba.MaterialBySLD`` became ``ba.SLDMaterial`` with a colour argument) and replaced
the particle layouts (``ba.ParticleLayout`` with an interference function) by structures that
contain the particles (``ba.FiniteCrystal2D``, ``ba.Crystal2D``, ``ba.Paracrystal2D``, a
``ba.Mixture`` for several particle types) placed with ``layer.deposit2D``/``layer.suspend2D``.
A model is therefore written for one API.

The built-in models are kept in version folders of ``bornagain_samples/``, named after the first
BornAgain major version of the API they are written for: ``ba21/`` (BornAgain 21), ``ba22/``
(BornAgain 22 and 23) and ``ba24/`` (BornAgain 24). A model can have an implementation in several
folders under the same name, and a file can declare the major versions it is tested with:

.. code-block:: python

    # BornAgain versions this implementation is tested with (first, last major version)
    BORNAGAIN_VERSIONS = (22, 23)

``--model <name>`` uses the implementation whose range contains the installed BornAgain version;
an implementation without ``BORNAGAIN_VERSIONS`` is treated as compatible with any version. If
neither exists, the run stops with an error that lists the available implementations: an
implementation is never used silently with a BornAgain version outside its declared range (a newer
BornAgain can change the results without breaking the API). ``--allow_untested_bornagain_version``
uses the implementation for the newest older version anyway, with a warning. Declaring the versions
is suggested, not required. The folder layout is explained in ``bornagain_samples/README.md``. ``--help`` lists the
built-in models with their tested versions. Model files given as a path, and files placed directly
in ``bornagain_samples/``, are not version-checked.

To support a new BornAgain version (say 25): run the tests with it.
``tests/test_sample_versions.py`` simulates every model available for the version against its
stored reference results (``tests/data/builtin_model_references.json``); for a model whose
implementation still works and gives the reference results, extend its range, e.g.
``BORNAGAIN_VERSIONS = (24, 25)``. For a model that needs the new API, add an implementation with the
same name and the same ``get_sample`` parameters in a new folder ``ba25/`` with
``BORNAGAIN_VERSIONS = (25, 25)``, write its reference results with
``python tests/make_builtin_model_references.py`` (with BornAgain 25 installed), and check that they
agree with those of the older implementation. Models that are not updated are simply not available
with the new version.

Differences to keep in mind when writing a BornAgain 24 implementation of a 22/23 model:

- the position variance: ``setPositionVariance(v)`` with a variance v [nm\ :sup:`2`] became
  ``setLateralPositionVariance(s)`` with the root-mean-square displacement s = sqrt(v) [nm];
- the vertical position: ``layer.suspend2D`` (particles below the top interface of a layer, as
  ``addLayout`` in a lower layer before) positions a particle by its *top*, ``layer.deposit2D``
  (particles on the bottom of the top layer) by its bottom; BornAgain 22/23 positioned it by its
  origin, the bottom of the form factor;
- the particle surface density is one particle per unit cell of the lattice in all versions
  (BornAgain 22/23 ignore ``setTotalParticleSurfaceDensity`` with a 2D lattice);
- BornAgain 24 uses average materials by default (``mcstas_gisans`` always sets the option, see
  ``--use_avg_materials``); with average materials the BornAgain 24 implementations of the
  built-in models give the same results as the 22/23 ones, without them particles inside a lower
  layer give a much lower intensity with BornAgain 24.1.

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
