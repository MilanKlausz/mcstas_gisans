=============================
Creating Custom Sample Models
=============================

One of the most powerful features of ``mcstas_gisans`` is the ability to fit custom sample models directly against experimental data using ``mg_fit``. 

To do this, you must provide a Python script containing a `get_sample(**kwargs)` function. This script must be placed inside the `src/mcstas_gisans/bornagain_samples/` directory, or in the current working directory.

Basic Template
--------------

Here is a minimal, fully-commented template for a custom sample model.

.. code-block:: python

    import bornagain as ba

    def get_sample(**kwargs):
        """
        Dynamically constructs a BornAgain sample based on fitting parameters.
        
        Any keyword arguments passed here match the names given in the CLI
        `--fit_parameters` or `--sample_arguments` flags.
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

To run a grid fit over the ``radius`` and ``height`` parameters, execute:

.. code-block:: bash

    mg_fit resources/mcstas_models/output_dir/test_events.mcpl.gz \
      --nxs d22_experiment.nxs \
      --instrument d22 \
      --model my_custom_sample \
      --wavelength_selected 6.0 \
      --fit_parameters "radius=5:1:20;height=10:5:50" \
      --mask_rect "ymin=-0.05 ymax=0.05 zmin=-0.02 zmax=0.02"

The ``--fit_parameters`` flag uses a specific syntax: ``parameter=start:step:max``. 
In this case, ``radius=5:1:20`` tells the optimizer to search for radii starting at 5 nm, stepping by 1 nm, up to a maximum of 20 nm. The framework automatically maps these values to the ``**kwargs`` dictionary passed into your ``get_sample`` function.

Under the hood, ``mg_fit`` uses `scipy.optimize.differential_evolution` (or Nelder-Mead, depending on the configuration) to minimize a Poisson-weighted cost function between the simulated scattering pattern and the experimental NeXus data over the masked regions.
