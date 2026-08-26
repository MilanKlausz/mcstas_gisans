===============================
Architecture & Workflow Details
===============================

This document provides in-depth technical details about the architecture of `mcstas_gisans`, the simulation workflows, and the data structures used.

1. Workflows: TOF vs. Non-TOF
-----------------------------

The simulation workflow diverges depending on whether the experiment utilizes Time-of-Flight (TOF) data or non-TOF data:

* **Non-TOF Workflow**: The `mg_run` (via `run.py`) script simulates the particles and directly bins them into a 2D intensity grid representing the detector pixels. 
* **TOF Workflow**: The simulation produces intermediate event-based datasets. Events are preserved with their wavelength and time-of-flight information. These events are saved into intermediate Scipp HDF5 (`.h5`) files. Currently, the data is exclusively saved in **binned data** format (often referred to as event mode), preserving the exact TOF of every particle. A **dense data** (histogrammed TOF slices) mode is planned for a future release to drastically reduce output file sizes for long simulations. The binning into final Q-space or detector space is performed flexibly afterwards, allowing for dynamic slicing and filtering without needing to rerun the BornAgain simulation.

2. Coordinate Systems & Transformations
---------------------------------------

The coordinate systems have been deeply refactored to maintain mathematical strictness:

* **BornAgain System**: The core simulation step now strictly uses the standard BornAgain coordinate system for the sample definition and particle propagation. 
* **NeXus System**: For the propagation to the detector surface and pixel hit detection, the coordinates are transformed back into the NeXus coordinate system.
* **Sample Orientation & Plotting**: When defining sample orientations and generating plots (via `mg_plot`), a standard visual convention is used where the **y-axis is horizontal** and the **z-axis is vertical**.
* **Limitation**: Currently, the calculation assumes a vertical detector surface (in the NeXus coordinate system). Highly inclined or curved detectors may not be fully supported without additional coordinate projections.

3. Core Tools and Modules
-------------------------

With the modularization of the codebase, specific tasks are handled by dedicated scripts:

* **nexus_reader.py**: The legacy, hardcoded `read_d22.py` has been entirely replaced by a general `nexus_reader.py`. This reader utilizes the `Instrument` object to seamlessly convert pixel data from experimental NeXus files into Q-space data used for direct comparison with simulations. *(Note: The internal HDF5 path to the pixel dataset inside the NeXus files is currently hardcoded, but a CLI option to provide custom data paths is planned for a future release).*
* **fit.py (`mg_fit`)**: Automates the parameter fitting process. It employs optimizers like Nelder-Mead or Differential Evolution to minimize objective loss metrics (e.g. `reduced_chi2`) between the BornAgain simulated data and the NeXus measurement data.
* **beam_centre_correction.py**: A specialized utility designed to handle subtle beam offsets and alignment corrections required to accurately project simulated scattering onto the rigid NeXus pixel grid.

4. Testing
----------

To ensure the physical and mathematical integrity of the coordinate transformations and scattering simulations, the codebase now includes a dedicated suite of regression tests (e.g., `test_d22_regression.py` and `test_d22_microgel_regression.py`). These tests run full simulations and assert that the computed `reduced_chi2` against reference datasets remains mathematically stable.

5. BornAgain Compatibility
--------------------------

The core `mcstas_gisans` framework is compatible with **BornAgain versions 21.2 through 24**. 
*(Note: Individual custom sample models defined in the `bornagain_samples/` directory may require minor syntax adjustments depending on the specific BornAgain version being used, due to deprecations in BornAgain's Python API across these versions).*
