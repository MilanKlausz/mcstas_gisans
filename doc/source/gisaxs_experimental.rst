=================================
GISAXS Simulation (Experimental)
=================================

As a yet untested experimental option, simulation with photons is also supported by our framework, making it possible to use it for GISAXS simulation by the connection of an X-ray simulation code (e.g., McXtrace) with BornAgain.

Given that the BornAgain DWBA calculation applies to photon–sample interaction just as well as for the neutron–sample interaction, the main differences are:

- Definition of the BornAgain sample (different SLD for photons)
- Calculation of the wavelength from the parameters in the MCPL file

The framework reads the particle definition code (PDG=2112 for neutron, PDG=22 for photon) that is stored in the MCPL file, and handles the wavelength calculation accordingly.

Therefore, there is no difference in the simulation of photons with the framework, except for the need to provide BornAgain sample models suitable for the calculation of photon–sample interactions.
