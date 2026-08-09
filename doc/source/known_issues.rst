============
Known Issues
============

- With a low outgoing direction number (``--outgoing_direction_number``, ``-n`` < 20–30) the specular peak can create a huge artefact. This is an inherent BornAgain issue, not a mcstas_gisans issue.

- Detection process is not simulated:

  - Detection efficiency is 100%, without any scattering effects.

- Simulation with polarisation is not supported, although both McStas and MCPL are capable of it, so this could be changed on demand. (Work in progress)
