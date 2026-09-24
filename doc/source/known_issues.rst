============
Known Issues
============

- With a low outgoing direction number (``--outgoing_directions``, ``-n`` < 20–30) the specular peak can create a huge artefact. This is an inherent BornAgain issue, not a mcstas_gisans issue.

- The detection process is not simulated: the detection efficiency is 100%, without scattering in the detector, dead time or attenuators. Only the position resolution (``resolution`` in the instrument definition) and the pixelation are applied.

- The detector is assumed to be flat and perpendicular to the beam axis (see :doc:`technical_details`).

- Simulation with polarisation (``--use_polarization`` and the analyser options of ``mg_run``) is implemented, but not validated or covered by tests yet.

- Fitting of TOF data (``mg_fit`` with a TOF instrument) is not implemented yet.
