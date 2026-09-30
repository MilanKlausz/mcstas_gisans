============
Known Issues
============

- With ``--specular include_specular``, BornAgain adds the whole specular reflection to the bin of the outgoing-direction grid that contains the specular direction, and the ray of that bin goes to the (randomly shifted) bin centre. The simulated specular spot is therefore smeared uniformly over one grid bin: with a coarse grid it is much wider than the measured one (D22 paper data: an RMS width of 1.1 pixels with bins of 3.5 pixels, instead of 0.57 pixel; measured: 0.62), and with a very low number of directions this appears as a large artefact. ``--specular specular_simulation`` sends the specular reflection as one extra ray per neutron in the exact mirror direction, with the reflectivity of the sample, so its position and width do not depend on the grid (0.56 pixel with both grids, with the same intensity). A fit region that excludes the specular spot with a margin of more than half a grid bin is not affected.

- The detection process is not simulated: the detection efficiency is 100%, without scattering in the detector, dead time or attenuators. Only the position resolution (``resolution`` in the instrument definition) and the pixelation are applied.

- The detector is assumed to be flat and perpendicular to the beam axis (see :doc:`technical_details`).

- Simulation with polarisation (``--use_polarization`` and the analyser options of ``mg_run``) is implemented, but not validated or covered by tests yet.

- Fitting of TOF data (``mg_fit`` with a TOF instrument) is not implemented yet.
