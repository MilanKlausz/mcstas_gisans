# Test data

- `d22_1e8/test_events.mcpl.gz`: a small McStas output of the D22 beam of the paper example (10^8 source neutrons,
  1726 particles at the sample, sum of the weights 9639.83; intensity factor 120538 / 60 / 9639.83 = 0.2084 for the
  direct beam measurement `data/paper/d22_measurement/073162.nxs`). The same beam as
  `data/paper/mcstas_output/d22_1e9` with 10x fewer neutrons, kept only so that the tests run fast; the examples use
  `d22_1e9`.
