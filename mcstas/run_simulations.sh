#!/bin/bash

#rm -rf GISANS_test
#./GISANS_test.out -n 1e6 -d GISANS_test

#rm -rf GISANS_events
#./GISANS_events.out -n 1e4 -d GISANS_events

# run the "python events2BA" to create the scattered events file
rm -rf GISANS_events_scattered
./GISANS_events_back.out -n 1e7 -d GISANS_events_scattered