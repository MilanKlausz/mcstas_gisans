#!/bin/bash

export CFLAGS=" -DFUNNEL"

for IN_FILE in GISANS_events GISANS_events_back GISANS_test GISANS_tt
do
  if [ "$IN_FILE.instr" -nt "$IN_FILE.out" ] || [ ! -f "$IN_FILE.out" ]; then
    rm -f "$IN_FILE.out" "$IN_FILE.c"
    echo "Generating executable for $IN_FILE:"
    mcstas -o "$IN_FILE.c" "$IN_FILE.instr"
    cc -O2 -o "$IN_FILE.out" "$IN_FILE.c" -lm \
       -Wno-unused-result -Wno-format-truncation -Wno-format-overflow -Wno-format-security
  fi
done

