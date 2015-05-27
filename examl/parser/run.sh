#!/bin/bash
rm -f RAxML_info.P37 P37.binary
./parse-examl -s dna.phy.dat -m POMO16 -p pomoMap -n P37
