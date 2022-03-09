#!/bin/bash
rm -r solutions/
python3 baseline_gen.py
cd solutions/trunc_buse_0/
. run-trunc_buse_0.sh
cd ../..
python3 optimizer.py
