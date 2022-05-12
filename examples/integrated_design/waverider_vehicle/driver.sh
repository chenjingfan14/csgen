#!/bin/bash
python3 design_vals.py
python3 waverider.py
python3 trunc_buse.py
source run-diffuser.sh
python3 inlet_gen.py
