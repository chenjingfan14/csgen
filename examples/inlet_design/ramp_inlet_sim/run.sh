#!/bin/bash
prep-gas ideal-air.inp ideal-air-gas-model.lua
puffin-prep --job=ramp_inlet
puffin --job=ramp_inlet
puffin-post --job=ramp_inlet --output=vtk
puffin-post --job=ramp_inlet --output=stream --cell-index=$ --stream-index=0
