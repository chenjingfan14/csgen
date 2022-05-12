#!/bin/bash
cd diffuser
prep-gas ideal-air.inp ideal-air-gas-model.lua
puffin-prep --job=diffuser
puffin --job=diffuser
puffin-post --job=diffuser --output=vtk
puffin-post --job=diffuser --output=stream --cell-index=$ --stream-index=0
cd ..
