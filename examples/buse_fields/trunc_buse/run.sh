#!/bin/bash
prep-gas ideal-air.inp ideal-air-gas-model.lua
puffin-prep --job=trunc_buse
puffin --job=trunc_buse
puffin-post --job=trunc_buse --output=vtk
puffin-post --job=trunc_buse --output=stream --cell-index=$ --stream-index=0