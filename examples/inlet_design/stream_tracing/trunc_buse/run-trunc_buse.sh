#!/bin/bash
prep-gas ideal-air.inp ideal-air-gas-model.lua
puffin-prep --job=trunc-buse
puffin --job=trunc-buse
puffin-post --job=trunc-buse --output=vtk
puffin-post --job=trunc-buse --output=stream --cell-index=$ --stream-index=0

