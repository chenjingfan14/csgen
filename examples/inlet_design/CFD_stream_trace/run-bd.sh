#!/bin/bash
prep-gas ideal-air.inp ideal-air-gas-model.lua
puffin-prep --job=trunc-bd
puffin --job=trunc-bd
puffin-post --job=trunc-bd --output=vtk
puffin-post --job=trunc-bd --output=stream --cell-index=$ --stream-index=0

