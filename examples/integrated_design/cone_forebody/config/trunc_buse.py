"""
Generate truncated Busemann contour based on exit flow conditions from forebody.

Author: Reece Otto 29/03/2022
"""
from csgen.busemann import busemann_M1_p3p1
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# read inflow parameters from json file
main_dir = os.getcwd()
working_dir = main_dir + '/diffuser'
os.chdir(working_dir)
f = open('diffuser_vals.json')
diffuser_vals = json.load(f)
f.close()
buse_vals = diffuser_vals['buse_vals']
buse_settings = diffuser_vals['buse_settings']
inflow_data = diffuser_vals['inflow_data']

# generate Busemann field
buse_vals['M1'] = inflow_data['mach_no']
buse_vals['p3_p1'] = buse_vals['p_exit'] / inflow_data['press']
field = busemann_M1_p3p1(buse_vals, buse_settings)

# truncate contour
field.Streamline = field.Streamline.truncate(
    trunc_angle=buse_vals['trunc_angle'])

# scale to accommodate capture shape
scaler = 1.6
field.Streamline = field.Streamline.scale(scaler, scaler, scaler)

# translate so that contour starts at x = 0
z_shift = abs(field.Streamline.zs[0])
field.Streamline = field.Streamline.translate(z_shift=z_shift)

# save plot of field
field.plot(file_name='trunc_buse', show_mach_wave=False, show_exit_shock=False)

# save csv of streamline
field.Streamline.save_to_csv('trunc_buse')

# TODO: generate puffin scripts