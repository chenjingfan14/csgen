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
f = open('inflow.json')
inflow_data = json.load(f)
f.close()

# Busemann field design parameters
design_vals = {
    'M1': inflow_data['M'],           # Mach number at station 2
    'p3_p1': 50E3 / inflow_data['p'], # angle of terminating shock [rad]
    'gamma': 1.4,                     # ratio of specific heats
    'r0': 1                           # initial radius
}

# integration settings for Taylor-Maccoll equations
settings = {
    'dtheta': 0.05*pi/180, # theta step size [rad]
    'beta2_guess': 0.2088, # initial guess for beta2 [rad]
    'M2_guess': 5.912,     # initial guess for M2
    'max_steps': 10000,    # maximum number of integration steps
    'print_freq': 500,     # printing frequency of integration info
    'interp_sing': True,   # interpolate for Taylor-Macoll singularity
    'verbosity': 1         # verbosity level
}

# generate Busemann field
field = busemann_M1_p3p1(design_vals, settings)

# truncate contour
field.Streamline = field.Streamline.truncate(trunc_angle=8*pi/180)

# scale so to accommodate capture shape
scaler = 1.6
field.Streamline = field.Streamline.scale(scaler, scaler, scaler)

# translate so that contour starts at x = 0
z_shift = abs(field.Streamline.zs[0])
field.Streamline = field.Streamline.translate(z_shift=z_shift)

# save plot of field
field.plot(file_name='trunc_buse', show_mach_wave=False, show_exit_shock=False)

# save csv of streamline
field.Streamline.save_to_csv('trunc_buse')