"""
Generating a truncated Busemann contour.

Author: Reece Otto 11/02/2022
"""
from csgen.busemann import busemann_M1_p3p1
from math import pi
import matplotlib.pyplot as plt
import numpy as np

# Busemann field design parameters
design_vals = {
    'M1': 7.95,   # Mach number at station 2
    'p3_p1': 19,  # angle of terminating shock [rad]
    'gamma': 1.4, # ratio of specific heats
    'r0': 1       # initial radius
}

# integration settings for Taylor-Maccoll equations
settings = {
    'dtheta': 0.05*pi/180, # theta step size [rad]
    'beta2_guess': 0.2324, # initial guess for beta2 [rad]
    'M2_guess': 5.356,     # initial guess for M2
    'max_steps': 10000,    # maximum number of integration steps
    'print_freq': 500,     # printing frequency of integration info
    'interp_sing': True,   # interpolate for Taylor-Macoll singularity
    'verbosity': 1         # verbosity level
}

# generate Busemann field
field = busemann_M1_p3p1(design_vals, settings)

# truncate contour
field.Streamline = field.Streamline.truncate(trunc_angle=5*pi/180)

# scale so that entrance height = 1
scale_factor = 1/field.Streamline.ys[0]
field.Streamline = field.Streamline.scale(scale_factor)

# translate so that contour starts at x = 0
z_shift = abs(field.Streamline.zs[0])
field.Streamline = field.Streamline.translate(z_shift=z_shift)

# save plot of field
field.plot(file_name='trunc_buse', show_mach_wave=False, show_exit_shock=False)

# save contour as CSV file
field.Streamline.save_to_csv()