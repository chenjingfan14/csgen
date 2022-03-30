"""
Generating a Busemann flow field by using M2 and beta as design parameters.

Author: Reece Otto 11/02/2022
"""
from csgen.busemann import busemann_M2_beta2
from math import pi
import matplotlib.pyplot as plt
import numpy as np

# Busemann field design parameters
design_vals = {
    'M2': 3,            # Mach number at station 2
    'beta2': 30*pi/180, # angle of terminating shock [rad]
    'gamma': 1.4,       # ratio of specific heats
    'r0': 1             # initial radius
}

# integration settings for Taylor-Maccoll equations
settings = {
    'dtheta': 0.1*pi/180, # theta step size [rad]
    'max_steps': 10000,   # maximum number of integration steps
    'print_freq': 100,     # printing frequency of integration info
    'interp_sing': True,  # interpolate for Taylor-Macoll singularity
    'verbosity': 1        # verbosity level
}

# generate Busemann field
field = busemann_M2_beta2(design_vals, settings)

# save plot of field
field.plot()

# save contour as CSV file
field.Streamline.save_to_csv()

# save surfaces as VTK files
field.buse_surf()
field.mach_cone_surface()
field.term_shock_surface()