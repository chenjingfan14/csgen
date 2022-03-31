"""
Generating a conical flow field characterised by M0 and beta.

Author: Reece Otto 14/12/2021
"""
from csgen.conical_field import conical_M0_thetac
from math import pi

# design values for conical field and streamline
design_vals = {
    'M0': 10,           # fre-stream Mach number
    'thetac': 7*pi/180, # angle of conical shock [rad]
    'gamma': 1.4,       # ratio of specific heats
    'L_field': 10,      # length of conical field when tracing streamline
    'r0': 1             # initial radius of streamline
}

# integration settings
settings = {
    'beta_guess': 20*pi/180, # initial guess for shock angle [rad]
    'tol': 1.0E-6,           # tolerance for thetac convergence [rad]
    'dtheta': 0.01*pi/180,   # integration step size for theta [rad]
    'max_steps': 10000,      # maximum number of integration steps
    'print_freq': 20,        # printing frequency of integration info
    'verbosity': 1           # verbosity level
}

# generate flow field and streamline
field = conical_M0_thetac(design_vals, settings)
Stream = field.Streamline(design_vals, settings)

# save streamline as CSV
Stream.save_to_csv(file_name='stream')

# generate plot
field.plot(Stream)

# generate surfaces
field.cone_surface(design_vals['L_field'])
field.shock_surface(design_vals['L_field'])