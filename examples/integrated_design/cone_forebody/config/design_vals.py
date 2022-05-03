"""
Define design values for integrated forebody-inlet configuration.
The following configuration is based on the 
'Airbreather Space Access Performance Parameter Optimised SPARTAN' from:

Preller, Dawid (2018). Multidisciplinary design and optimisation of a pitch 
trimmed hypersonic airbreathing accelerating vehicle. PhD Thesis, School of 
Mechanical and Mining Engineering, The University of Queensland.
https://doi.org/10.14264/uql.2018.437

Author: Reece Otto 07/04/2022
"""
from csgen.atmosphere import atmos_interp
import json
from math import pi

# calculate free-stream values
gamma = 1.4
M0 = 10
q0 = 50E3
p0 = 2*q0 / (gamma*M0*M0)
T0 = atmos_interp(p0, 'Pressure', 'Temperature')

# create dictionary
design_vals = {
    # free-stream flow parameters
    'free_stream': {
        'rat_spec_heats': gamma, # ratio of specific heats
        'mach_no': M0,           # Mach number
        'dynam_press': q0,       # dynamic pressure [Pa]
        'press': p0,             # static pressure [Pa]
        'temp': T0,              # temperature [K]
    },
    # forebody design values
    'forebody': {
        'cone_angle': 4.0*pi/180, # half-angle of conical forebody [rad]
        'length': 15.0,           # length of forebody
    },
    # inlet design values
    'inlet': {
        'exit_press': 50E3,        # exit pressure [Pa]
        'trunc_angle': 8*pi/180,   # truncation angle of Busemann diffuser [rad]
        'smile_angle': 180*pi/180, # total smile angle of engine modules [rad]
        'no_modules': 4,           # number of engine modules
        'blend_factor': 2.5        # inlet blending parameter
    }
}

# export dictionary as json file
with open('design_vals.json', 'w') as f:
    json.dump(design_vals, f, ensure_ascii=False, indent=2)





"""

#------------------------------------------------------------------------------#
#                                 Diffuser                                     #
#------------------------------------------------------------------------------#
# define parameters for truncated Busemann diffuser
diffuser_vals = {
    # parameters to generate truncated Busemann diffuser
    'buse_vals': {
        'rat_spec_heats': gamma, # ratio of specific heats
        'p_exit': 50E3,          # exit pressure [Pa]
        'trunc_angle': 8*pi/180, # truncation angle [rad]
        'beta2_guess': 0.2088,   # guess for terminating shock angle [rad]
        'M2_guess': 5.912        # guess for Mach no at station 2
    },
    # Taylor-Maccoll and streamline integration settings 
    'buse_settings': {
        'theta_step': 0.05*pi/180, # theta step size [rad]
        'max_steps': 10000,        # maximum number of integration steps
        'print_freq': 500,         # printing frequency of integration info
        'interp_sing': True,       # interpolate for Taylor-Macoll singularity
        'verbosity': 1             # verbosity level
    }
}

#------------------------------------------------------------------------------#
#                                   Inlet                                      #
#------------------------------------------------------------------------------#
# define parameters for shape-transitioning inlet
no_modules = 4
n_phi_inlet = 2*forebody_vals['surf_vals']['n_phi']/no_modules
inlet_vals = {
    # inlet design values
    '': {
        
        'n_phi': n_phi_inlet,
        'n_z': 100
    }
    # settings for streamline tracing
    'stream_settings': {
        'p_rat_shock': 2.0,  # pressure ratio for shock detection
        'time_step': 1.0E-5, # integration time step [s]
        'max_step': 100000   # maximum number of integration steps
    }
}

#------------------------------------------------------------------------------#
#                                 File Export                                  #
#------------------------------------------------------------------------------#
"""