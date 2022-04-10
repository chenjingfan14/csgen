"""
Define design values and settings for waverider, diffuser and inlet routines.

Author: Reece Otto 07/04/2022
"""
from csgen.atmosphere import atmos_interp
import json
from math import pi
import os

#------------------------------------------------------------------------------#
#                                 Waverider                                    #
#------------------------------------------------------------------------------#
# define free-stream values
gamma = 1.4 
M0 = 10     
q0 = 50E3   

# calculate remaining free-stream properties from US standard atmopshere 1976
p0 = 2*q0 / (gamma*M0*M0)                        
T0 = atmos_interp(p0, 'Pressure', 'Temperature') 

free_stream = {
    'rat_spec_heats': gamma, # ratio of specific heats
    'mach_no': M0,           # Mach number
    'dynam_press': q0,       # dynamic pressure [Pa]
    'press': p0,             # static pressure [Pa]
    'temp': T0,              # temperature [K]
}

# define parameters for conical waverider
waverider_vals = {
    # parameters to generate conical flow field
    'cone_vals': {
        'mach_no': M0,              # free-stream Mach number
        'shock_angle': 10.5*pi/180, # angle of conical shockwave [rad]
        'rat_spec_heats': gamma,    # ratio of specific heats
        'field_len': 10.0,          # length of conical flow field
        'init_radius': 1.0          # initial radius of streamline
    },
    # Taylor-Maccoll and streamline integration settings 
    'cone_settings': {
        'theta_step': 0.01*pi/180, # integration step size [rad]
        'max_steps': 10000,        # maximum number of integration steps
        'print_freq': 20,          # printing frequency of integration info
        'verbosity': 1             # verbosity level
    },
    # waverider surface parameters
    'surf_vals': {
        'z_base': 5.0,   # z location of waverider base
        'n_phi': 51,     # number of phi points
        'n_z': 51,       # number of z points
        'tol': 1.0E-4,   # tolerance for streamline-finding algorithm
        'save_VTK': True # save surface as VTK file
    }
}

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
n_phi_inlet = 2*waverider_vals['surf_vals']['n_phi']
inlet_vals = {
    # settings for streamline tracing
    'stream_settings': {
        'p_rat_shock': 2.0,  # pressure ratio for shock detection
        'time_step': 1.0E-5, # integration time step [s]
        'max_step': 100000   # maximum number of integration steps
    },
    # inlet surface parameters
    'surf_vals': {
        'n_phi': n_phi_inlet, # number of phi points
        'n_z': 100,           # number of z points
        'blend_factor': 2.5   # inlet blending factor
    }
}

#------------------------------------------------------------------------------#
#                                 File Export                                  #
#------------------------------------------------------------------------------#
# save waverider dictionaries
main_dir = os.getcwd()
waverider_dir = main_dir + '/waverider'
os.chdir(waverider_dir)
with open('free_stream.json', 'w') as f:
    json.dump(free_stream, f, ensure_ascii=False, indent=2)
with open('waverider_vals.json', 'w') as f:
    json.dump(waverider_vals, f, ensure_ascii=False, indent=2)

# save diffuser dictionary
diffuser_dir = main_dir + '/diffuser'
os.chdir(diffuser_dir)
with open('diffuser_vals.json', 'w') as f:
    json.dump(diffuser_vals, f, ensure_ascii=False, indent=2)

# save inlet dictionary
inlet_dir = main_dir + '/inlet'
os.chdir(inlet_dir)
with open('inlet_vals.json', 'w') as f:
    json.dump(inlet_vals, f, ensure_ascii=False, indent=2)
os.chdir(main_dir)