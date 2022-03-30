from csgen.waverider import conical_stream
from math import pi

# free-stream flow parameters
M0 = 10     # free-stream Mach number
q0 = 50E3   # dynamic pressure (Pa)
gamma = 1.4 # ratio of specific heats

# waverider design parameters
dtheta = 0.01*pi/180 # integration step size (rad)
theta_c = 7*pi/180       # angle of conical shock (rad)
z_base = 5              # z plane where base of waverider exists
n_phi = 51          # number of streamlines to be traced around base shape
n_z = 51                # number of points evaluated in z direction

# settings for conical flow field solver
beta_guess = 20*pi/180
n_steps = 10000
max_iter = 10000
tol = 1E-6
interp_sing = True
verbosity = 1
print_freq = 10

design_param = {
	'M0': M0,
	'q0': q0,
	'gamma': gamma,
	'theta_c': theta_c,
	'z_base': z_base,
	'n_phi': n_phi,
	'n_z': n_z
}

settings = {
	'dtheta': dtheta,
	'beta_guess': beta_guess,
	'n_steps': n_steps,
	'max_iter': max_iter,
	'tol': tol,
	'interp_sing': interp_sing,
	'verbosity': verbosity,
	'print_freq': print_freq
}


print(conical_stream(design_param, settings))