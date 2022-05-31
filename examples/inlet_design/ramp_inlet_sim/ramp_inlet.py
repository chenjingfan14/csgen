from math import pi, tan

# defining inflow conditions
q1 = 50E3               # dynamic pressure [Pa]
M1 = 8.0                # Mach number
gamma = 1.4             # ratio of specific heats
p1 = 2*q1/(gamma*M1**2) # static pressure [Pa]
T1 = 300                # temperature [K]

# constructing gas model
init_gas_model('ideal-air-gas-model.lua')
gas1 = GasState(config.gmodel)
gas1.p = p1
gas1.T = T1
gas1.update_thermo_from_pT()
gas1.update_sound_speed()
V1 = M1*gas1.a

# design variables
theta_1 = 7.0*pi/180; theta_2 = 4.5*pi/180; theta_3 = 3.0*pi/180
L1 = 0.4; L2 = 0.4; L3 = 0.4; L_exit = 0.2; r_c = 0.04

# define points along top of domain
x_thrt = L1 + L2 + L3
x_max = x_thrt + L_exit
A = [x_max, r_c]
B = [x_thrt, r_c]
C = [L1 + L2, r_c + L3*tan(theta_1 + theta_2 + theta_3)]
D = [L1, C[1] + L2*tan(theta_1 + theta_2)]
E = [0.0, D[1] + L1*tan(theta_1)]

# define line function for ramps
def line(x, point_1, point_2):
	x1 = point_1[0]; y1 = point_1[1]
	x2 = point_2[0]; y2 = point_2[1]
	return (y2 - y1)/(x2 - x1) * (x - x1) + y1 

# define boundaries
def upper_y(x):
	if x <= L1:
		return line(x, E, D)
	if L1 < x and x <= L1 + L2:
		return line(x, D, C)
	if L1 + L2 < x and x <= x_thrt:
		return line(x, C, B)
	if x < x_thrt:
		return line(x, B, A)

def lower_y(x): return 0.0
def lower_bc(x): return 0
def upper_bc(x): return 0

# solver settings
config.title = 'Three-Ramp Inlet'
config.max_step_relax = 40
config.max_x = x_max
config.dx = config.max_x/2000.0

st1 = StreamTube(gas=gas1, velx=V1, vely=0.0,
                 y0=lower_y, y1=upper_y,
                 bc0=lower_bc, bc1=upper_bc,
                 ncells=75)