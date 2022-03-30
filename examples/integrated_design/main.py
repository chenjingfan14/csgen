"""
Main driver script for integrated forebody-inlet design.

Author: Reece Otto 29/03/2022
"""
# free-stream flow parameters
M0 = 10     # free-stream Mach number
q0 = 50E3   # dynamic pressure (Pa)
gamma = 1.4 # ratio of specific heats

# waverider design parameters
dtheta_wr = 0.01*pi/180 # integration step size (rad)
thetac = 7*pi/180       # angle of conical shock (rad)
z_base = 5              # z plane where base of waverider exists
n_streams = 51          # number of streamlines to be traced around base shape
n_z = 51                # number of points evaluated in z direction

# inlet design parameters
p3 = 50E3                 # desired exit pressure [Pa]
#p3_p1 = p3/p1             # desired compression ratio
dtheta_buse = 0.09*pi/180 # theta step size [rad]

# 