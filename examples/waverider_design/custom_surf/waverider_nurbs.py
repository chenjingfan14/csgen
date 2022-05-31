from nurbskit.path import Ellipse, BSpline
from nurbskit.utils import auto_knot_vector
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import pi, sin, cos, tan

r_cone = 0.5
r_shock = 1.0
z_base = 3.5

# define waverider design parameters
n_phi = 101

# create cone cross-section at z=z_base
r_cone = z_base * tan(6*pi/180)
cone_base = Ellipse(r_cone, r_cone)
cone_base_coords = cone_base.discretize(n_points=101)

# create shock cross-section at z=z_base
r_shock = z_base * tan(12*pi/180)
shock_base = Ellipse(r_shock, r_shock)
shock_base_coords = shock_base.discretize(n_points=101)

# create base contour for bottom surface of waverider
phi_int = -40*pi/180
x_shock = r_shock*cos(phi_int)
y_shock = r_shock*sin(phi_int)
ang_shock = pi/2 + phi_int
rd_param = 2*x_shock/100
del_x_rd = rd_param*cos(ang_shock)
del_y_rd = rd_param*sin(ang_shock)
max_y_bot = -1.01*r_cone
max_y_top = -0.7*r_cone

P_bot = [[0, max_y_bot],
         [x_shock/4, max_y_bot],
         [x_shock/2, max_y_bot*1.02],
         [x_shock - 2*del_x_rd, y_shock],
         [x_shock - del_x_rd, y_shock - del_y_rd],
         [x_shock, y_shock]]
p_bot = 3
U_bot = auto_knot_vector(len(P_bot), p_bot)
wr_bot = BSpline(P=P_bot, p=p_bot, U=U_bot)
wr_bot_coords = wr_bot.discretize(n_points=n_phi)

# create base contour for top surface of waverider
P_top = [[0, max_y_top],
         [x_shock/4, max_y_top],
         [x_shock/2, max_y_top*1.02],
         [x_shock - del_x_rd, y_shock + 2*del_y_rd],
         [x_shock + del_x_rd, y_shock + del_y_rd],
         [x_shock, y_shock]]
p_top = 3
U_top = auto_knot_vector(len(P_top), p_top)
wr_top = BSpline(P=P_top, p=p_top, U=U_top)
wr_top_coords = wr_top.discretize(n_points=n_phi)

# plot cross-section at z=z_base
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
matplotlib.rc('text', usetex=True)
ax = plt.axes()
ax.plot(cone_base_coords[:,0], cone_base_coords[:,1], 'k', label='Cone Surface')
ax.plot(shock_base_coords[:,0], shock_base_coords[:,1], 'r', 
    label='Shock Surface')
ax.plot(wr_bot_coords[:,0], wr_bot_coords[:,1], 'b', label='Base Contour')
ax.plot(wr_top_coords[:,0], wr_top_coords[:,1], 'b')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.axis('square')
plt.grid()
plt.legend(bbox_to_anchor=[1.0, 1.0])
plt.savefig('waverider_base.svg', bbox_inches='tight')



