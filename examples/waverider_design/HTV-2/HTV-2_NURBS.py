from nurbskit.path import BSpline
from nurbskit.utils import auto_knot_vector
from nurbskit.visualisation import path_plot_2D
import matplotlib.pyplot as plt
from math import pi

# feature-based parameters
L1 = 3.0; L2 = 1.0; W = 1.0; H1 = 0.42; H2 = 0.18; r = 0.01 
theta1 = 16*pi/180; theta2 = 5.7*pi/180
L = L1 + L2

# control point parameters
x0 = 0.0; y0 = H1
x1 = W/5; y1 = H1
x2 = W/2; y2 = H1/2
x3 = 3*W/4; y3 = 5*r
x4 = W - r; y4 = r
x5 = W; y5 = r
x6 = W; y6 = -r
x7 = W - r; y7 = -r
x8 = 3*W/4; y8 = -3*H2/4
x9 = W/5; y9 = -H2
x10 = 0.0; y10 = -H2

# control net at base
P_base = [[x0,y0,L],
            [x1,y1,L],
            [x2,y2,L],
            [x3,y3,L],
            [x4,y4,L],
            [x5,y5,L],
            [x6,y6,L],
            [x7,y7,L],
            [x8,y8,L],
            [x9,y9,L],
            [x10,y10,L]]

p = 3
n_phi = 100
U = auto_knot_vector(len(P_base), p)
base = BSpline(P=P_base, p=p, U=U)
#base_coords = base.discretize(n_points=n_phi)

# plot cross-section at z=z_base
plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
#matplotlib.rc('text', usetex=True)
ax = path_plot_2D(base, show_control_net=False)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
plt.savefig('base.svg', bbox_inches='tight')