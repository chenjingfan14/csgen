"""

Author: Reece Otto 14/12/2021
"""
from math import pi, tan, sqrt, atan2, cos, sin, atan
from csgen.conical_field_generator import ConicalSolMach, Conical_M0_thetac
from csgen.atmosphere import AtmosInterpolate
import matplotlib.pyplot as plt
import numpy as np
from nurbskit.path import NURBSEllipse, BSpline
from nurbskit.surface import BSplineSurface
from nurbskit.utils import auto_knot_vector
from nurbskit.spline_fitting import global_curve_interp, GlobalSurfInterp
import pyvista as pv
from scipy import optimize
from nurbskit.point_inversion import PtInvPTPath, PointCoincidence

# free-stream flow parameters
M0 = 10                       # free-stream Mach number
q0 = 50E3                     # dynamic pressure (Pa)
gamma = 1.4                   # ratio of specific heats
p0 = 2 * q0 / (gamma * M0*M0) # static pressure (Pa)

# calculate remaining free-stream properties from US standard atmopshere 1976
T0 = AtmosInterpolate(p0, 'Pressure', 'Temperature')
a0 = AtmosInterpolate(p0, 'Pressure', 'Sonic Speed')
V0 = M0 * a0

# geometric properties of conical flow field
L = 10.0                # length of flow field
thetac = 6 * pi / 180  # imaginary cone half-angle (rad)
dtheta = 0.01*pi / 180   # integration step size (rad)
beta = 15 * pi / 180

# generate flow field
#field = ConicalSolMach(M0, beta, gamma, dtheta)
field = Conical_M0_thetac(M0, thetac, gamma, dtheta)

# calculate coordinates of flow field, cone surface and shock
field_coords = field.xyz_coords_mirror_x(L)

# parametrise flow field contour with B-Spline curve
Q = [None] * len(field_coords[0])
for i in range(len(field_coords[0])):
    Q[i] = [field_coords[0][i], field_coords[1][i], field_coords[2][i]]
p = 3
U, P = global_curve_interp(Q, p)
field_spline = BSpline(U=U, P=P, p=p)
us = np.linspace(0, 1, 100)
field_spline_coords = np.array([field_spline(u) for u in us])

# plot shock and imaginary cone
cone_max_y = -max(field_coords[2]) * tan(field.thetac)
cone_coords = [[0, 0], 
               [cone_max_y, 0], 
               [max(field_coords[2]), 0]]

shock_max_y = -max(field_coords[2]) * tan(field.beta)
shock_coords = [[0, 0], 
               [shock_max_y, 0], 
               [max(field_coords[2]), 0]]

axis_coords = [[0, max(field_coords[2])], [0, 0]]

# plot flow field
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({"text.usetex": True, "font.size": 16})
plt.plot(cone_coords[2], cone_coords[1], 'k', label='Cone Surface')
plt.plot(shock_coords[2], shock_coords[1], 'r', label='Shock Wave')
plt.plot(field_coords[2], field_coords[1], 'b', label='Streamline')
plt.plot(field_spline_coords[:,2], field_spline_coords[:,1], 'g', label='Spline Streamline')
#plt.plot(axis_coords[0], axis_coords[1], 'k-.', label='Axis of Symmetry')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('conical_field.svg', bbox_inches='tight')
plt.show()

# create cone cross-section at z=L_i
L_i = max(field_coords[2])
print('L_i = ', L_i)
N_streams = 10
us = np.linspace(0, 1, N_streams)
r_cone = L_i * tan(field.thetac)
cone_shape = NURBSEllipse(r_cone, r_cone)
cone_coords = np.array([cone_shape(u) for u in us])
xs_cone = cone_coords[:,0]
ys_cone = cone_coords[:,1]

# create shock cross-section at z=L_i
r_shock = L_i * tan(field.beta)
shock_shape = NURBSEllipse(r_shock, r_shock)
shock_coords = np.array([shock_shape(u) for u in us])
xs_shock = shock_coords[:,0]
ys_shock = shock_coords[:,1]

# create baseline contour for bottom surface of waverider
max_y = 1.05 * min(ys_cone)
phi_intercept = 50 * pi / 180
max_x = r_shock * cos(phi_intercept)
min_y = -r_shock * sin(phi_intercept)
P = [[-max_x, min_y], [-max_x/2, (max_y + min_y)/1.9], [-max_x/3, max_y], 
     [0, max_y], 
     [max_x/3, max_y], [max_x/2,(max_y + min_y)/1.9], [max_x, min_y]]
p = 3
U = auto_knot_vector(len(P), p)
fb_bot = BSpline(P=P, p=p, U=U)
fb_coords = np.array([fb_bot(u) for u in us])
xs_fb = fb_coords[:,0]
ys_fb = fb_coords[:,1]

# plot cross-section at z=L_i
plt.figure(figsize=(16, 9))
plt.rcParams.update({"text.usetex": True, "font.size": 16})
plt.plot(xs_fb, ys_fb, 'b', label='Forebody')
plt.plot(xs_shock, ys_shock, 'r', label='Conical Shock')
plt.plot(xs_cone, ys_cone, 'k', label='Base Cone')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('forebody_inlet_interface_given.svg', bbox_inches='tight')
plt.show()

def stream_transform(scale, point, field_coords):
    """
    Rotates and scales a streamline to intersect a given point.
    """
    # calculate phi angle required for streamline
    phi = atan2(point[1], point[0])

    # rotate streamline to angle phi and scale
    Q = [None] * len(field_coords[0])
    for i in range(len(field_coords[0])):
        r = sqrt(field_coords[0][i]**2 + field_coords[1][i]**2 + \
                 field_coords[2][i])
        theta = atan(sqrt(field_coords[0][i]**2 + field_coords[1][i]**2) / \
                          field_coords[2][i])
        Q[i] = [float(scale * r * sin(theta) * cos(phi)), 
                float(scale * r * sin(theta) * sin(phi)), 
                float(scale * r * cos(theta))]
    return Q

def stream_param(Q, point):
    # parametrise flow field contour with B-Spline curve
    p = 3
    U, P = global_curve_interp(Q, p)
    field_spline = BSpline(U=U, P=P, p=p)
    # run point inversion routine to find point on curve closest
    # to given point
    u_cand = PtInvPTPath(field_spline, point, tol1=1E-9, tol2=1E-9, 
                         initGuess='BruteForce', N=30, maxIt=100)
    
    print(f'C(u_cand) = {field_spline(u_cand)}')
    print(f'|C(u_cand) - P| = {PointCoincidence(field_spline, point, u_cand):.4} \n')
    return field_spline, u_cand

def stream_find(scale, point, field_coords):
    # transform the field coordinates
    Q = stream_transform(scale, point, field_coords)

    # parameterise the transformed field coordinates
    field_spline, u_cand = stream_param(Q, point)

    # return L3 norm between points as the residual
    return PointCoincidence(field_spline, point, u_cand)
    
Nz = 10
streams = np.zeros((len(xs_fb), Nz, 3))
for i in range(len(xs_fb)):
    # find the scale of the streamline that runs through each 
    # point along the waverider base contour
    point = [xs_fb[i], ys_fb[i], L_i]
    stream_sol = optimize.root(stream_find, 1.1, args=(point, field_coords), tol=1E-4)
    if stream_sol.success == False:
        raise Exception('Root finder failed.')
    scale = stream_sol.x
    
    # transform the field coordinates
    Q = stream_transform(scale, point, field_coords)

    # parameterise the field coordinates and evaluate the
    # spline up to u_cand
    field_spline, u_cand = stream_param(Q, point)
    us = np.linspace(0, u_cand, Nz)
    stream_coords = np.array([field_spline(u) for u in us])
    streams[i] = stream_coords

cs_grid = pv.StructuredGrid(streams[:,:,0], streams[:,:,1], streams[:,:,2])
cs_grid.save("waverider.vtk")

U, V, P = GlobalSurfInterp(Q=streams[1:-1], p=3, q=3)
P_new = np.zeros((len(P)+2, len(P[0]), len(P[0][0])))
corner_left = np.array([streams[:,:,0][0][0], streams[:,:,1][0][0], streams[:,:,2][0][0]])
corner_right = np.array([streams[:,:,0][-1][-1], streams[:,:,1][-1][-1], streams[:,:,2][-1][-1]])
for i in range(len(P_new)):
    #P_new[i][0] = corner_left
    #P_new[i][-1] = corner_right
    for j in range(len(P_new[0])):
        P_new[0][j] = corner_left
        P_new[-1][j] = corner_right
for i in range(len(P)):
    for j in range(len(P[0])):
        P_new[i+1][j] = P[i][j]


U_new = auto_knot_vector(len(P_new), 3)
V_new = auto_knot_vector(len(P_new[0]), 3)
spline_patch = BSplineSurface(U=U_new, V=V_new, P=P_new, p=3, q=3)
N_u = 20*2
N_v = 10*2
spline_coords = spline_patch.list_eval(N_u=N_u, N_v=N_v)
spline_xs = np.zeros((N_u, N_v))
spline_ys = np.zeros((N_u, N_v))
spline_zs = np.zeros((N_u, N_v))
for i in range(N_u):
    for j in range(N_v):
        coord = spline_coords[i][j]
        spline_xs[i][j] = coord[0]
        spline_ys[i][j] = coord[1]
        spline_zs[i][j] = coord[2]

spline_grid = pv.StructuredGrid(spline_xs, spline_ys, spline_zs)
spline_grid.save("waverider_spline_edit.vtk")
"""
spline_patch_r = BSplineSurface(U=U, V=V, P=P, p=3, q=3)
N_u_r = 11
N_v_r = 11
spline_coords_r = spline_patch.list_eval(N_u=N_u_r, N_v=N_v_r)
spline_xs_r = np.zeros((N_u_r, N_v_r))
spline_ys_r = np.zeros((N_u_r, N_v_r))
spline_zs_r = np.zeros((N_u_r, N_v_r))
for i in range(N_u_r):
    for j in range(N_v_r):
        coord = spline_coords_r[i][j]
        spline_xs_r[i][j] = coord[0]
        spline_ys_r[i][j] = coord[1]
        spline_zs_r[i][j] = coord[2]
   
spline_grid = pv.StructuredGrid(spline_xs_r, spline_ys_r, spline_zs_r)
spline_grid.save("waverider_spline_raw.vtk")
"""
# generate shockwave cone surfaces
def cone_x(r, phi):
    return r * np.cos(phi)
    
def cone_y(r, phi):
    return r * np.sin(phi)
    
def cone_z(x, y, theta, sign):
    a = tan(theta)
    return sign * np.sqrt((x*x + y*y) / (a*a))

# conical shock surface
rs_cs = np.linspace(0, max(field_coords[2]) * tan(field.beta), 100)
phis_cs = np.linspace(0, 2 * pi, 100)
Rs_cs, Phis_cs = np.meshgrid(rs_cs, phis_cs)
X_cs = cone_x(Rs_cs, Phis_cs)
Y_cs = cone_y(Rs_cs, Phis_cs)
Z_cs = cone_z(X_cs, Y_cs, field.beta, 1)

# imaginary cone surface
rs_c = np.linspace(0, max(field_coords[2]) * tan(field.thetac), 100)
phis_c = np.linspace(0, 2 * pi, 100)
Rs_c, Phis_c = np.meshgrid(rs_c, phis_c)
X_c = cone_x(Rs_c, Phis_c)
Y_c = cone_y(Rs_c, Phis_c)
Z_c = cone_z(X_c, Y_c, field.thetac, 1)

"""
# waverider shock surface
xs_wr = np.zeros((N_streams, Nz))
ys_wr = np.zeros((N_streams, Nz))
zs_wr = np.zeros((N_streams, Nz))
for i in range(N_streams):
    phi_1 = atan2(streams[i][0][1], streams[i][0][0])
    phi_2 = atan2(streams[N_streams-1-i][0][1], streams[N_streams-1-i][0][0])
    phis = np.linspace(phi_1, phi_2, N_streams)
    z_sec = streams[i][0][2]
    r = z_sec * tan(field.beta)
    for j in range(N_streams):
        xs_wr[i][j] = r * cos(phis[j]) 
        ys_wr[i][j] = r * sin(phis[j])
        zs_wr[i][j] = z_sec
"""

def FindYDistanceRes(params, x, z, surface):
        u = params[0]
        v = params[1]

        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        if v > 1:
            v = 1
        elif v < 0:
            v = 0

        surf_point = surface(u, v)
        return [surf_point[0] - x,  surf_point[2] - z]

def FindYSurf(x, z, surface):
    us = np.linspace(0, 1, 10)
    vs = np.linspace(0, 1, 10)
    ress = np.zeros((len(us), len(vs)))
    for i in range(len(us)):
        for j in range(len(vs)):
            param_guess = [us[i], vs[j]]
            sol = optimize.root(FindYDistanceRes, param_guess, args=(x, z, surface))
            """
            if sol.success == False:
                raise ValueError('Root finder did not converge.')
            """
            param_sol = sol.x
            #print(f'param_sol = {param_sol}')
            u = param_sol[0]
            v = param_sol[1]
            if u > 1:
                u = 1
            elif u < 0:
                u = 0
            if v > 1:
                v = 1
            elif v < 0:
                v = 0
            res = FindYDistanceRes([u, v], x, z, surface)
            dist = sqrt(res[0]**2 + res[1]**2)
            ress[i][j] = dist
    #print()
    ind_guess = np.unravel_index(np.argmin(ress, axis=None), ress.shape)
    u_guess = us[ind_guess[0]]
    v_guess = vs[ind_guess[1]]
    sol = optimize.root(FindYDistanceRes, [u_guess, v_guess], args=(x, z, surface))
    print(sol.success)
    u = sol.x[0]
    v = sol.x[1]
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    if v > 1:
        v = 1
    elif v < 0:
        v = 0

    return surface(u, v)

xs_wr_rp = np.zeros((int(len(spline_xs)/2), int(len(spline_xs[0])/2)))
ys_wr_rp = np.zeros((int(len(spline_xs)/2), int(len(spline_xs[0])/2)))
zs_wr_rp = np.zeros((int(len(spline_xs)/2), int(len(spline_xs[0])/2)))
for i in range(int(len(spline_xs)/2)):
    x_i_0 = spline_xs[i][0]
    x_i_f = spline_xs[len(spline_xs)-1-i][0]
    xs_i = np.linspace(x_i_0, x_i_f, int(len(spline_xs[0])/2))
    z_i = spline_zs[i][0]
    for j in range(int(len(spline_xs[0])/2)):
        #print(f'i={i}, j ={j}')
        point = FindYSurf(xs_i[j], z_i, spline_patch)
        xs_wr_rp[i][j] = point[0]
        ys_wr_rp[i][j] = point[1]
        zs_wr_rp[i][j] = point[2]

# waverider shock surface
xs_wr = np.zeros(xs_wr_rp.shape)
ys_wr = np.zeros(ys_wr_rp.shape)
zs_wr = np.zeros(zs_wr_rp.shape)
for i in range(len(xs_wr_rp)):
    phi_1 = atan2(ys_wr_rp[i][0], xs_wr_rp[i][0])
    phi_2 = atan2(ys_wr_rp[i][-1], xs_wr_rp[i][-1])
    print(f'phi1 = {phi_1*180/pi:.4}, phi2 = {phi_2*180/pi:.4}')
    phis = np.linspace(phi_1, phi_2, int(len(xs_wr_rp[0])))
    for j in range(len(phis)):
        z_sec = zs_wr_rp[i][j]
        r = z_sec * tan(field.beta)
        xs_wr[i][j] = r * cos(phis[j]) 
        ys_wr[i][j] = r * sin(phis[j])
        zs_wr[i][j] = z_sec

inletA_xs = np.zeros((len(xs_wr_rp), len(xs_wr_rp[0]) + len(xs_wr[0]) - 1))
inletA_ys = np.zeros((len(ys_wr_rp), len(ys_wr_rp[0]) + len(ys_wr[0]) - 1))
inletA_zs = np.zeros((len(zs_wr_rp), len(zs_wr_rp[0]) + len(zs_wr[0]) - 1))

for i in range(len(xs_wr)):
    x1 = np.append(np.flip(xs_wr[i,1:-1]), np.array(xs_wr_rp[i][0]))
    inletA_xs[i,:] = np.append(xs_wr_rp[i], x1)
    y1 = np.append(np.flip(ys_wr[i,1:-1]), np.array(ys_wr_rp[i][0]))
    inletA_ys[i,:] = np.append(ys_wr_rp[i], y1)
    z1 = np.append(np.flip(zs_wr[i,1:-1]), np.array(zs_wr_rp[i][0]))
    inletA_zs[i,:] = np.append(zs_wr_rp[i], z1)


inletA = np.array([inletA_xs[:-10], inletA_ys[:-10], inletA_zs[:-10]])



from math import sqrt, pi
import numpy as np
from csgen.busemann_streamline_tracer import BusemannStreamlineTracer
from pyffd.spline_fitting import GlobalSurfInterp
from pyffd.surface import BSplineSurface
from pyffd.path import NURBSEllipse
import matplotlib.pyplot as plt
import pyvista as pv
from pyffd.cad_output import NURBSSurfaceToIGES



# Busemann flow field design parameters
q0 = 50E3
M0 = 10
gamma = 1.4
p0 = 2 * q0 / (gamma * M0*M0)
p3 = 50E3
p3p0 = p3/p0
dtheta = 0.001
h = 1

"""
# define capture shape
def ellipse(theta, a, b, h, k):
    e = np.sqrt(1 - (b/a)**2)
    r = a * (1 - e*e) / (1 + e * np.cos(theta))
    return r * np.cos(theta) + e*a, r * np.sin(theta) + k

a_cap = 2*4/50*3
b_cap = 1.5*4/50*3
h_cap = 0
k_cap = b_cap + 0.1
"""
Nstreams = 20
"""
thetas = np.linspace(pi, 3*pi, Nstreams)
xs_cap = [ellipse(theta, a_cap, b_cap, h_cap, k_cap)[0] for theta in thetas]
ys_cap = [ellipse(theta, a_cap, b_cap, h_cap, k_cap)[1] for theta in thetas]
"""
"""
us = np.linspace(0, 1, Nstreams)
xs_cap = [NURBSEllipse(a_cap, b_cap, h_cap, k_cap)(u)[0] for u in us]
ys_cap = [NURBSEllipse(a_cap, b_cap, h_cap, k_cap)(u)[1] for u in us]
"""
xs_cap = inletA_xs[-10]
ys_cap = inletA_ys[-10]
del_y = abs(np.amin(ys_cap))
h_y = abs(np.amin(ys_cap)) - abs(np.amax(ys_cap))
for i in range(len(ys_cap)):
    ys_cap[i] += del_y + h_y/10

cap_coords = []
for i in range(len(xs_cap)):
    cap_coords.append([xs_cap[i], ys_cap[i]])

# streamline trace capture shape through Busemann flow field
xs_surf, ys_surf, zs_surf = BusemannStreamlineTracer(p3p0, M0, gamma, h, dtheta, cap_coords)
field_length = np.amax(zs_surf) - np.amin(zs_surf)
wr_length = np.amax(inletA_zs[:-10]) - np.amin(inletA_zs[:-10])
scale = wr_length / field_length

# create interpolative B-Spline surface
Q = np.zeros((len(xs_surf), len(xs_surf[0]), 3))
for i in range(len(Q)):
    for j in range(len(Q[0])):
        Q[i][j] = [xs_surf[i][j] * scale, ys_surf[i][j] * scale, zs_surf[i][j] * scale]

p = 3
q = 3
U, V, P = GlobalSurfInterp(Q, p, q)
spline_patch = BSplineSurface(p=p, q=q, U=U, V=V, P=P)

N_u = 19
N_v = len(inletA_xs[:-10])
spline_coords = spline_patch.list_eval(N_u=N_u, N_v=N_v)
splineA_xs = np.zeros((N_u, N_v))
splineA_ys = np.zeros((N_u, N_v))
splineA_zs = np.zeros((N_u, N_v))
for i in range(N_u):
    for j in range(N_v):
        coord = spline_coords[i][j]
        splineA_xs[i][j] = coord[0]
        splineA_ys[i][j] = coord[1] + (np.amax(inletA_ys[:-10]) - np.amax(ys_surf)*scale) - 0.1
        splineA_zs[i][j] = coord[2] + (np.amin(inletA_zs[:-10]) - np.amin(zs_surf)*scale)


inletB = np.array([splineA_xs.T, splineA_ys.T, splineA_zs.T])
inletB_grid = pv.StructuredGrid(splineB_xs.T, splineB_ys.T, splineB_zs.T)
inletB_grid.save("capture_inlet.vtk")



# Busemann flow field design parameters
q0 = 50E3
M0 = 10
gamma = 1.4
p0 = 2 * q0 / (gamma * M0*M0)
p3 = 50E3
p3p0 = p3/p0
dtheta = 0.001
h = 1

a_cap = 2*4/50*3
b_cap = 1*4/50*3
h_cap = 0
k_cap = b_cap + 0.1

Nstreams = 20

us = np.linspace(0, 1, Nstreams)
xs_cap = [NURBSEllipse(a_cap, b_cap, h_cap, k_cap)(u)[0] for u in us]
ys_cap = [NURBSEllipse(a_cap, b_cap, h_cap, k_cap)(u)[1] for u in us]

cap_coords = []
for i in range(len(xs_cap)):
    cap_coords.append([xs_cap[i], ys_cap[i]])

# streamline trace capture shape through Busemann flow field
xs_surf, ys_surf, zs_surf = BusemannStreamlineTracer(p3p0, M0, gamma, h, dtheta, cap_coords)
field_length = np.amax(zs_surf) - np.amin(zs_surf)
wr_length = np.amax(inletA_zs[:-10]) - np.amin(inletA_zs[:-10])
scale = wr_length / field_length

# create interpolative B-Spline surface
Q = np.zeros((len(xs_surf), len(xs_surf[0]), 3))
for i in range(len(Q)):
    for j in range(len(Q[0])):
        Q[i][j] = [xs_surf[i][j] * scale, ys_surf[i][j] * scale, zs_surf[i][j] * scale]

p = 3
q = 3
U, V, P = GlobalSurfInterp(Q, p, q)
spline_patch = BSplineSurface(p=p, q=q, U=U, V=V, P=P)

N_u = 19
N_v = len(inletA_xs[:-10])
spline_coords = spline_patch.list_eval(N_u=N_u, N_v=N_v)
splineA_xs = np.zeros((N_u, N_v))
splineA_ys = np.zeros((N_u, N_v))
splineA_zs = np.zeros((N_u, N_v))
for i in range(N_u):
    for j in range(N_v):
        coord = spline_coords[i][j]
        splineA_xs[i][j] = coord[0]
        splineA_ys[i][j] = coord[1] + (np.amax(inletA_ys[:-10]) - np.amax(ys_surf)*scale) - 0.1
        splineA_zs[i][j] = coord[2] + (np.amin(inletA_zs[:-10]) - np.amin(zs_surf)*scale)


inletD = np.array([splineA_xs.T, splineA_ys.T, splineA_zs.T])
inletD_grid = pv.StructuredGrid(splineA_xs.T, splineA_ys.T, splineA_zs.T)
inletD_grid.save("exit_inlet.vtk")

"""
# plot calculatied cross-section at z=L_i
plt.figure(figsize=(16, 9))
plt.rcParams.update({"text.usetex": True, "font.size": 16})
plt.plot(spline_xs[-1,:], spline_ys[-1,:], 'b', label='Forebody')
plt.plot(xs_shock, ys_shock, 'r', label='Conical Shock')
plt.plot(xs_cone, ys_cone, 'k', label='Base Cone')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig('forebody_inlet_interface_calc.svg', bbox_inches='tight')
plt.show()
"""




# plot surfaces
plotter = pv.Plotter() 
cs_grid = pv.StructuredGrid(X_cs, Y_cs, Z_cs)
cs_grid.save("shock.vtk")
c_grid = pv.StructuredGrid(X_c, Y_c, Z_c)
c_grid.save("cone.vtk")
#wr_grid = pv.StructuredGrid(X_wr, Y_wr, Z_wr)
wr_grid = pv.StructuredGrid(xs_wr, ys_wr, zs_wr)
wr_grid.save("waverider_shock.vtk")
wr_rp_grid = pv.StructuredGrid(xs_wr_rp, ys_wr_rp, zs_wr_rp)
wr_rp_grid.save("waverider_reparam.vtk")
inletA_grid = pv.StructuredGrid(inletA_xs[:-10], inletA_ys[:-10], inletA_zs[:-10])
inletA_grid.save("waverider_inlet.vtk")
"""
inletA_grid = pv.StructuredGrid(splineA_xs.T, splineA_ys.T, splineA_zs.T)
inletA_grid.save("sugar_scoop_inlet.vtk")
"""



print(inletA_xs[:-10])
print(splineA_xs)









def blend(inletA, inletB, z_cap, z_cowl, alpha):

    # check if inletA and inletB surface arrays have same shape
    print(inletA.shape)
    print(inletB.shape)
    if inletA.shape != inletB.shape:
        raise ValueError('Inlet surfaces contain different \
                          number of points.')

    def E(z, z_cap, z_cowl):
        return ((z - z_cap) / (z_cowl - z_cap))**alpha

    def f(z, sectionA_i, sectionB_i, z_cap, z_cowl, alpha):
        #print(E(z, z_cap, z_cowl))
        if sectionA_i < 0 or sectionB_i < 0:
            return -abs(sectionA_i) ** (1 - E(z, z_cap, z_cowl)) * \
                    abs(sectionB_i) ** E(z, z_cap, z_cowl)
        else:
            return abs(sectionA_i) ** (1 - E(z, z_cap, z_cowl)) * \
                   abs(sectionB_i) ** E(z, z_cap, z_cowl)

    inletA_xs = inletA[0]
    inletA_ys = inletA[1]
    inletA_zs = inletA[2]
    
    inletB_xs = inletB[0]
    inletB_ys = inletB[1]
    inletB_zs = inletB[2]
    
    inletC_xs = np.zeros(inletA_xs.shape)
    inletC_ys = np.zeros(inletA_ys.shape)
    inletC_zs = inletA_zs

    for i in range(len(inletA_xs)):
        for j in range(len(inletA_xs[0])):
            z_ij = inletC_zs[i][j]

            Ax_ij = inletA_xs[i][j]
            Bx_ij = inletB_xs[i][j]
            inletC_xs[i][j] = f(z_ij, Ax_ij, Bx_ij, z_cap, z_cowl, alpha)

            Ay_ij = inletA_ys[i][j]
            By_ij = inletB_ys[i][j]
            inletC_ys[i][j] = f(z_ij, Ay_ij, By_ij, z_cap, z_cowl, alpha)

    return inletC_xs, inletC_ys, inletC_zs


z_cap = np.amin(inletA_zs[:-10])
z_cowl = np.amax(splineA_zs)
alpha = 1.8
inletC_xs, inletC_ys, inletC_zs = blend(inletA, inletB, z_cap, z_cowl, alpha)
splineC_grid = pv.StructuredGrid(inletC_xs, inletC_ys, inletC_zs)
splineC_grid.save("blended_inlet.vtk")
