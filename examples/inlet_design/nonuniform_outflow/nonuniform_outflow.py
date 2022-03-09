"""
Creating a compressive flow field with a prescribed outflow pressure 
distribution.

Author: Reece Otto 08/12/2021
TODO: need a more appropriate set of initial guesses for busemann integrator
"""

from math import sin, cos, pi
import numpy as np
from csgen.busemann import busemann_M1_p3p1
from nurbskit.spline_fitting import global_curve_interp
from nurbskit.path import BSpline
import pyvista as pv

# define pressure distribution function
def p_func(p_mid, p_top, phi):
    return (p_mid - p_top) * cos(2 * phi) / 2 + (p_top + p_mid) / 2

# free-stream conditions
q0 = 50E3
M0 = 10
gamma = 1.4
p0 = 2 * q0 / (gamma * M0*M0)

# Taylor-Maccoll integration
dtheta = 0.001 # theta step size
h = 1          # flow field throat radius (this is changed later)

# define number of streamlines and corresponding phi angles
Nstreams = 20
phis = np.linspace(0, 2*pi, Nstreams)

# define pressure at the top and bottom of inlet
p_mid = 30E3
p_top = 70E3

# specify discretization of flow field surface in z direction
Nz = 100
us = np.linspace(0, 1, Nz)

# initialise flow field surface coordinate arrays
xs_surf = np.zeros((Nstreams, Nz))
ys_surf, zs_surf = np.copy(xs_surf), np.copy(xs_surf)

# initialise Mach wave surface coordinate arrays
xs_mw = np.zeros((Nstreams, 2))
ys_mw, zs_mw = np.copy(xs_mw), np.copy(xs_mw)

# initialise terminating shock surface coordinate arrays
xs_ts = np.zeros((Nstreams, 2))
ys_ts, zs_ts = np.copy(xs_ts), np.copy(xs_ts)

# calculate coordinates of flow field surface on a structured mesh
for i in range(Nstreams):
    # calculate compression ratio for streamline i
    p3 = p_func(p_mid, p_top, phis[i])
    p3p0 = p3 / p0
    
    # generate Busemann flow field for streamline i
    print(f'Generating flow field for streamline {i}.')
    buse = busemann_M1_p3p1(M0, p3p0, gamma, dtheta, verbosity=0)
    buse_coords = buse.xyz_coords()
    
    # fit degree 3 B-spline curve to Busemann contour
    U, P = global_curve_interp(buse_coords, 3)
    spline = BSpline(P=P, U=U, p=3)
    
    # evaluate Nz points along B-Spline curve
    xs_spline = np.zeros((Nz))
    ys_spline = np.zeros((Nz))
    zs_spline = np.zeros((Nz))
    for j in range(Nz):
        spline_coord_j = spline(us[j])
        xs_spline[j] = spline_coord_j[0]
        ys_spline[j] = spline_coord_j[1]
        zs_spline[j] = spline_coord_j[2]
    
    # apply rotation matrix to calculate coordinates once rotated by angle phi
    for j in range(Nz):
        xs_surf[i][j] = cos(phis[i])*xs_spline[j]  + sin(phis[i])*ys_spline[j]
        ys_surf[i][j] = -sin(phis[i])*xs_spline[j] + cos(phis[i])*ys_spline[j]
        zs_surf[i][j] = zs_spline[j]
    
    # calculate Mach wave coordinates
    xs_mw[i] = np.array([xs_surf[i][-1], 0.0])
    ys_mw[i] = np.array([ys_surf[i][-1], 0.0])
    zs_mw[i] = np.array([zs_surf[i][-1], 0.0])
    
    # calculate terminating shock wave coordinates
    xs_ts[i] = np.array([0.0, xs_surf[i][0]])
    ys_ts[i] = np.array([0.0, ys_surf[i][0]])
    zs_ts[i] = np.array([0.0, zs_surf[i][0]])

# translate streamlines along z-axis to ensure they begin at z = 0.0
for i in range(len(zs_surf)):
    zs_mw[i][:] -= zs_surf[i].min()
    zs_ts[i][:] -= zs_surf[i].min()
    zs_surf[i][:] -= zs_surf[i].min()

# scale streamlines to ensure they end at same z coord
z_end = zs_surf.max()
for i in range(len(zs_surf)):
    scale_i = z_end / zs_surf[i][0]
    
    xs_surf[i][:] *= scale_i 
    ys_surf[i][:] *= scale_i 
    zs_surf[i][:] *= scale_i 
    
    xs_mw[i][:] *= scale_i 
    ys_mw[i][:] *= scale_i 
    zs_mw[i][:] *= scale_i
    
    xs_ts[i][:] *= scale_i 
    ys_ts[i][:] *= scale_i 
    zs_ts[i][:] *= scale_i         

# save grids as vtk
surf_grid = pv.StructuredGrid(xs_surf, ys_surf, zs_surf)
mw_grid = pv.StructuredGrid(xs_mw, ys_mw, zs_mw)
ts_grid = pv.StructuredGrid(xs_ts, ys_ts, zs_ts)
surf_grid.save("nonuniform_surf.vtk")
mw_grid.save("nonuniform_mw.vtk")
ts_grid.save("nonuniform_ts.vtk")
