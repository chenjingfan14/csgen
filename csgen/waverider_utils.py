"""
Utility functions for waverider design.

Author: Reece Otto 04/04/2022
"""
import numpy as np
import pyvista as pv
from math import sqrt, tan

def top_surface(top_contour, bottom_surface):
	# ensure top_contour and bottom_surface have same amount of phi points
	if len(top_contour) != len(bottom_surface):
		raise Exception('Top and bottom contours have different number of '
			'points.')

	n_phi = len(top_contour)
	n_z = len(bottom_surface)
	z_base = bottom_surface[0][0][2]
	surf = np.nan * np.ones((n_phi, n_z, 3))
	for i in range(n_phi):
		coord_start = bottom_surface[i][0]
		coord_end = [top_contour[i][0], top_contour[i][1], z_base]
		xs_i = np.linspace(coord_start[0], coord_end[0], n_z)
		ys_i = np.linspace(coord_start[1], coord_end[1], n_z)
		zs_i = np.linspace(coord_start[2], coord_end[2], n_z)
		for j in range(n_z):
			surf[i][j] = [xs_i[j], ys_i[j], zs_i[j]]

	top_grid = pv.StructuredGrid(surf[:,:,0], surf[:,:,1], surf[:,:,2])
	top_grid.save("waverider_top.vtk")
	return surf



def normal_exit_shock(x, beta, tau, y_exit, z_exit):
	A = tan(beta)**2 * tan(tau)
	B = A * tan(tau)
	C = 1 + B
	D = 2*(B*y_exit + A*z_exit)
	E = B*y_exit**2 + 2*A*y_exit*z_exit + tan(beta)**2 * z_exit**2
	G = (4*E*C + D*D)/(4*C*C)
	F = G * C
	H = D/(2*C)
	return -H - sqrt(G*(1 - x*x/F))
