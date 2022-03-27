"""
Tracing streamlines through truncated Busemann field.

Author: Reece Otto 25/03/2022

TODO: generate locus points based on given capture shape
"""
from csgen.stream_trace import stream_trace
from nurbskit.path import Ellipse, BSpline
from nurbskit.spline_fitting import global_curve_interp
from math import cos, sin, sqrt, atan2
import numpy as np
import pyvista as pv

# generate capture shape
n_streams = 101
n_z = 100
a_cap = 2 * 0.15
b_cap = 1 * 0.15
h_cap = 0
k_cap = 1.5 * b_cap
cap_shape = Ellipse(a_cap, b_cap, h_cap, k_cap)
cap_coords = cap_shape.list_eval(n_points=n_streams)

# specify streamline data parameters
job_name = 'trunc-bd'
flow_data_name = 'flow-0.data'
n_cells = 75

# trace streamlines through field and construct surface grid
surface_grid = np.nan * np.ones((n_streams, n_z, 3))
for i in range(n_streams):
	file_name = f'streamline_data-{i}'
	print(f'Generating {file_name}')
	locus_point = [0.0, sqrt(cap_coords[i][0]**2 + cap_coords[i][1]**2)]
	stream_i_data = stream_trace(locus_point, file_name, job_name, 
		                         flow_data_name, n_cells, dt=1E-5)

	# extract streamline and rotate coords
	stream_coords = np.nan * np.ones((len(stream_i_data), 3))
	phi = atan2(cap_coords[i][1], cap_coords[i][0])
	for j in range(len(stream_coords)):
		x = 0.0
		y = np.array(stream_i_data)[j,1]
		z = np.array(stream_i_data)[j,0]
		stream_coords[j][0] = x*cos(phi) - y*sin(phi)
		stream_coords[j][1] = x*sin(phi) + y*cos(phi)
		stream_coords[j][2] = z
	
	# fit B-Spline to streamline
	p = 3
	U, P = global_curve_interp(stream_coords, p)
	spline = BSpline(P=P, U=U, p=p)

	# evaluate B-Spline for n_z points and add coords to surface grid array
	spline_points = spline.list_eval(n_points=n_z)
	surface_grid[i] = spline_points

inlet_grid = pv.StructuredGrid(surface_grid[:,:,0], surface_grid[:,:,1], surface_grid[:,:,2])
inlet_grid.save("stream_traced_inlet.vtk")