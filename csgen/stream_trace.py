"""
Tracing streamlines through a puffin flow field.

Author: Reece Otto 24/03/2022
"""
import os
import numpy as np
from scipy import interpolate
import csv
from nurbskit.path import BSpline
from nurbskit.spline_fitting import global_curve_interp
from math import sqrt, sin, cos, atan2

def read_stream_flow_file(file_name, n_cells):
    """
    Read the flow data for a single streamtube.
    """
    data_file = open(file_name, "r")
    txt = data_file.readline()
    if txt.startswith('#'):
        variable_names = txt.strip('# ').split()
        #print('variable_names=', variable_names)
        stream_data = []
        while True:
            slice_data = []
            for j in range(n_cells):
                txt = data_file.readline()
                if txt:
                    items = [float(item) for item in txt.strip().split()]
                    if len(items) > 0: slice_data.append(items)
                else:
                    break
            # At this point, we have the data for a full slice of cells
            # and/or we have arrived at the end of the file.
            if len(slice_data) == 0: break
            stream_data.append(slice_data)
        # At this point, we have the full stream data.
    else:
        print("First line of stream flow file did not start with #")
        stream_data = None
        variableNames = None
    data_file.close()
    return stream_data, variable_names

def enclosing_cell_centres(point, stream_data):
	n_x_cells = len(stream_data)
	n_y_cells = len(stream_data[0])

	# calculate x bounds of streamtube
	x_min = stream_data[0][0][0]
	x_max = stream_data[-1][0][0]
	x_0 = x_min
	slice_ind_0 = 0

	# check if x coord is within streamtube
	if point[0] < x_min or point[0] > x_max:
		raise Exception('Point is not within the x bounds of given StreamTube.')
	# find x coords of bounding cell centres
	else:
		for i in range(1, n_x_cells):
			slice_ind_1 = i
			x_1 = stream_data[i][0][0]
			if point[0] >= x_0 and point[0] <= x_1:
				break
			else:
				x_0 = x_1
				slice_ind_0 = slice_ind_1

	# calculate y bounds of stream at x coordinate of given point
	y_min_left = stream_data[slice_ind_0][0][1]
	y_max_left = stream_data[slice_ind_0][-1][1]
	y_min_right = stream_data[slice_ind_1][0][1]
	y_max_right = stream_data[slice_ind_1][-1][1]
	y_min = np.interp(point[0], [x_min, x_max], [y_min_left, y_min_right])
	y_max = np.interp(point[0], [x_min, x_max], [y_max_left, y_max_right])
	y_A = y_min_left
	y_D = y_min_right
	y0_at_point = np.interp(point[1], [x_0, x_1], [y_A, y_D])
	row_ind_0 = 0

	# check if y coord is in streamtube
	if point[1] < y_min or point[1] > y_max:
		raise Exception('Point is not within the y bounds of given ' + \
			            'StreamTube at the given x coordinate.')
	# find y coords of bounding cell centres
	else:
		for i in range(1, n_y_cells):
			row_ind_1 = i
			y_B = stream_data[slice_ind_0][i][1]
			y_C = stream_data[slice_ind_1][i][1]
			y1_at_point = np.interp(point[1], [x_0, x_1], [y_B, y_C])

			if point[1] >= y0_at_point and point[1] <= y1_at_point:
				break
			else:
				y_A = y_B
				y_D = y_C
				row_ind_0 = row_ind_1
	
	return slice_ind_1, row_ind_1

def interpolate_data(x_ind_1, y_ind_1, point, stream_data, variable_names):
	"""
	Interpolates streamtube data between 4 adjacent cell centres.
	It is assumed that the cell centres are arranged like this:
	   
	B-----C
	|     |
	|     |
	A-----D

	Points A and B, as well as C and D, must share the same x coordinate.
	All 4 y coordinates can be different.
	"""
	# construct x-y grid
	x_ind_0 = x_ind_1 - 1
	y_ind_0 = y_ind_1 - 1

	point_A = [stream_data[x_ind_0][y_ind_0][0], 
	           stream_data[x_ind_0][y_ind_0][1]]
	point_D = [stream_data[x_ind_1][y_ind_0][0], 
	           stream_data[x_ind_1][y_ind_0][1]]
	point_B = [stream_data[x_ind_0][y_ind_1][0], 
	           stream_data[x_ind_0][y_ind_1][1]]
	point_C = [stream_data[x_ind_1][y_ind_1][0], 
	           stream_data[x_ind_1][y_ind_1][1]]

	x_grid = np.array([[point_A[0], point_D[0]],
		               [point_B[0], point_C[0]]])
	y_grid = np.array([[point_A[1], point_D[1]],
		               [point_B[1], point_C[1]]])

	# interpolate each flow property at given point
	point_data = np.nan * np.ones(len(variable_names))
	for i in range(len(variable_names)):
		data_A = stream_data[x_ind_0][y_ind_0][i]
		data_D = stream_data[x_ind_1][y_ind_0][i]
		data_B = stream_data[x_ind_0][y_ind_1][i]
		data_C = stream_data[x_ind_1][y_ind_1][i]

		data_grid = np.array([[data_A, data_D],
		                      [data_B, data_C]])

		data_interp = interpolate.interp2d(x_grid, y_grid, data_grid)
		point_data[i] = data_interp(point[0], point[1])

	return point_data

def integrate_stream(point_data, time_inc):
	x_old = point_data[0]
	y_old = point_data[1]
	vel_x_old = point_data[2]
	vel_y_old = point_data[3]

	x_new = x_old + vel_x_old * time_inc
	y_new = y_old + vel_y_old * time_inc
	return [x_new, y_new]

def extrapolate_stream(x_extrap, streamline_data, variable_names):
	# extract last four x positions
	n_points = len(streamline_data)
	xs = np.nan * np.ones(4)
	for i in range(4):
		xs[i] = streamline_data[n_points-4+i][0]
	
	data_extrap = np.nan * np.ones(len(variable_names))
	data_extrap[0] = x_extrap
	# extrapolate each property at x_extrap
	for i in range(1, len(variable_names)):
		# extract last four values for property i
		values = np.nan * np.ones(4)
		for j in range(4):
			values[j] = streamline_data[n_points-4+j][i]
		extrap_f = interpolate.interp1d(xs, values, kind='cubic', 
				fill_value='extrapolate')
		data_extrap[i] = extrap_f(x_extrap)
	return data_extrap

def interpolate_pressure(p_desired, streamline_data, variable_names):
	# extract last 2 pressure values
	n_points = len(streamline_data)
	ps = np.array(streamline_data)[-2:,6]
	data_interp = np.nan * np.ones(len(variable_names))
	for i in range(len(variable_names)):
		# extract last 2 values for property i
		values = np.array(streamline_data)[-2:,i]
		extrap_f = interpolate.interp1d(ps, values)
		data_interp[i] = extrap_f(p_desired)
	return data_interp

def stream_trace(locus_point, job_name, flow_data_name, n_cells, p_buffer=1.8, 
	             dt=1.0E-6, max_step=100000):
	# read flow data
	# job_name = 'trunc-bd'
	# n_cells = 75 # TODO: read this from config.JSON
	current_dir = os.getcwd()
	os.chdir(current_dir + '/' + job_name)
	#flow_data_name = 'flow-0.data'
	stream_data, variable_names = read_stream_flow_file(flow_data_name, n_cells)


	# find indices of cell centres in stream_data the enclose locus point
	#locus_point = [0.000000e+00, 0.5]
	x_ind_1, y_ind_1 = enclosing_cell_centres(locus_point, stream_data)

	# interpolate data at locus point
	point_data = interpolate_data(x_ind_1, y_ind_1, locus_point, stream_data, 
								  variable_names)

	# integrate velocity to find next point on streamline
	p0 = stream_data[0][0][6]
	p_shock = p_buffer * p0
	point_i = locus_point
	point_i_data = point_data
	streamline_data = [point_i_data]
	step = 0
	max_x = stream_data[-1][0][0]
	shock_detected = False
	
	while step < max_step:
		point_i = integrate_stream(point_i_data, dt)
		if point_i[0] >= max_x:
			break
		x_ind_1, y_ind_1 = enclosing_cell_centres(point_i, stream_data)
		point_i_data = interpolate_data(x_ind_1, y_ind_1, point_i, stream_data, 
										variable_names)
		
		streamline_data.append(point_i_data)

		# detect entrance shock wave and trim
		if point_i_data[6] >= p_shock and shock_detected == False:
			# interpolate streamline for desired pressure
			point_i_data = interpolate_pressure(p_shock, streamline_data, 
				                                variable_names)

			# delete all entries in streamline_data list to trim inlet upstream
			# of shock
			streamline_data = [point_i_data]
			shock_detected = True
		step += 1

	# interpolate for properties at end of streamtube and append to data list
	x_extrap = max_x
	data_extrap = extrapolate_stream(x_extrap, streamline_data, variable_names)
	streamline_data.append(data_extrap)

	# return back to working directory
	os.chdir(current_dir)

	return streamline_data

def inlet_stream_trace(shape_coords, n_z, job_name, flow_data_name,
	                   n_cells, p_buffer=1.8, dt=1.0E-6, max_step=100000):
	# trace streamlines through field and construct surface grid
	n_streams = len(shape_coords)
	surface_grid = np.nan * np.ones((n_streams, n_z, 3))
	for i in range(n_streams):
		print(f'Generating streamline {i}')
		locus_point = [0.0, sqrt(shape_coords[i][0]**2 + shape_coords[i][1]**2)]
		stream_i_data = stream_trace(locus_point, job_name, flow_data_name, 
			                         n_cells, dt=dt, p_buffer=p_buffer)

		# extract streamline and rotate coords
		stream_coords = np.nan * np.ones((len(stream_i_data), 3))
		phi = atan2(shape_coords[i][1], shape_coords[i][0])
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
	return surface_grid