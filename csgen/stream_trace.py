"""
Tracing streamlines through a puffin flow field.

Author: Reece Otto 24/03/2022
"""
import os
import numpy as np
from scipy import interpolate
import csv

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

def stream_trace(locus_point, file_name, job_name, flow_data_name, n_cells, 
	p_buffer=1.4, dt=1.0E-6, max_step=100000):
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
	#dt = 1.0E-6
	point_i = locus_point
	point_i_data = point_data
	p0 = stream_data[0][0][6]
	if point_i_data[6] >= p_buffer * p0:
		streamline_data = [point_i_data]
	else:
		streamline_data = []
	step = 0
	max_x = stream_data[-1][0][0]
	
	while step < max_step:
		point_i = integrate_stream(point_i_data, dt)
		if point_i[0] >= max_x:
			break
		x_ind_1, y_ind_1 = enclosing_cell_centres(point_i, stream_data)
		point_i_data = interpolate_data(x_ind_1, y_ind_1, point_i, stream_data, 
										variable_names)
		
		if point_i_data[6] >= p_buffer * p0:
			streamline_data.append(point_i_data)
		step += 1

	# interpolate for properties at end of streamtube and append to data list
	x_extrap = max_x
	data_extrap = extrapolate_stream(x_extrap, streamline_data, variable_names)
	streamline_data.append(data_extrap)

	"""
	# write data to csv file
	with open(file_name + '.csv', 'w') as f:
		write = csv.writer(f, delimiter=' ')
		write.writerow(variable_names)
		write.writerows(streamline_data)
	"""
	# return back to working directory
	os.chdir(current_dir)

	return streamline_data