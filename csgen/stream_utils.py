"""
Streamline-tracing tools.

Author: Reece Otto 08/11/2021
"""
import os
from nurbskit.path import BSpline
from nurbskit.spline_fitting import global_curve_interp
from nurbskit.point_inversion import point_proj_path
from math import pi, sqrt, sin, cos, tan, atan2, atan, acos
import numpy as np
from scipy import optimize, interpolate
import csv
import pyvista as pv
import json
import matplotlib.pyplot as plt

class Streamline():
    def __init__(self, polar_coords=None, xyz_coords=None):
        # ensure either spherical or cartesian coords were given
        if polar_coords is None and xyz_coords is None: 
           raise Exception('No coordinates were given.')
        
        elif polar_coords is not None and xyz_coords is None:
            self.polar_coords = polar_coords
            self.rs = polar_coords[:,0]
            self.thetas = polar_coords[:,1]
            self.phis = polar_coords[:,2]

            # calculate Cartesian coordinates
            self.xs = np.zeros(len(self.thetas))
            self.ys = np.nan * np.ones(len(self.thetas))
            self.zs = np.copy(self.ys)
            self.xyz_coords = np.nan * np.ones((len(self.thetas), 3))
            for i in range(len(self.thetas)):
                self.ys[i] = self.rs[i] * sin(self.thetas[i])
                self.zs[i] = self.rs[i] * cos(self.thetas[i])
                self.xyz_coords[i] = [self.xs[i], self.ys[i], self.zs[i]]
        
        elif polar_coords is None and xyz_coords is not None:
            self.xyz_coords = xyz_coords
            self.xs = xyz_coords[:,0]
            self.ys = xyz_coords[:,1]
            self.zs = xyz_coords[:,2]

            # calculate spherical coordinates
            self.rs = np.nan * np.ones(len(self.xs))
            self.thetas = np.copy(self.rs)
            self.phis = np.copy(self.rs)
            self.polar_coords = np.nan * np.ones((len(self.thetas), 3))
            for i in range(len(self.xs)):
                self.rs[i] = sqrt(self.xs[i]**2 + self.ys[i]**2 + self.zs[i]**2)
                self.thetas[i] = acos(self.zs[i] / self.rs[i])
                self.phis[i] = atan2(self.ys[i], self.xs[i])
                self.polar_coords[i] = [self.rs[i], self.thetas[i], 
                                        self.phis[i]]

        else:
            # ensure polar and cartesian coords have same shape
            if polar_coords.shape != xyz_coords.shape:
                raise Exception('Polar and Cartesian coordinate arrays have' 
                    'different shape.')

    def save_to_csv(self, file_name='streamline'):
        with open(file_name + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(['#x', 'y', 'z'])
            writer.writerows(self.xyz_coords)

    def truncate(self, trunc_angle=None, lead_edge_angle=None):
        # TODO: add lead_edge_angle procedure
        # ensure either trunc_angle or lead_edge_angle were given, but not both
        if trunc_angle == None and lead_edge_angle == None:
            raise Exception('Either a truncation angle or a leading edge angle '
                'angle must be supplied.')
        if trunc_angle != None and lead_edge_angle != None:
            raise Exception('A truncation angle and leading edge angle were '
                'both given, but only one is required.')

        # truncate streamline with truncation angle
        if trunc_angle != None:
            theta_trunc = self.thetas[0]-trunc_angle
            ind_upper = 0
            found = False
            # find index of truncation point
            while not found:
                if self.thetas[ind_upper] > theta_trunc:
                    ind_upper += 1
                else:
                    found = True

            # extract coords that bound truncation point
            thetas_interp = self.thetas[ind_upper-1:ind_upper+1]
            rs_interp = self.rs[ind_upper-1:ind_upper+1]
            phis_interp = self.phis[ind_upper-1:ind_upper+1]

            # interpolate r and phi at truncation point
            r_func = interpolate.interp1d(thetas_interp, rs_interp)
            r_trunc = r_func(theta_trunc)
            phi_func = interpolate.interp1d(thetas_interp, phis_interp)
            phi_trunc = phi_func(theta_trunc)

            # delete coords that exist before truncation point
            thetas = self.thetas[ind_upper:]
            rs = self.rs[ind_upper:]
            phis = self.phis[ind_upper:]

            # prepend truncated coordinates to respective lists
            thetas = np.insert(thetas, 0, float(theta_trunc))
            rs = np.insert(rs, 0, float(r_trunc))
            phis = np.insert(phis, 0, float(phi_trunc))
            polar_coords = np.nan * np.ones((len(thetas), 3))
            for i in range(len(polar_coords)):
                polar_coords[i] = [rs[i], thetas[i], phis[i]]

        return Streamline(polar_coords=polar_coords)

    def scale(self, x_scale=1.0, y_scale=1.0, z_scale=1.0):
        xyz_coords = np.copy(self.xyz_coords)
        for i in range(len(xyz_coords)):
            xyz_coords[i][0] = self.xs[i] * x_scale
            xyz_coords[i][1] = self.ys[i] * y_scale
            xyz_coords[i][2] = self.zs[i] * z_scale

        return Streamline(xyz_coords=xyz_coords)

    def translate(self, x_shift=0.0, y_shift=0.0, z_shift=0.0):
        xyz_shifted = np.copy(self.xyz_coords)
        for i in range(len(xyz_shifted)):
            x = self.xs[i] + x_shift
            y = self.ys[i] + y_shift
            z = self.zs[i] + z_shift
            xyz_shifted[i] = [x, y, z]
        return Streamline(xyz_coords=xyz_shifted)

#------------------------------------------------------------------------------#
#               Streamline Tracing Function for Busemann Fields                #
#------------------------------------------------------------------------------#
def busemann_stream_trace(shape_coords, field, plane='capture'):
    if plane == 'capture':
        print('\nTracing capture shape through flow field.')
    elif plane == 'exit':
        print('\nTracing exit shape through flow field.')
    else:
        raise ValueError("Plane must either be 'capture' or 'exit'.")

    # extract coords of entrance or exit shape
    shape_xs, shape_ys = shape_coords[:,0], shape_coords[:,1]
    n_streams = len(shape_xs)

    # intialise raw streamline to perform transformations on
    n_z = len(field.zs)
    
    # get phi and radial scaling factors for each streamline of shape
    shape_zs = np.nan * np.zeros(len(shape_xs))
    phis_shape = np.nan * np.zeros(len(shape_xs))
    sfs_shape = np.nan * np.zeros(len(shape_xs))
    for i in range(n_streams):
        phis_shape[i] = atan2(shape_ys[i], shape_xs[i])
        if plane == 'capture':
            # project capture shape onto Mach wave
            shape_zs[i] = -sqrt((shape_xs[i]**2 + shape_ys[i]**2) / \
                (tan(field.mu)**2))
            r_shape = sqrt(shape_xs[i]**2 + shape_ys[i]**2 + shape_zs[i]**2)
            sfs_shape[i] = r_shape / field.rs[-1]
            
        else:
            # project capture shape onto terminating shock
            shape_zs[i] = sqrt((shape_xs[i]**2 + shape_ys[i]**2) / \
            (tan(field.thetas[0])**2))
            r_shape = sqrt(shape_xs[i]**2 + shape_ys[i]**2 + shape_zs[i]**2)
            sfs_shape[i] = r_shape / field.rs[0]
            
    # find Cartesian coordinates of each point along each streamline
    inlet = np.nan * np.ones((n_streams, n_z, 3))
    for i in range(n_streams):
        for j in range(n_z):
            r_j = sqrt(field.xs[j]**2 + field.ys[j]**2 + field.zs[j]**2)
            x_b_rot = r_j * sin(field.thetas[j]) * cos(phis_shape[i])
            y_b_rot = r_j * sin(field.thetas[j]) * sin(phis_shape[i])
            z_b_rot = r_j * cos(field.thetas[j])
            r_buse = sqrt(x_b_rot**2 + y_b_rot**2 + z_b_rot**2)
            
            inlet[i][j][0] = sfs_shape[i] * r_buse * sin(field.thetas[j]) \
                             * cos(phis_shape[i])
            inlet[i][j][1] = sfs_shape[i] * r_buse * sin(field.thetas[j]) \
                             * sin(phis_shape[i])
            inlet[i][j][2] = sfs_shape[i] * r_buse * cos(field.thetas[j])
    
    return inlet

#------------------------------------------------------------------------------#
#                  Streamline Tracing Function for Waveriders                  #
#------------------------------------------------------------------------------#
def waverider_stream_trace(design_vals):
    """
    Traces streamlines from a given base shape through a conical flow field.

    Arguments:
        base_coords: array of coordinates for waverider base shape
        stream_coords: array of coordinates for a conical field streamline
        z_base: z coordinate of waverider base
        n_z: number of coordinates to be evaluated along each streamline
        tol: convergence tolerance for intersection between streamline and base
             shape
    """
    # unpack dictionary
    z_base = design_vals['z_base']
    base_coords = design_vals['base_coords']
    stream_coords = design_vals['stream_coords']
    n_phi = design_vals.get('n_phi', 51)
    n_z = design_vals.get('n_z', 51)
    tol = design_vals.get('tol', 1.0E-5)
    save_VTK = design_vals.get('save_VTK', False)

    def stream_transform(scale, point, stream):
        """
        Rotates a streamline to the phi angle of a given point, then scales the 
        streamline by given scaling factor.

        This function is iterated upon to find the scaling factor that causes 
        the streamline to intersect the given point.
        """
        # calculate phi angle required for streamline
        phi = atan2(point[1], point[0])

        # rotate streamline to angle phi and scale
        stream_trans = np.nan * np.ones(stream.shape)
        for i in range(len(stream_coords)):
            r = sqrt(stream[i][0]**2 + stream[i][1]**2 + stream[i][2]**2)
            theta = atan(sqrt(stream[i][0]**2 + stream[i][1]**2) / stream[i][2])
            stream_trans[i][0] = float(scale * r * sin(theta) * cos(phi))
            stream_trans[i][1] = float(scale * r * sin(theta) * sin(phi))
            stream_trans[i][2] = float(scale * r * cos(theta))
        
        return stream_trans

    def stream_param(point, stream):
        """
        Parameterizes a given streamline with a cubic B-Spline, then calculates
        the point on the B-Spline that is closest to the given Cartesian point.
        """
        # parameterise streamline
        p = 3
        U, P = global_curve_interp(stream, p)
        stream_spline = BSpline(U=U, P=P, p=p)

        # calculate point on curve closest to given Cartesian point
        u_cand = point_proj_path(stream_spline, point, tol_dist=1E-6, 
            tol_cos=1E-6, tol_end=1E-6)

        return stream_spline, u_cand

    def stream_find(scale, point, stream):
        """
        Returns the distance between a given point and the closest point on a
        scaled streamline.
        """
        # transform streamline
        stream_trans = stream_transform(scale, point, stream)

        # parameterize transformed streamline and calculate candidate u coord
        stream_spline, u_cand = stream_param(point, stream_trans)

        # return L3 norm between given Cartesian point and point projected on
        # B-Spline path
        curve_point = stream_spline(u_cand)

        return np.linalg.norm(point - curve_point)

    print('\nRunning streamline tracer.')
    wr_coords = np.nan * np.ones((len(base_coords), n_z, 3))
    for i in range(len(base_coords)):
        # extract point i of waverider base shape
        point = np.array([base_coords[i][0], base_coords[i][1], z_base])

        # find the scale of the streamline that runs through each point along 
        #the waverider base contour
        stream_sol = optimize.root(stream_find, 1.2, 
            args=(point, stream_coords), tol=tol)
        if stream_sol.success == False:
            raise AssertionError('Root finder failed to converge.')
        scale = stream_sol.x

        # transform the field coordinates
        stream_trans = stream_transform(scale, point, stream_coords)

        # parameterise the field coords and evaluate the spline up to u_cand
        stream_spline, u_cand = stream_param(point, stream_trans)
        us = np.linspace(0, u_cand, n_z)
        stream_i_coords = np.array([stream_spline(u) for u in us])
        wr_coords[i] = stream_i_coords

    if save_VTK == True:
        wr_grid = pv.StructuredGrid(wr_coords[:,:,0], wr_coords[:,:,1], 
            wr_coords[:,:,2])
        wr_grid.save("waverider_surf.vtk")

    return wr_coords

#------------------------------------------------------------------------------#
#                    Streamline Tracing Tools for Puffin                       #
#------------------------------------------------------------------------------#
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

def stream_trace(locus_point, job_name, p_rat_shock=2.0, dt=1.0E-6, 
    shock_data_only=False, max_step=100000):
    # change directory to access puffin data
    current_dir = os.getcwd()
    os.chdir(current_dir + '/' + job_name)

    # extract number of y cells from JSON file
    f = open('config.json')
    config_data = json.load(f)
    n_cells = config_data['ncells_0']
    f.close()

    # read flow data
    flow_data_name = 'flow-0.data'
    stream_data, variable_names = read_stream_flow_file(flow_data_name, n_cells)

    # find indices of cell centres in stream_data the enclose locus point
    x_ind_1, y_ind_1 = enclosing_cell_centres(locus_point, stream_data)

    # interpolate data at locus point
    point_data = interpolate_data(x_ind_1, y_ind_1, locus_point, stream_data, 
                                  variable_names)

    # integrate velocity to find next point on streamline
    p0 = stream_data[0][0][6]
    p_shock = p_rat_shock * p0
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
            if shock_data_only == True:
                break
        step += 1

    # interpolate for properties at end of streamtube and append to data list
    if shock_data_only == False:
        x_extrap = max_x
        data_extrap = extrapolate_stream(x_extrap, streamline_data, 
            variable_names)
        streamline_data.append(data_extrap)

    # return back to working directory
    os.chdir(current_dir)

    return streamline_data

def inlet_stream_trace(design_vals):
    # unpack dictionary
    puffin_dir = design_vals['puffin_dir']
    job_name = design_vals['job_name']
    shape_coords = design_vals['shape_coords']
    n_phi = design_vals.get('n_phi', 51)
    n_z = design_vals.get('n_z', 100)
    p_rat_shock = design_vals.get('p_rat_shock', 2.0)
    dt = design_vals.get('dt', 1.0E-6)
    max_step = design_vals.get('max_step', 100000)
    plot_shape = design_vals.get('plot_shape', False)
    shape_label = design_vals.get('shape_label', 'Shape')
    file_name_shape = design_vals.get('file_name_shape', 'shape')
    save_VTK = design_vals.get('save_VTK', False)
    file_name_VTK = design_vals.get('file_name_VTK', 'inlet')

    # plot shape, if desired
    if plot_shape == True:
        fig = plt.figure(figsize=(16, 9))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.size": 20
            })
        ax = plt.axes()
        ax.plot(shape_coords[:,0], shape_coords[:,1], 'k-', label=shape_label)
        ax.scatter(shape_coords[:,0], shape_coords[:,1], color='black')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        fig.savefig(file_name_shape + '.svg', bbox_inches='tight')

    # trace streamlines through field and construct surface grid
    working_dir = os.getcwd()
    os.chdir(puffin_dir)
    n_streams = len(shape_coords)
    surface_grid = np.nan * np.ones((n_phi, n_z, 3))
    for i in range(n_phi):
        print(f'Generating streamline {i}')
        locus_point = [0.0, sqrt(shape_coords[i][0]**2 + shape_coords[i][1]**2)]
        stream_i_data = stream_trace(locus_point, job_name, dt=dt, 
            p_rat_shock=p_rat_shock, max_step=max_step)

        # extract streamline and rotate coords
        stream_coords = np.nan * np.ones((len(stream_i_data), 3))
        phi = atan2(shape_coords[i][1], shape_coords[i][0]) - pi/2
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
    
    os.chdir(working_dir)

    # save VTK file, if desired
    if save_VTK == True:
        inlet_grid = pv.StructuredGrid(surface_grid[:,:,0], 
            surface_grid[:,:,1], surface_grid[:,:,2])
        inlet_grid.save(file_name_VTK + '.vtk')

    return surface_grid

def shock_surface(design_vals):
    puffin_dir = design_vals['puffin_dir']
    job_name = design_vals['job_name']
    n_r = design_vals.get('n_r', 20)
    n_phi = design_vals.get('n_phi', 20)
    p_rat_shock = design_vals.get('p_rat_shock', 2.0)
    dt = design_vals.get('dt', 1.0E-6)
    max_step = design_vals.get('max_step', 100000)

    # change directory to access puffin data
    current_dir = os.getcwd()
    os.chdir(puffin_dir + '/' + job_name)

    # extract number of y cells from JSON file
    f = open('config.json')
    config_data = json.load(f)
    n_cells = config_data['ncells_0']
    f.close()

    # read flow data
    flow_data_name = 'flow-0.data'
    stream_data, variable_names = read_stream_flow_file(flow_data_name, n_cells)
    max_y = stream_data[0][-1][1]
    dy = max_y - stream_data[0][-2][1]

    # return back to working directory
    os.chdir(puffin_dir)

    # calculate r and phi values
    rs = np.linspace(0+dy, max_y-dy, n_r)
    phis = np.linspace(pi, 3*pi, n_phi)
    surface_grid = np.nan * np.ones((n_phi, n_r, 3))
    shock_coords = np.nan * np.ones((n_r, 3))
    for j in range(n_r):
        locus_point = [0.0, rs[j]]
        stream_i_data = stream_trace(locus_point, job_name, 
                p_rat_shock=p_rat_shock, dt=dt, shock_data_only=True, 
                max_step=max_step)[0]
        shock_coords[j][0] = 0.0
        shock_coords[j][1] = stream_i_data[1]
        shock_coords[j][2] = stream_i_data[0]
    
    for i in range(n_phi):
        for j in range(n_r):
            x = shock_coords[j][0]
            y = shock_coords[j][1]
            z = shock_coords[j][2]

            surface_grid[i][j][0] = x*cos(phis[i]) - y*sin(phis[i])
            surface_grid[i][j][1] = x*sin(phis[i]) + y*cos(phis[i])
            surface_grid[i][j][2] = z

    os.chdir(current_dir)
    """
    surface_grid = pv.StructuredGrid(surface_grid[:,:,0], surface_grid[:,:,1], 
        surface_grid[:,:,2])
    surface_grid.save('inlet_shock.vtk')
    """
    return surface_grid