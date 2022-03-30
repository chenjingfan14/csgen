"""
Streamline-tracing tools.

Author: Reece Otto 08/11/2021
"""
from nurbskit.path import BSpline
from nurbskit.spline_fitting import global_curve_interp
from nurbskit.point_inversion import point_proj_path
from math import pi, sqrt, sin, cos, tan, atan2, atan, acos
import numpy as np
from scipy import optimize, interpolate
import csv

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

    def scale(self, scale_factor):
        polar_coords = np.copy(self.polar_coords)
        for i in range(len(polar_coords)):
            polar_coords[i][0] = self.rs[i] * scale_factor

        return Streamline(polar_coords=polar_coords)

    def translate(self, x_shift=0.0, y_shift=0.0, z_shift=0.0):
        xyz_shifted = np.copy(self.xyz_coords)
        for i in range(len(xyz_shifted)):
            x = self.xs[i] + x_shift
            y = self.ys[i] + y_shift
            z = self.zs[i] + z_shift
            xyz_shifted[i] = [x, y, z]
        return Streamline(xyz_coords=xyz_shifted)

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

def waverider_stream_trace(base_coords, stream_coords, z_base, n_z, tol=1E-4):
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

    return wr_coords