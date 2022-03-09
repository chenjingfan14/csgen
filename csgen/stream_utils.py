"""
Streamline-tracing tools.

Author: Reece Otto 08/11/2021
"""
from nurbskit.transform import polar
from nurbskit.path import BSpline
from nurbskit.spline_fitting import global_curve_interp
from nurbskit.point_inversion import point_proj_path
from math import pi, sqrt, sin, cos, tan, atan2, atan
import numpy as np
from scipy import optimize

class Streamline():
    def __init__(self, xs, ys, zs, **kwargs):
        # check if rs, thetas and phis have same length
        if len(xs) != len(ys):
            raise AttributeError('The x and y arrays have different length.')
        if len(xs) != len(zs)
            raise AttributeError('The x and z arrays have different length.')

        # assign Cartesian coordinates to object
        self.xs = xs
        self.ys = ys
        self.zs = zs

    def scale(x_scale, y_scale, z_scale):
        for i in len(self.thetas):
            for j in len(self.phis):
                self.xs[i] *= x_scale
                self.ys[i] *= y_scale
                self.zs[i] *= z_scale

    def translate(x_shift, y_shift, z_shift):
        for i in len(self.thetas):
            for j in len(self.phis):
                self.xs[i] += x_shift
                self.ys[i] += y_shift
                self.zs[i] += z_shift

    def rotate_xyz(roll, pitch, yaw, x_origin=0, y_origin=0, z_origin=0):
        # define rotation matrices
        def R_roll(roll):
            return np.array([[1.0, 0.0, 0.0],
                             [0.0, cos(roll), -sin(roll)],
                             [0.0, sin(roll), cos(roll)]])

        def R_pitch(pitch):
            return np.array([[cos(pitch), 0.0, sin(pitch)],
                             [0.0, 1.0, 0.0],
                             [-sin(pitch), 0.0, cos(roll)]])

        def R_yaw(yaw):
            return np.array([[cos(yaw), -sin(yaw), 0.0],
                             [sin(yaw), cos(yaw), 0.0],
                             [0.0, 0.0, 1.0]])

        # translate streamline to new coordinate frame that has desired rotation
        # point as the origin
        stream_trans = self.translate(-x_origin, -y_origin, -z_origin)
        stream_xs = stream_trans.xs
        stream_ys = stream_trans.ys
        stream_zs = stream_trans.zs

        # rotate the translated streamline
        for i in range(len(stream_trans.xs)):
            # construct position vector for point i
            coords_i = np.array([stream_xs[i], stream_ys[i], stream_zs[i]])
            
            # construct rotation matrix
            rot_zy = np.matmul(R_yaw(yaw), R_pitch(pitch))
            rot_zyx = np.matmul(rot_zy, R_roll(roll))

            # rotate streamline and save coordinates
            coord_rot = np.matmul(rot_zyx, coords_i)
            stream_trans.xs[i] = coord_rot[0]
            stream_trans.ys[i] = coord_rot[1]
            stream_trans.zs[i] = coord_rot[2]

        # translate streamline back to original coordinate frame
        stream_trans = self.translate(x_origin, y_origin, z_origin)

        # replace object's coordinates with rotated coordinates
        self.xs = stream_trans.xs
        self.ys = stream_trans.ys
        self.zs = stream_trans.zs

class PolarStreamline(Streamline):
    # Streamline defined by a given set of 3D polar coordinates.
    
    def __init__(self, rs, thetas, phis):
        # check if rs, thetas and phis have same length
        if len(rs) != len(thetas):
            raise AttributeError('Radius and theta arrays have different size.')
        if len(rs) != len(phis)
            raise AttributeError('Radius and phi arrays have different size.')

        # assign polar coordinates
        self.rs = rs
        self.thetas = thetas
        self.phis = phis

        # calculate Cartesian coordinates
        xs = np.nan * np.ones(len(self.thetas))
        ys = np.nan * np.ones(len(self.thetas))
        zs = np.nan * np.ones(len(self.thetas))
        for i in len(self.thetas):
            for j in len(self.phis):
                xs[i] = self.rs[i] * cos(self.phis[j]) * sin(self.thetas[i])
                ys[i] = self.rs[i] * sin(self.phis[j]) * sin(self.thetas[i])
                zs[i] = self.rs[i] * cos(self.thetas[j])
        
        # assign Cartesian coordinates to object
        self.xs = xs
        self.ys = ys
        self.zs = zs

def busemann_stream_trace(shape_coords, field, stream_coords, plane='capture'):
    if plane == 'capture':
        print('\nTracing capture shape through flow field.')
    elif plane == 'exit':
        print('\nTracing exit shape through flow field.')
    else:
        raise ValueError("Plane must either be 'capture' or 'exit'.")

    # extract coords of entrance or exit shape
    shape_xs, shape_ys = shape_coords[:,0], shape_coords[:,1]
    n_streams = len(shape_xs)
    
    # get phi and radial scaling factors for each streamline of shape
    shape_zs = np.nan * np.zeros(len(shape_xs))
    phis_shape = np.nan * np.zeros(len(shape_xs))
    sfs_shape = np.nan * np.zeros(len(shape_xs))
    for i in range(len(shape_xs)):
        phis_shape[i] = atan2(shape_ys[i], shape_xs[i])
        if plane == 'capture':
            shape_zs[i] = -sqrt((shape_xs[i]**2 + shape_ys[i]**2) / \
                (tan(field.mu)**2))
            r_shape = sqrt(shape_xs[i]**2 + shape_ys[i]**2 + shape_zs[i]**2)
            sfs_shape[i] = r_shape / \
            sqrt(stream_coords[-1][0]**2 + stream_coords[-1][1]**2)
        else:
            shape_zs[i] = sqrt((shape_xs[i]**2 + shape_ys[i]**2) / \
            (tan(field.thetas[0])**2))
            r_shape = sqrt(shape_xs[i]**2 + shape_ys[i]**2 + shape_zs[i]**2)
            sfs_shape[i] = r_shape / \
            sqrt(stream_coords[0][0]**2 + stream_coords[0][1]**2)
            
    # find Cartesian coordinates of each point along each streamline
    stream_coords = field.streamline()
    inlet_xs = np.zeros((len(shape_xs), len(stream_coords)))
    inlet_ys = np.zeros((len(shape_xs), len(stream_coords)))
    inlet_zs = np.zeros((len(shape_xs), len(stream_coords)))
    for i in range(len(shape_xs)):
        for j in range(len(stream_coords)):
            r_j = sqrt(stream_coords[j][0]**2 + stream_coords[j][1]**2)
            x_b_rot = r_j * sin(field.thetas[j]) * cos(phis_shape[i])
            y_b_rot = r_j * sin(field.thetas[j]) * sin(phis_shape[i])
            z_b_rot = r_j * cos(field.thetas[j])
            rxyz_b_rot = sqrt(x_b_rot**2 + y_b_rot**2 + z_b_rot**2)
            
            inlet_xs[i][j] = sfs_shape[i] * rxyz_b_rot * sin(field.thetas[j]) \
                             * cos(phis_shape[i])
            inlet_ys[i][j] = sfs_shape[i] * rxyz_b_rot * sin(field.thetas[j]) \
                             * sin(phis_shape[i])
            inlet_zs[i][j] = sfs_shape[i] * rxyz_b_rot * cos(field.thetas[j])
    
    return np.array([inlet_xs, inlet_ys, inlet_zs])

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