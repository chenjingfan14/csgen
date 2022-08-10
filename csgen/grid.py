"""
Grid creation tools.

Author: Reece Otto 02/05/2022
"""
import numpy as np
import csv
from math import isclose, sqrt, atan2, sin, cos
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter

#------------------------------------------------------------------------------#
#                                  Paths                                       #
#------------------------------------------------------------------------------#
class CircularArc():
    # creates parametric circular arc on x-y plane from point_0 to point_1
    def __init__(self, point_0, point_1):
        # check if point_0 and point_1 have same number of coords
        if len(point_0) == len(point_1):
            self.dim = len(point_0)
        else:
            raise AssertionError(f'The given points do not have the same '
                'number of coordinates.')

        # check if point_0 and point_1 exist on same circle
        rad_0 = sqrt(point_0[0]**2 + point_0[1]**2)
        rad_1 = sqrt(point_1[0]**2 + point_1[1]**2)

        if isclose(rad_0, rad_1):
            self.point_0 = point_0
            self.point_1 = point_1
            self.radius = rad_0
        else:
            raise AssertionError('The given points do no exist on the same x-y ' 
                f'circle. Radius 0: {rad_0:.4}. Radius 1: {rad_1:.4}.')

    def __call__(self, t):
        phi_0 = atan2(self.point_0[1], self.point_0[0])
        phi_1 = atan2(self.point_1[1], self.point_1[0])
        phi = (phi_1 - phi_0)*t + phi_0
        x = self.radius * cos(phi)
        y = self.radius * sin(phi)
        if self.dim == 2:
            return np.array([x, y])
        if self.dim == 3:
            return np.array([x, y, self.point_0[2]])

    def list_eval(self, u_i=0, u_f=1, n_points=100):
        """
        Evalutes points along a path
        
        Keyword arguments:
            u_i = initial u coordinate
            u_f = final u coordinate
            n_points = number of point evaluated along path
        """
        u_vals = np.linspace(u_i, u_f, n_points)
        coords = np.nan * np.ones((n_points, 3))
        for i in range(n_points):
            coords[i] = self(u_vals[i])
        return coords

#------------------------------------------------------------------------------#
#                                   Grids                                      #
#------------------------------------------------------------------------------#
class StructuredGrid():
    def __init__(self, point_array, point_data=None):
        # check if point_array has valid dimensions
        if len(point_array.shape) > 4:
            raise AssertionError('Point array has invalid shape.')

        self.point_array = point_array
        self.dimensions = point_array.shape
        self.point_data = point_data

    def export_to_vtk_xml(self, filename='s_grid'):
        n_i = len(self.point_array)
        n_j = len(self.point_array[0])
        if len(self.dimensions) == 4:
            n_k = len(self.point_array[0][0])
        else:
            n_k = 1

        s_grid = vtkStructuredGrid()
        s_grid.SetDimensions([n_i, n_j, n_k])
        points = vtkPoints()
        points.Allocate(n_i*n_j*n_k)

        for k in range(n_k):
            k_offset = k*n_i*n_j
            for j in range(n_j):
                j_offset = j*n_i
                for i in range(n_i):
                    offset = i + j_offset + k_offset
                    if len(self.dimensions) == 3:
                        points.InsertPoint(offset, self.point_array[i][j])
                    elif len(self.dimensions) == 4:
                        points.InsertPoint(offset, self.point_array[i][j][k])

        s_grid.SetPoints(points)
        writer = vtkXMLStructuredGridWriter()
        writer.SetInputData(s_grid)
        writer.SetFileName(filename + '.vtu')
        writer.SetDataModeToAscii()
        writer.Update()

def import_dat(file_name):
    with open(file_name) as grid_file:
        raw_data = list(csv.reader(grid_file, delimiter=" "))
        n_blocks = int(raw_data[0][0])
        blocks = [None] * n_blocks
        line = 1
        for n in range(n_blocks):
            n_i = int(raw_data[line][0]); n_j = int(raw_data[line][1])
            block_i = np.array(raw_data[line+1:line+1+n_i*n_j], dtype=float)
            block_i = np.reshape(block_i, (n_i, n_j, 3))
            blocks[n] = block_i
            line += n_i*n_j + 1

    return blocks

#------------------------------------------------------------------------------#
#                                Surfaces                                      #
#------------------------------------------------------------------------------#
class CoonsPatch():
    def __init__(self, north, south, east, west):
        self.west = west
        self.north = north
        self.east = east
        self.south = south

    def __call__(self, s, t):
        west_to_east = (1 - t)*self.west(s) + t*self.east(s)
        south_to_north = (1 - s)*self.south(t) + s*self.north(t)
        corners = self.west(0)*(1 - s)*(1 - t) + self.west(1)*s*(1 - t) + \
                  self.east(0)*(1 - s)*t + self.east(1)*s*t
        return west_to_east + south_to_north - corners

    def grid_eval(self, s_i=0, s_f=1, t_i=0, t_f=1, n_s=100, n_t=100):
        ss = np.linspace(s_i, s_f, n_s)
        ts = np.linspace(t_i, t_f, n_t)
        grid = np.nan * np.ones((n_s, n_t, 3))
        for i in range(n_s):
            for j in range(n_t):
                grid[i][j] = self(ss[i], ts[j])
        
        return grid