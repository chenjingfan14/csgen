"""
Tracing streamlines through truncated Busemann field.

Author: Reece Otto 25/03/2022
"""
from csgen.stream_trace import inlet_stream_trace
from csgen.inlet_utils import inlet_blend
from nurbskit.path import Ellipse, Rectangle
import pyvista as pv
import matplotlib.pyplot as plt

# specify streamline data parameters
job_name = 'trunc-bd'
flow_data_name = 'flow-0.data'
n_cells = 75

#------------------------------------------------------------------------------#
#                         Inlet defined by capture shape                       #
#------------------------------------------------------------------------------#
n_streams = 101
n_z = 100


width = 1.0
height = width / 2
centre_x = 0
centre_y = height/2 + 1 / 20
"""
cap_shape = Rectangle(width, height, centre_x, centre_y)
cap_coords = cap_shape.list_eval(n_points=n_streams)
"""



# plot capture shape on entrance of flow field
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = plt.axes()
"""
ax = path_plot_2D(cap_shape, show_control_points=False, show_control_net=False,
    show_knots=False, path_label='Capture Shape', path_style='r-')
"""
ax.scatter(cap_coords[:,0], cap_coords[:,1])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
#plt.legend()
plt.show()
fig.savefig('capture_shape.svg', bbox_inches='tight')

# run streamline tracer
inlet_A_surf = inlet_stream_trace(cap_coords, n_z, job_name, 
	                            flow_data_name, n_cells, p_buffer=1.8, 
	                            dt=1.0E-5, max_step=100000)
inlet_A_grid = pv.StructuredGrid(inlet_A_surf[:,:,0], inlet_A_surf[:,:,1], 
	                             inlet_A_surf[:,:,2])
inlet_A_grid.save("inlet_A.vtk")

#------------------------------------------------------------------------------#
#                        Inlet defined by exit shape                           #
#------------------------------------------------------------------------------#
# generate capture shape
a_cap = 2 * 0.25
b_cap = 1 * 0.25
h_cap = 0
#k_cap = 1.5 * b_cap + centre_y
k_cap = centre_y
cap_shape = Ellipse(a_cap, b_cap, h_cap, k_cap)
cap_coords = cap_shape.list_eval(n_points=n_streams)

# plot capture shape on entrance of flow field
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 20
        })
ax = plt.axes()

"""
ax = path_plot_2D(cap_shape, show_control_points=False, show_control_net=False,
    show_knots=False, path_label='Capture Shape', path_style='r-')
"""

ax.scatter(cap_coords[:,0], cap_coords[:,1])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.axis('equal')
plt.grid()
#plt.legend()
plt.show()
fig.savefig('exit_shape.svg', bbox_inches='tight')

# run streamline tracer
inlet_B_surf = inlet_stream_trace(cap_coords, n_z, job_name, 
	                            flow_data_name, n_cells, p_buffer=1.8, 
	                            dt=1.0E-5, max_step=100000)
inlet_B_grid = pv.StructuredGrid(inlet_B_surf[:,:,0], inlet_B_surf[:,:,1], 
	inlet_B_surf[:,:,2])
inlet_B_grid.save("inlet_B.vtk")

#------------------------------------------------------------------------------#
#                               Blended inlet                                  #
#------------------------------------------------------------------------------#
inlet_C_coords = inlet_blend(inlet_A_surf, inlet_B_surf, 2.5)
inlet_grid_C = pv.StructuredGrid(inlet_C_coords[0], inlet_C_coords[1], 
        inlet_C_coords[2])
inlet_grid_C.save("inlet_C.vtk")
