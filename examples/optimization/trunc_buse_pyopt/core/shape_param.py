"""
Parameterize contour using free-form deformation.

Author: Reece Otto 20/05/2022
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from csgen.file_io import coords_to_csv
from nurbskit.ffd_utils import auto_hull_2D
from nurbskit.visualisation import surf_plot_2D
from nurbskit.point_inversion import point_inv_surf

# import truncated Busemann contour
core_dir = os.getcwd()
main_dir = os.path.dirname(core_dir)
sol_0_dir = main_dir + '/sol_0'
os.chdir(sol_0_dir)
contour_coords = []
with open('contour_0.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    next(csv_reader)
    for row in csv_reader:
    	contour_coords.append([float(row[0]), float(row[1])])
contour_coords = np.array(contour_coords)

# construct FFD hull
N_Pu = 5; N_Pv = 4; p = 3
hull = auto_hull_2D(contour_coords, N_Pu, N_Pv, p=p).cast_to_nurbs_surface()

# plot contour and hull
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 16
    })
ax = surf_plot_2D(hull, show_knots=False)
ax.plot(contour_coords[:,0], contour_coords[:,1], color='k', linestyle='-', 
    label='Busemann Contour')
ax.set_aspect('equal', adjustable="datalim")
plt.legend()
fig.savefig('geom_param.svg', bbox_inches='tight')

# run point inversion routine
print('Running point inversion routine...')
contour_params = np.zeros((len(contour_coords), 2))
for i in range(len(contour_coords)):
    contour_params[i] = point_inv_surf(hull, contour_coords[i], tol=1E-6)
print('Done.')

# export contour params
os.chdir(main_dir)
coords_to_csv(contour_params, file_name='contour_params', header=['u', 'v'])
hull.write_to_csv(file_name='ffd_hull_0')