# A truncated Busemann diffuser
# Author: Rowan J Gollan
# Date: 2022-02-08
#

config.axisymmetric = True
L_ext = 1.0

init_gas_model('ideal-air-gas-model.lua')
gas1 = GasState(config.gmodel)
gas1.p = 2374 # Pa
gas1.T = 376.1 # K
gas1.update_thermo_from_pT()
gas1.update_sound_speed()
M1 = 7.74
V1 = M1 * gas1.a

config.max_step_relax = 40

import csv
xs = []; ys = []
with open('trunc-diffuser-scaled.path', 'r') as df:
    pt_data = csv.reader(df, delimiter=' ', skipinitialspace=True)
    for row in pt_data:
        if row[0] == '#': continue
        xs.append(float(row[0]))
        ys.append(float(row[1]))

x_thrt = xs[-1]
y_thrt = ys[-1]

from eilmer.spline import CubicSpline
busemann_contour = CubicSpline(xs, ys)
def upper_y(x):
    return busemann_contour(x) if x < x_thrt else y_thrt
def lower_y(x): return 0.0
def lower_bc(x): return 0
def upper_bc(x): return 0

config.max_x = x_thrt + L_ext
config.dx = config.max_x/1000

st1 = StreamTube(gas=gas1, velx=V1, vely=0.0,
                 y0=lower_y, y1=upper_y,
                 bc0=lower_bc, bc1=upper_bc,
                 ncells=75)


