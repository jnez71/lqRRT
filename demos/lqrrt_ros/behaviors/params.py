"""
Parameters that are the same for all behaviors.

"""
from __future__ import division
import numpy as np

################################################# DIMENSIONALITY

nstates = 6
ncontrols = 3

################################################# BEHAVIOR CONTROL

real_tol = [0.5, 0.5, np.deg2rad(10), np.inf, np.inf, np.inf]
pointshoot_tol = np.deg2rad(20)  # rad
free_radius = 6  # m
basic_duration = 1  # s
fudge_factor = 0.85

################################################# ISSUE CONTROL

stuck_threshold = 2  # moves
fail_threshold = 5  # stucks
collision_threshold = 0.3  # s

################################################# TREE GROWTH

horizon = 0  # s, 0 == adaptive
dt = 0.1  # s
FPR = 0
ss_start = 10  # m
ss_step = 5  # m
max_nodes = 1E5

################################################# INERTIA

m = 350  # kg
I = 400  # kg*m^2
invM = np.array([1/m, 1/m, 1/I])

################################################# SPEED AND THRUST LIMITS

velmax_pos = np.array([1.2, 0.6, 0.22])  # (m/s, m/s, rad/s), body-frame forward
velmax_neg = np.array([-0.6, -0.6, -0.22])  # (m/s, m/s, rad/s), body-frame backward
thrust_max = np.array([220, 220, 220, 220])  # N, per thruster

################################################# THRUSTER CONFIGURATION

# Thruster layout, [back-left, back-right, front-left front-right] (m)
thruster_positions = np.array([[-1.9000,  1.0000, -0.0123],
                               [-1.9000, -1.0000, -0.0123],
                               [ 1.6000,  0.6000, -0.0123],
                               [ 1.6000, -0.6000, -0.0123]])

# Thruster heading vectors, [back-left, back-right, front-left front-right] (m)
thruster_directions = np.array([[ 0.7071,  0.7071,  0.0000],
                                [ 0.7071, -0.7071,  0.0000],
                                [ 0.7071, -0.7071,  0.0000],
                                [ 0.7071,  0.7071,  0.0000]])

# Fresh sprinkles
thrust_levers = np.cross(thruster_positions, thruster_directions)
B = np.concatenate((thruster_directions.T, thrust_levers.T))[[0, 1, 5]]
invB = np.linalg.pinv(B)

################################################# EFFECTIVE DRAG

Fx_max = B.dot(thrust_max * [1, 1, 1, 1])[0]
Fy_max = B.dot(thrust_max * [1, -1, -1, 1])[1]
Mz_max = B.dot(thrust_max * [-1, 1, -1, 1])[2]
D_pos = np.abs([Fx_max, Fy_max, Mz_max] / velmax_pos)
D_neg = np.abs([Fx_max, Fy_max, Mz_max] / velmax_neg)

################################################# GEOMETRY

# Boat shape
boat_length = 210 * 0.0254  # m
boat_width = 96 * 0.0254  # m
boat_buffer = 0.15  # m

# Grid of points defining boat
vps_spacing = 0.1  # m
vps_grid_x, vps_grid_y = np.mgrid[slice(-(boat_length+boat_buffer)/2, (boat_length+boat_buffer)/2+vps_spacing, vps_spacing),
                                  slice(-(boat_width+boat_buffer)/2, (boat_width+boat_buffer)/2+vps_spacing, vps_spacing)]
vps_grid_x = vps_grid_x.reshape(vps_grid_x.size)
vps_grid_y = vps_grid_y.reshape(vps_grid_y.size)
vps = np.zeros((vps_grid_x.size, 2))
for i in range(len(vps)):
    vps[i] = [vps_grid_x[i], vps_grid_y[i]]
vps = vps.T

################################################# MISC

def unset(*args):
    raise AttributeError("This function has not been set yet!")
