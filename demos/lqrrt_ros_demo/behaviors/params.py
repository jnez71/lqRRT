"""
Parameters that are the same for all planners.

"""
from __future__ import division
import numpy as np

#################################################

# Dimensionality
nstates = 6
ncontrols = 3

#################################################

# Tree growth
ss_buff = 10  # m
horizon = 2  # s
dt = 0.1  # s
max_nodes = 1E5
real_tol = [0.1, 0.1, np.deg2rad(5), 0.1, 0.1, np.deg2rad(5)]

#################################################

# Speed limits for the planner to use (so we don't move too fast for controller tracking)
velmax_pos_plan = np.array([1.1, 0.4, 0.2])  # (m/s, m/s, rad/s), body-frame forward
velmax_neg_plan = np.array([-0.7, -0.4, -0.2])  # (m/s, m/s, rad/s), body-frame backward

#################################################

# Mass and yaw inertia
m = 350  # kg
I = 400  # kg*m^2
invM = np.array([1/m, 1/m, 1/I])

#################################################

# Physical top speeds and thrusts
# velmax_pos = np.copy(velmax_pos_plan)
# velmax_neg = np.copy(velmax_neg_plan)
velmax_pos = np.array([2, 0.8, 0.5])  # (m/s, m/s, rad/s), body-frame forward
velmax_neg = np.array([-0.8, -0.8, -0.5])  # (m/s, m/s, rad/s), body-frame backward
thrust_max = np.array([220, 220, 220, 220])  # N, per thruster

#################################################

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

#################################################

# Effective linear drag coefficients given thrust and speed limits
Fx_max = B.dot(thrust_max * [1, 1, 1, 1])[0]
Fy_max = B.dot(thrust_max * [1, -1, -1, 1])[1]
Mz_max = B.dot(thrust_max * [-1, 1, -1, 1])[2]
D_pos = np.abs([Fx_max, Fy_max, Mz_max] / velmax_pos)
D_neg = np.abs([Fx_max, Fy_max, Mz_max] / velmax_neg)

#################################################

# Boat shape
boat_length = 210 * 0.0254  # m
boat_width = 96 * 0.0254  # m
boat_buffer = 0.15  # m

# Grid of points defining boat
vps_spacing = 1.0  # m
vps_grid_x, vps_grid_y = np.mgrid[slice(-(boat_length+boat_buffer)/2, (boat_length+boat_buffer)/2+vps_spacing, vps_spacing),
								  slice(-(boat_width+boat_buffer)/2, (boat_width+boat_buffer)/2+vps_spacing, vps_spacing)]
vps_grid_x = vps_grid_x.reshape(vps_grid_x.size)
vps_grid_y = vps_grid_y.reshape(vps_grid_y.size)
vps = np.zeros((vps_grid_x.size, 2))
for i in range(len(vps)):
	vps[i] = [vps_grid_x[i], vps_grid_y[i]]
vps = vps.T

#################################################

def erf(xgoal, x):
	"""
	Returns error e given two states xgoal and x.
	Angle differences are taken properly on SO3.

	"""
	e = xgoal - x
	c = np.cos(x[2])
	s = np.sin(x[2])
	cg = np.cos(xgoal[2])
	sg = np.sin(xgoal[2])
	e[2] = np.arctan2(sg*c - cg*s, cg*c + sg*s)
	return e

#################################################

def unset(*args):
	raise AttributeError("This function has not been set yet!")
