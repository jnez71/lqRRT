"""
Demo of using the lqrrt planner for boats.
Criteria is to get to some goal without crashing and
while maintaining a nice, smooth, car-like path. Also,
we account for the boat's actual shape so it can squeeze
through tight spaces depending on its heading.

State:   [x, y, h, vx, vy, vh]  (m, m, rad, m/s, m/s, rad/s)
Effort:  [ux, uy, uh]           (N, N, N*m)

Pose states are world-frame.
Twist states and wrench are body-frame.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl
import lqrrt

################################################# PHYSICAL PARAMETERS

# Experimentally determined mass and inertia
m = 500  # kg
I = 500  # kg/m^2
invM = np.array([1/m, 1/m, 1/I])

# Experimentally determined top speeds of the boat
velmax_pos = [1.1, 0.45, 0.2]  # (m/s, m/s, rad/s), body-frame forward
velmax_neg = [0.68, 0.45, 0.2]  # (m/s, m/s, rad/s), body-frame backward

# Maximum wrench, with severly degraded body y thrust
thrust_max = 220  # N
thrust_lever = 2.15  # m
u_max = np.array([2*np.sqrt(2)*thrust_max, 0.2*np.sqrt(2)*thrust_max, 4*thrust_lever*thrust_max])  # (N, N, N*m)

# Effective linear drag coefficients given wrench and speed limits
D_pos = np.abs(u_max / velmax_pos)
D_neg = np.abs(u_max / velmax_neg)

################################################# DYNAMICS

nstates = 6
ncontrols = 3

def dynamics(x, u, dt):
	"""
	Returns next state given last state x, wrench u, and timestep dt.
	Simple holonomic boat-like dynamics, with some "force of nature"
	keeping the boat looking along its velocity line.

	"""
	# Rotation matrix (orientation, converts body to world)
	R = np.array([
				  [np.cos(x[2]), -np.sin(x[2]), 0],
				  [np.sin(x[2]),  np.cos(x[2]), 0],
				  [           0,             0, 1]
				])

	# Construct drag coefficients based on our motion signs
	D = np.zeros(3)
	for i, v in enumerate(x[3:]):
		if v >= 0:
			D[i] = D_pos[i]
		else:
			D[i] = D_neg[i]

	# Heading controller trying to keep us car-like
	vw = R[:2, :2].dot(x[3:5])
	ang = np.arctan2(vw[1], vw[0])
	c = np.cos(x[2])
	s = np.sin(x[2])
	cg = np.cos(ang)
	sg = np.sin(ang)
	u[2] = u[2] + 5000*np.arctan2(sg*c - cg*s, cg*c + sg*s)

	# Actuator saturation
	for i, mag in enumerate(np.abs(u)):
		if mag > u_max[i]:
			u[i] = u_max[i] * np.sign(u[i])

	# M*vdot + D*v = u  and  pdot = R*v
	xdot = np.concatenate((R.dot(x[3:]), invM*(u - D*x[3:])))

	# First-order integrate
	xnext = x + xdot*dt

	# Impose not turning in place
	if x[3] > 0:
		xnext[5] = np.clip(np.abs(xnext[3]/velmax_pos[0]), 0, 1) * xnext[5]
	elif x[3] < 0:
		xnext[5] = np.clip(np.abs(xnext[3]/velmax_neg[0]), 0, 1) * xnext[5]

	# Impose not driving backwards
	if xnext[3] < 0:
		xnext[3] = 0

	return xnext

################################################# VEHICLE DIMENSIONS

# Boat shape and resolution
boat_length = 6  # m
boat_width = 3  # m
boat_buffer = 2  # m
vps_spacing = 0.5  # m

# Grid of points defining boat
vps_grid_x, vps_grid_y = np.mgrid[slice(-(boat_length+boat_buffer)/2, (boat_length+boat_buffer)/2+vps_spacing, vps_spacing),
								  slice(-(boat_width+boat_buffer)/2, (boat_width+boat_buffer)/2+vps_spacing, vps_spacing)]
vps_grid_x = vps_grid_x.reshape(vps_grid_x.size)
vps_grid_y = vps_grid_y.reshape(vps_grid_y.size)
vps = np.zeros((vps_grid_x.size, 2))
for i in range(len(vps)):
	vps[i] = [vps_grid_x[i], vps_grid_y[i]]
vps = vps.T

################################################# CONTROL POLICY

# Body-frame gains
# (let the dynamic's heading controller take care of heading)
kp = np.diag([120, 120, 0])
kd = np.diag([120, 120, 0])

def lqr(x, u):
	"""
	Returns cost-to-go matrix S and policy matrix K given local state x and effort u.

	"""
	R = np.array([
				  [np.cos(x[2]), -np.sin(x[2]), 0],
				  [np.sin(x[2]),  np.cos(x[2]), 0],
				  [           0,             0, 1]
				])

	S = np.diag([1, 1, 1, 1, 1, 1])
	K = np.hstack((kp.dot(R.T), kd))

	return (S, K)


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

################################################# OBJECTIVES

# Initial condition and goal
x0 = np.array([0, 0, np.deg2rad(0), 0, 0, 0])
goal = [40, 40, np.deg2rad(90), 0, 0, 0]
goal_buffer = [8, 8, np.inf, np.inf, np.inf, np.inf]
error_tol = np.copy(goal_buffer)/8

################################################# CONSTRAINTS

obs_choice = 'some'

# No obstacles
if obs_choice == 'none':
	obs = []

# Some obstacles [x, y, radius]
elif obs_choice == 'some':
	obs = np.array([[20, 20, 5],
					[10, 30, 2],
					[40, 10, 3]
				  ])

# Noised grid of obstacles
elif obs_choice == 'grid':
	obs_spacing = 12  # m
	obs_range = (5, 60)
	obs_grid_x, obs_grid_y = np.mgrid[slice(obs_range[0], obs_range[1]+obs_spacing, obs_spacing),
									  slice(obs_range[0], obs_range[1]+obs_spacing, obs_spacing)]
	obs_grid_x = obs_grid_x.reshape(obs_grid_x.size)
	obs_grid_y = obs_grid_y.reshape(obs_grid_y.size)
	obs = [-9999*np.ones(3)] * obs_grid_x.size
	for i in range(len(obs)):
		p = np.round([obs_grid_x[i], obs_grid_y[i]] + 3*(np.random.rand(2)-0.5), 2)
		if npl.norm(p - goal[:2]) > 2*boat_length and npl.norm(np.array(p - x0[:2])) > 2*boat_length:
			obs[i] = np.concatenate((p, [1]))

####

# Definition of collision
def is_feasible(x, u):
	# Body to world
	R = np.array([
				  [np.cos(x[2]), -np.sin(x[2])],
				  [np.sin(x[2]),  np.cos(x[2])],
				])
	# Boat vertices in world frame
	verts = x[:2] + np.vstack((R.dot(vps).T, x[:2]))
	# Check for collisions over all obstacles
	for ob in obs:
		if np.any(npl.norm(verts - ob[:2], axis=1) <= ob[2]):
			return False
	return True

################################################# HEURISTICS

sample_space = [(x0[0], goal[0]),
				(x0[1], goal[1]),
				(0, 0),
				(0.9*velmax_pos[0], velmax_pos[0]),
				(-velmax_neg[1], velmax_pos[1]),
				(-velmax_neg[2], velmax_pos[2])]

goal_bias = [0.2, 0.2, 0, 0, 0, 0]

xrand_gen = None

################################################# PLAN

constraints = lqrrt.Constraints(nstates=nstates, ncontrols=ncontrols,
								goal_buffer=goal_buffer, is_feasible=is_feasible)

planner = lqrrt.Planner(dynamics, lqr, constraints,
						horizon=2, dt=0.1, FPR=0.5,
						error_tol=error_tol, erf=erf,
						min_time=2, max_time=3, max_nodes=1E5,
						goal0=goal)

planner.update_plan(x0, sample_space, goal_bias=goal_bias,
					xrand_gen=xrand_gen, finish_on_goal=False)

################################################# SIMULATION

# Prepare "real" domain
dt = 0.03  # s
T = planner.T  # s
t_arr = np.arange(0, T, dt)
framerate = 10

# Preallocate results memory
x = np.copy(x0)
x_history = np.zeros((len(t_arr), nstates))
xref_history = np.zeros((len(t_arr), nstates))
goal_history = np.zeros((len(t_arr), nstates))
u_history = np.zeros((len(t_arr), ncontrols))

# Give back real control
kp = np.diag([120, 120, 200])
kd = np.diag([120, 120, 10])

# Track plan
for i, t in enumerate(t_arr):

	# Planner's decision
	xref = planner.get_state(t)
	uref = planner.get_effort(t)

	# Controller's decision
	u = 2*lqr(x, uref)[1].dot(erf(xref, np.copy(x)))

	# Record this instant
	x_history[i, :] = x
	xref_history[i, :] = xref
	goal_history[i, :] = goal
	u_history[i, :] = u

	# Step dynamics
	x = dynamics(x, u, dt)

################################################# VISUALIZATION

print("\n...plotting...")
from matplotlib import pyplot as plt
import matplotlib.animation as ani

# Plot results
fig1 = plt.figure()
fig1.suptitle('Results', fontsize=20)

ax1 = fig1.add_subplot(2, 4, 1)
ax1.set_ylabel('X Position (m)', fontsize=16)
ax1.plot(t_arr, x_history[:, 0], 'k',
		 t_arr, goal_history[:, 0], 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 2)
ax1.set_ylabel('Y Position (m)', fontsize=16)
ax1.plot(t_arr, x_history[:, 1], 'k',
		 t_arr, goal_history[:, 1], 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 3)
ax1.set_ylabel('Heading (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(x_history[:, 2]), 'k',
		 t_arr, np.rad2deg(goal_history[:, 2]), 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 4)
ax1.set_ylabel('Efforts (N, N*m)', fontsize=16)
ax1.plot(t_arr, u_history[:, 0], 'b',
		 t_arr, u_history[:, 1], 'g',
		 t_arr, u_history[:, 2], 'r')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 5)
ax1.set_ylabel('X Velocity (m/s)', fontsize=16)
ax1.plot(t_arr, x_history[:, 3], 'k',
		 t_arr, goal_history[:, 3], 'g--')
ax1.grid(True)
ax1.set_xlabel('Time (s)')

ax1 = fig1.add_subplot(2, 4, 6)
ax1.set_ylabel('Y Velocity (m/s)', fontsize=16)
ax1.plot(t_arr, x_history[:, 4], 'k',
		 t_arr, goal_history[:, 4], 'g--')
ax1.grid(True)
ax1.set_xlabel('Time (s)')

ax1 = fig1.add_subplot(2, 4, 7)
ax1.set_ylabel('Yaw Rate (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(x_history[:, 5]), 'k',
		 t_arr, np.rad2deg(goal_history[:, 5]), 'g--')
ax1.grid(True)
ax1.set_xlabel('Time (s)')

ax1 = fig1.add_subplot(2, 4, 8)
dx = 0; dy = 1
ax1.set_xlabel('- State {} +'.format(dx))
ax1.set_ylabel('- State {} +'.format(dy))
ax1.grid(True)
for ID in xrange(planner.tree.size):
	x_seq = np.array(planner.tree.x_seq[ID])
	if ID in planner.node_seq:
		ax1.plot((x_seq[:, dx]), (x_seq[:, dy]), color='r', zorder=2)
	else:
		ax1.plot((x_seq[:, dx]), (x_seq[:, dy]), color='0.75', zorder=1)
ax1.scatter(planner.tree.state[0, dx], planner.tree.state[0, dy], color='b', s=48)
ax1.scatter(planner.tree.state[planner.node_seq[-1], dx], planner.tree.state[planner.node_seq[-1], dy], color='r', s=48)
ax1.scatter(goal[dx], goal[dy], color='g', s=48)
for ob in obs:
	ax1.add_patch(plt.Circle((ob[0], ob[1]), radius=ob[2], fc='r'))

print("\nClose the plot window to continue to animation.")
plt.show()


# Animation
fig2 = plt.figure()
fig2.suptitle('Evolution', fontsize=24)
plt.axis('equal')

ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_xlabel('- X Position +')
ax2.set_ylabel('- Y Position +')
ax2.grid(True)

radius = boat_width/2
xlim = (min(x_history[:, 0])*1.1 - radius, max(goal_history[:, 0])*1.1 + radius)
ylim = (min(x_history[:, 1])*1.1 - radius, max(goal_history[:, 1])*1.1 + radius)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

dx = 0; dy = 1
ax2.set_xlabel('- State {} +'.format(dx))
ax2.set_ylabel('- State {} +'.format(dy))
ax2.grid(True)
for ID in xrange(planner.tree.size):
	x_seq = np.array(planner.tree.x_seq[ID])
	if ID in planner.node_seq:
		ax2.plot((x_seq[:, dx]), (x_seq[:, dy]), color='r', zorder=2)
	else:
		ax2.plot((x_seq[:, dx]), (x_seq[:, dy]), color='0.75', zorder=1)
ax2.scatter(planner.tree.state[0, dx], planner.tree.state[0, dy], color='b', s=48)
ax2.scatter(planner.tree.state[planner.node_seq[-1], dx], planner.tree.state[planner.node_seq[-1], dy], color='r', s=48)

td = boat_length/3/2 + 0.5
graphic_robot_1 = ax2.add_patch(plt.Circle(((x_history[0, 0]+td*np.cos(x_history[0, 2]), x_history[0, 1]+td*np.sin(x_history[0, 2]))), radius=radius, fc='k'))
graphic_robot_2 = ax2.add_patch(plt.Circle((x_history[0, 0], x_history[0, 1]), radius=radius, fc='k'))
graphic_robot_3 = ax2.add_patch(plt.Circle(((x_history[0, 0]-td*np.cos(x_history[0, 2]), x_history[0, 1]-td*np.sin(x_history[0, 2]))), radius=radius, fc='k'))

llen = boat_length/2
graphic_goal = ax2.add_patch(plt.Circle((goal_history[0, 0], goal_history[0, 1]), radius=npl.norm([goal_buffer[0], goal_buffer[1]]), color='g', alpha=0.1))
graphic_goal_heading = ax2.plot([goal_history[0, 0] - 0.5*llen*np.cos(goal_history[0, 2]), goal_history[0, 0] + 0.5*llen*np.cos(goal_history[0, 2])],
								[goal_history[0, 1] - 0.5*llen*np.sin(goal_history[0, 2]), goal_history[0, 1] + 0.5*llen*np.sin(goal_history[0, 2])], color='g', linewidth=5)

for ob in obs:
	ax2.add_patch(plt.Circle((ob[0], ob[1]), radius=ob[2], fc='r'))

def ani_update(arg, ii=[0]):

	i = ii[0]  # don't ask...

	if np.isclose(t_arr[i], np.around(t_arr[i], 1)):
		fig2.suptitle('Evolution (Time: {})'.format(t_arr[i]), fontsize=24)

	graphic_robot_1.center = ((x_history[i, 0]+td*np.cos(x_history[i, 2]), x_history[i, 1]+td*np.sin(x_history[i, 2])))
	graphic_robot_2.center = ((x_history[i, 0], x_history[i, 1]))
	graphic_robot_3.center = ((x_history[i, 0]-td*np.cos(x_history[i, 2]), x_history[i, 1]-td*np.sin(x_history[i, 2])))

	ii[0] += int(1 / (dt * framerate))
	if ii[0] >= len(t_arr):
		print("Resetting animation!")
		ii[0] = 0

	return None

# Run animation
print("\nStarting animation. \nBlack: robot \nRed: obstacles \nGreen: goal\n")
animation = ani.FuncAnimation(fig2, func=ani_update, interval=dt*1000)
plt.show()
