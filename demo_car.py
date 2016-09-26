"""
Demo of using the lqrrt planner for (nonholonomic) cars.
We do this with a similar model to the heading-constrained
boat, but with an actual degenerate state-space (no vy).
This is a very simplified model that pretends our input
is a wrench instead of a steering-angle and throttle-angle,
but it suffices to give reasonable car motion. Who needs
turning radii anyway?

State:   [x, y, h, vx, vh]  (m, m, rad, m/s, rad/s)
Effort:  [ux, uh]           (N, N*m)

Pose states are world-frame.
Twist states and wrench are body-frame.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl
import lqrrt

################################################# PHYSICAL PARAMETERS

# Experimentally determined mass and inertia ("inertia" said liberally)
m = 500  # kg
I = 500  # kg/m^2
invM = np.array([1/m, 1/I])

# Top speeds ("rotational top speed" said liberally)
velmax = [1.1, 1]  # (m/s, rad/s), body-frame

# Maximum wrench
u_max = np.array([650, 1800])  # (N, N*m)

# Effective linear drag coefficients given wrench and speed limits
D = np.abs(u_max / velmax)

################################################# DYNAMICS

nstates = 5
ncontrols = 2

def dynamics(x, u, dt):
	"""
	Returns next state given last state x, wrench u, and timestep dt.
	Very car-like characteristics.

	"""
	# Velocity in world frame
	vwx = np.cos(x[2]) * x[3]
	vwy = np.sin(x[2]) * x[3]

	# Actuator saturation
	u = np.clip(u, [-u_max[0]/10, -u_max[1]], u_max)

	# M*vdot + D*v = u  and  pdot = R*v
	xdot = np.concatenate(([vwx, vwy, x[4]], invM*(u - D*x[3:])))

	# First order integrate
	xnext = x + xdot*dt

	# Impose that we can't drive backward
	if xnext[3] < 0:
		xnext[3] = 0

	# Impose not being able to turn in place
	xnext[4] = np.clip(np.abs(xnext[3]/velmax[0]), 0, 1) * xnext[4]

	return xnext

################################################# VEHICLE DIMENSIONS

# Boat shape and resolution
car_length = 6  # m
car_width = 3  # m
car_buffer = 2  # m
vps_spacing = 0.5  # m

# Grid of points defining car
vps_grid_x, vps_grid_y = np.mgrid[slice(-(car_length+car_buffer)/2, (car_length+car_buffer)/2+vps_spacing, vps_spacing),
								  slice(-(car_width+car_buffer)/2, (car_width+car_buffer)/2+vps_spacing, vps_spacing)]
vps_grid_x = vps_grid_x.reshape(vps_grid_x.size)
vps_grid_y = vps_grid_y.reshape(vps_grid_y.size)
vps = np.zeros((vps_grid_x.size, 2))
for i in range(len(vps)):
	vps[i] = [vps_grid_x[i], vps_grid_y[i]]
vps = vps.T

################################################# CONTROL POLICY

# Body-frame gains
kp = np.diag([120, 600])
kd = np.diag([120, 600])

def lqr(x, u):
	"""
	Returns cost-to-go matrix S and policy matrix K given local state x and effort u.

	"""
	# First and third rows of R.T
	w2b = np.array([
					[np.cos(x[2]), np.sin(x[2]), 0],
					[           0,            0, 1]
				  ])

	# Policy
	S = np.diag([1, 1, 1, 1, 1])
	K = np.hstack((kp.dot(w2b), kd))

	return (S, K)

def erf(xgoal, x):
	"""
	Returns error e given two states xgoal and x.

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
x0 = np.array([0, 0, np.deg2rad(0), 0, 0])
goal = [40, 40, np.deg2rad(90), 0, 0]
goal_buffer = [8, 8, np.inf, np.inf, np.inf]
error_tol = np.copy(goal_buffer)/2

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
		if npl.norm(p - goal[:2]) > 2*car_length and npl.norm(np.array(p - x0[:2])) > 2*car_length:
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

buff = 40
sample_space = [(goal[0]-buff, goal[0]+buff),
				(goal[0]-buff, goal[1]+buff),
				(-np.pi, np.pi),
				(0.9*velmax[0], velmax[0]),
				(-velmax[1], velmax[1])]

goal_bias = [0.5, 0.5, 0, 0, 0]

xrand_gen = None

################################################# PLAN

constraints = lqrrt.Constraints(nstates=nstates, ncontrols=ncontrols,
								goal_buffer=goal_buffer, is_feasible=is_feasible)

planner = lqrrt.Planner(dynamics, lqr, constraints,
						horizon=5, dt=0.1,
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
goal_history = np.zeros((len(t_arr), nstates))
u_history = np.zeros((len(t_arr), ncontrols))

# Interpolate plan
for i, t in enumerate(t_arr):

	# Planner's decision
	x = planner.get_state(t)
	u = planner.get_effort(t)

	# Record this instant
	x_history[i, :] = x
	goal_history[i, :] = goal
	u_history[i, :] = u

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
		 t_arr, u_history[:, 1], 'g')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 5)
ax1.set_ylabel('X Velocity (m/s)', fontsize=16)
ax1.plot(t_arr, x_history[:, 3], 'k',
		 t_arr, goal_history[:, 3], 'g--')
ax1.grid(True)
ax1.set_xlabel('Time (s)')

ax1 = fig1.add_subplot(2, 4, 7)
ax1.set_ylabel('Yaw Rate (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(x_history[:, 4]), 'k',
		 t_arr, np.rad2deg(goal_history[:, 4]), 'g--')
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

radius = car_width/2
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

td = car_length/3/2 + 0.5
graphic_robot_1 = ax2.add_patch(plt.Circle(((x_history[0, 0]+td*np.cos(x_history[0, 2]), x_history[0, 1]+td*np.sin(x_history[0, 2]))), radius=radius, fc='k'))
graphic_robot_2 = ax2.add_patch(plt.Circle((x_history[0, 0], x_history[0, 1]), radius=radius, fc='k'))
graphic_robot_3 = ax2.add_patch(plt.Circle(((x_history[0, 0]-td*np.cos(x_history[0, 2]), x_history[0, 1]-td*np.sin(x_history[0, 2]))), radius=radius, fc='k'))

llen = car_length/2
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
