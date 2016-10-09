"""
Constructs a planner that is good for being like a car going forwards!

"""
from __future__ import division
import numpy as np
import numpy.linalg as npl

from params import *
import lqrrt

################################################# DYNAMICS

magic_rudder = 6000

def dynamics(x, u, dt):
	"""
	Returns next state given last state x, wrench u, and timestep dt.

	"""
	# Rotation matrix (orientation, converts body to world)
	R = np.array([
				  [np.cos(x[2]), -np.sin(x[2]), 0],
				  [np.sin(x[2]),  np.cos(x[2]), 0],
				  [           0,             0, 1]
				])

	# Construct drag coefficients based on our motion signs
	D = np.copy(D_neg)
	for i, v in enumerate(x[3:]):
		if v >= 0:
			D[i] = D_pos[i]

	# Heading controller trying to keep us car-like
	vw = R[:2, :2].dot(x[3:5])
	ang = np.arctan2(vw[1], vw[0])
	c = np.cos(x[2])
	s = np.sin(x[2])
	cg = np.cos(ang)
	sg = np.sin(ang)
	u[2] = u[2] + magic_rudder*np.arctan2(sg*c - cg*s, cg*c + sg*s)

	# Actuator saturation
	u = B.dot(np.clip(invB.dot(u), -thrust_max, thrust_max))

	# M*vdot + D*v = u  and  pdot = R*v
	xdot = np.concatenate((R.dot(x[3:]), invM*(u - D*x[3:])))

	# First-order integrate
	xnext = x + xdot*dt

	# # Impose not turning in place
	# if x[3] > 0:
	# 	xnext[5] = np.clip(np.abs(xnext[3]/velmax_pos[0]), 0, 1) * xnext[5]
	# elif x[3] < 0:
	# 	xnext[5] = np.clip(np.abs(xnext[3]/velmax_neg[0]), 0, 1) * xnext[5]
	
	# Impose not driving backwards
	if xnext[3] < 0:
		xnext[3] = abs(x[3])

	return xnext


################################################# POLICY

kp = np.diag([150, 5, 0])
kd = np.diag([150, 5, 0])
S = np.diag([1, 1, 1, 0, 0, 0])

def lqr(x, u):
	"""
	Returns cost-to-go matrix S and policy matrix K given local state x and effort u.

	"""
	R = np.array([
				  [np.cos(x[2]), -np.sin(x[2]), 0],
				  [np.sin(x[2]),  np.cos(x[2]), 0],
				  [           0,             0, 1]
				])
	K = np.hstack((kp.dot(R.T), kd))
	return (S, K)

################################################# HEURISTICS

goal_bias = [0.5, 0.5, 0, 0, 0, 0]
FPR = 0.65

goal_buffer = [1.5*boat_length, 1.5*boat_length, np.inf, np.inf, np.inf, np.inf]
error_tol = np.copy(goal_buffer)/10

def gen_ss(seed, goal):
	"""
	Returns a sample space given a seed state and goal state.

	"""
	return [(min([seed[0], goal[0]]) - ss_buff, max([seed[0], goal[0]]) + ss_buff),
			(min([seed[1], goal[1]]) - ss_buff, max([seed[1], goal[1]]) + ss_buff),
			(0, 0),
			(0.9*velmax_pos_plan[0], velmax_pos_plan[0]),
			(-abs(velmax_neg_plan[1]), velmax_pos_plan[1]),
			(-abs(velmax_neg_plan[2]), velmax_pos_plan[2])]

################################################# MAIN ATTRIBUTES

min_time = 3  # s

constraints = lqrrt.Constraints(nstates=nstates, ncontrols=ncontrols,
								goal_buffer=goal_buffer, is_feasible=unset)

planner = lqrrt.Planner(dynamics, lqr, constraints,
						horizon=horizon, dt=dt, FPR=FPR,
						error_tol=error_tol, erf=erf,
						min_time=min_time, max_time=min_time, max_nodes=max_nodes,
						system_time=unset)
