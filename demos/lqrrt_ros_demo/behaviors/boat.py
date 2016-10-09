"""
Constructs a planner that is good for boatin' around!

"""
from __future__ import division
import numpy as np
import numpy.linalg as npl

from params import *
import lqrrt

################################################# DYNAMICS

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

	# Actuator saturation
	u = B.dot(np.clip(invB.dot(u), -thrust_max, thrust_max))

	# M*vdot + D*v = u  and  pdot = R*v
	xdot = np.concatenate((R.dot(x[3:]), invM*(u - D*x[3:])))

	# First-order integrate
	return x + xdot*dt

################################################# POLICY

kp = np.diag([150, 150, 350])
kd = np.diag([150, 150, 50])
S = np.diag([1, 1, 1, 1, 1, 1])

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
FPR = 0.5

goal_buffer = [0.05, 0.05, np.inf, np.inf, np.inf, np.inf]
error_tol = np.copy(goal_buffer)

def gen_ss(seed, goal):
	"""
	Returns a sample space given a seed state and goal state.

	"""
	return [(min([seed[0], goal[0]]) - ss_buff, max([seed[0], goal[0]]) + ss_buff),
			(min([seed[1], goal[1]]) - ss_buff, max([seed[1], goal[1]]) + ss_buff),
			(0, 0),
			(-abs(velmax_neg_plan[0]), velmax_pos_plan[0]),
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
