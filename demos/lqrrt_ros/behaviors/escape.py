"""
Constructs a planner that is good for getting out of sticky situations!

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

    # Actuator saturation with even downscaling
    thrusts = invB.dot(u)
    ratios = thrust_max / np.clip(np.abs(thrusts), 1E-6, np.inf)
    if np.any(ratios < 1):
        u = B.dot(np.min(ratios) * thrusts)

    # M*vdot + D*v = u  and  pdot = R*v
    xdot = np.concatenate((R.dot(x[3:]), invM*(u - D*x[3:])))

    # First-order integrate
    xnext = x + xdot*dt

    return xnext

################################################# POLICY

kp = np.diag([150, 150, 2000])
kd = np.diag([120, 120, 0.01])
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

goal_buffer = [free_radius, free_radius, np.inf, np.inf, np.inf, np.inf]
error_tol = np.copy(goal_buffer)

def gen_ss(seed, goal, buff=40):
    """
    Returns a sample space given a seed state, goal state, and buffer.

    """
    return [(seed[0] - buff, seed[0] + buff),
            (seed[1] - buff, seed[1] + buff),
            (-np.pi, np.pi),
            (-abs(velmax_neg[0]), velmax_pos[0]),
            (-abs(velmax_neg[1]), velmax_pos[1]),
            (-abs(velmax_neg[2]), velmax_pos[2])]

################################################# MAIN ATTRIBUTES

constraints = lqrrt.Constraints(nstates=nstates, ncontrols=ncontrols,
                                goal_buffer=goal_buffer, is_feasible=unset)

planner = lqrrt.Planner(dynamics, lqr, constraints,
                        horizon=horizon, dt=dt,
                        FPR=FPR, CPF=CPF,
                        error_tol=error_tol, erf=unset,
                        min_time=0, max_time=5, max_nodes=max_nodes,
                        sys_time=unset, printing=False)
