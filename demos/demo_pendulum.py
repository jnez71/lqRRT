"""
Demo of using lqrrt to swing up and stabilize an underactuated double pendulum.

State:   [angle1, angle2, angvel1, angvel2]  (rad, rad, rad/s, rad/s)
Effort:  u1 (N*m)


TO BE FINISHED SOON!


"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl

from scipy.linalg import solve_discrete_are
import lqrrt

################################################# PHYSICAL PARAMETERS

# Simulation timestep
dt = 0.001  # s

# Initial condition
q = np.array([-np.pi/2, 0, 0, 0])  # (rad, rad, rad/s, rad/s)

# Link lengths
L = [1, 0.5]  # m

# Link masses
m = [5, 5]  # kg

# Local gravity
g = 9.81  # m/s^2

# Joint damping
d = [0.4, 0.4]  # (N*m)/(rad/s)

# Joint friction
b = [0.01, 0.01]  # N*m
c = [0.1, 0.1]  # s/rad

# Actuator limit
umax = np.inf#350  # N*m

################################################# EQUATIONS OF MOTION

nstates = 4
ncontrols = 1

def dynamics(q, u, dt):
    """
    Returns next state (qnext).
    Takes current state (q), control input (u), and timestep (dt).

    """
    # Mass matrix M(q)
    M = np.zeros((2, 2))
    M[0, 0] = (m[0]+m[1])*L[0]**2 + m[1]*L[1]**2 + 2*m[1]*L[0]*L[1]*np.cos(q[1])
    M[0, 1] = m[1]*L[1]**2 + m[1]*L[0]*L[1]*np.cos(q[1])
    M[1, 0] = M[0, 1]  # symmetry
    M[1, 1] = m[1]*L[1]**2

    # Centripetal and coriolis vector V(q)
    V = np.array([
                  -m[1]*L[0]*L[1]*(2*q[2]*q[3]+q[3]**2)*np.sin(q[1]),
                   m[1]*L[0]*L[1]*q[2]**2*np.sin(q[1])
                ])

    # Gravity vector G(q)
    G = np.array([
                  g*(m[0]+m[1])*L[0]*np.cos(q[0]) + m[1]*g*L[1]*np.cos(q[0]+q[1]),
                  m[1]*g*L[1]*np.cos(q[0]+q[1])
                ])

    # Joint damping D(q)
    D = np.array([
                  d[0]*q[2],
                  d[1]*q[3]
                ])

    # Joint friction
    F = np.array([
                  b[0]*np.tanh(c[0]*q[2]),
                  b[1]*np.tanh(c[1]*q[3])
                ])

    # Actuator saturation
    u = np.clip(u, -umax, umax)

    # Underactuatedness
    u = np.concatenate((u, [0]))

    # [theta1dot, theta2dot] = [w1, w2]   and   [w1dot, w2dot] = (M^-1)*(u-V-G-D-F)
    qnext = q + (np.concatenate((q[2:], npl.inv(M).dot(u - V - G - D - F))) * dt)

    return qnext

####

def kinem_forward(q):
    """
    Returns the state of the end effector ([px, py, vx, vy]).
    Takes the current joint state (q).

    """
    return np.array([
                     L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1]),
                     L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1]),
                     -L[0]*np.sin(q[0])*q[2] - L[1]*np.sin(q[0]+q[1])*(q[2]+q[3]),
                     L[0]*np.cos(q[0])*q[2] + L[1]*np.cos(q[0]+q[1])*(q[2]+q[3])
                   ])

################################################# CONTROL POLICY

def lqr(x, u):
    """
    Returns cost-to-go matrix S and policy matrix K given local state x and effort u.

    """
    S = np.diag([1, 1, 1, 1])
    K = np.array([[10, 200, 0, 0]])
    return (S, K)

####

def erf(qgoal, q):
    """
    Returns error e given two states qgoal and xq.

    """
    e = qgoal - q
    for i in [0, 1]:
        c = np.cos(q[i])
        s = np.sin(q[i])
        cg = np.cos(qgoal[i])
        sg = np.sin(qgoal[i])
        e[i] = np.arctan2(sg*c - cg*s, cg*c + sg*s)
    return e

################################################# OBJECTIVES AND CONSTRAINTS

# What is goal
goal = [np.pi/2, 0, 0, 0]
goal_buffer = [np.deg2rad(1), np.deg2rad(1), 0.001, 0.001]
error_tol = [np.deg2rad(10), np.deg2rad(10), 0.1, 0.1]

# Impose a conservative effort limit
umax_plan = 0.75*umax

def is_feasible(x, u):
    if abs(u) > umax_plan:
        return False
    return True

################################################# HEURISTICS

sample_space = [(0, 1.1*np.pi), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi), (-np.pi, np.pi)]

goal_bias = [0.5, 0.5, 0.5, 0.5]

xrand_gen = None

################################################# PLAN

constraints = lqrrt.Constraints(nstates=nstates, ncontrols=ncontrols,
                                goal_buffer=goal_buffer, is_feasible=is_feasible)

planner = lqrrt.Planner(dynamics, lqr, constraints,
                        horizon=0, dt=dt, FPR=0.5,
                        error_tol=error_tol, erf=erf,
                        min_time=60, max_time=61, max_nodes=1E5,
                        goal0=goal)

planner.update_plan(q, sample_space, goal_bias=goal_bias,
                    xrand_gen=xrand_gen, finish_on_goal=True)

################################################# SIMULATION

# Time domain
T = planner.T + 1  # s
t_arr = np.arange(0, T, dt)
framerate = 60  # fps

# Histories
q_history = np.zeros((len(t_arr), nstates))
u_history = np.zeros((len(t_arr), ncontrols))
goal_history = np.zeros((len(t_arr), nstates))
x_history = np.zeros((len(t_arr), nstates))

# Just interpolate for now
for i, t in enumerate(t_arr):
    q_history[i, :] = planner.get_state(t)
    u_history[i, :] = planner.get_effort(t)
    goal_history[i, :] = goal
    x_history[i, :] = kinem_forward(q_history[i, :])

# # Simulate
# for i, t in enumerate(t_arr):

#   # Planner's decision
#   qref = planner.get_state(t)
#   uref = planner.get_effort(t)

#   # Controllers decision
#   u = lqr(q, uref)[1].dot(erf(qref, np.copy(q))) + uref

#   # Record this instant
#   q_history[i, :] = q
#   qref_history[i, :] = qref
#   goal_history[i, :] = goal
#   u_history[i, :] = u

#   # Step dynamics
#   q = dynamics(q, u, dt)

################################################# VISUALIZATION

print("\n...plotting...")
from matplotlib import pyplot as plt
import matplotlib.animation as ani

def plot_tree(dx, dy, ax):
    ax.set_xlabel('State {}'.format(dx))
    ax.set_ylabel('State {}'.format(dy))
    ax.grid(True)
    for ID in xrange(planner.tree.size):
        x_seq = np.array(planner.tree.x_seq[ID])
        if ID in planner.node_seq:
            ax.plot((np.rad2deg(x_seq[:, dx])), (np.rad2deg(x_seq[:, dy])), color='r', zorder=2)
        else:
            ax.plot((np.rad2deg(x_seq[:, dx])), (np.rad2deg(x_seq[:, dy])), color='0.75', zorder=1)
    ax.scatter(np.rad2deg(planner.tree.state[0, dx]), np.rad2deg(planner.tree.state[0, dy]), color='b', s=48)
    ax.scatter(np.rad2deg(planner.tree.state[planner.node_seq[-1], dx]), np.rad2deg(planner.tree.state[planner.node_seq[-1], dy]), color='r', s=48)
    ax.scatter(np.rad2deg(goal[dx]), np.rad2deg(goal[dy]), color='g', s=48)

# Figure for joint space results
fig1 = plt.figure()
fig1.suptitle('Results', fontsize=20)

# Plot joint angle 1
ax1 = fig1.add_subplot(2, 3, 1)
ax1.set_ylabel('Angle 1 (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 0]), 'k',
         t_arr, np.rad2deg(goal_history[:, 0]), 'g--')
ax1.grid(True)

# Plot joint angle 2
ax1 = fig1.add_subplot(2, 3, 2)
ax1.set_ylabel('Angle 2 (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 1]), 'k',
         t_arr, np.rad2deg(goal_history[:, 1]), 'g--')
ax1.grid(True)

# Plot control efforts
ax1 = fig1.add_subplot(2, 3, 3)
ax1.set_ylabel('Torque (N*m)', fontsize=16)
ax1.plot(t_arr, u_history[:, 0], 'k')
ax1.grid(True)

# Plot joint velocity 1
ax1 = fig1.add_subplot(2, 3, 4)
ax1.set_ylabel('Velocity 1 (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 2]), 'k',
         t_arr, np.rad2deg(goal_history[:, 2]), 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot joint velocity 1
ax1 = fig1.add_subplot(2, 3, 5)
ax1.set_ylabel('Velocity 2 (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 3]), 'k',
         t_arr, np.rad2deg(goal_history[:, 3]), 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot angle space path
ax1 = fig1.add_subplot(2, 3, 6)
ax1.set_xlabel('Angle 1 (deg)', fontsize=16)
ax1.set_ylabel('Angle 2 (deg)', fontsize=16)
ax1.plot(np.rad2deg(q_history[:, 0]), np.rad2deg(q_history[:, 1]), 'k')
ax1.scatter(np.rad2deg(planner.tree.state[0, 0]), np.rad2deg(planner.tree.state[0, 1]), color='b', s=48)
ax1.scatter(np.rad2deg(planner.tree.state[planner.node_seq[-1], 0]), np.rad2deg(planner.tree.state[planner.node_seq[-1], 1]), color='r', s=48)
ax1.scatter(np.rad2deg(goal[0]), np.rad2deg(goal[1]), color='g', s=48)
ax1.grid(True)

# plt.show()

# print("\n...plotting more...")
# # Figure for detailed tree
# fig2 = plt.figure()
# fig2.suptitle('Tree', fontsize=20)

# # Plot joint 1 tree
# ax2 = fig2.add_subplot(1, 2, 1)
# plot_tree(0, 1, ax2)

# # Plot joint 2 tree
# ax2 = fig2.add_subplot(1, 2, 2)
# plot_tree(1, 3, ax2)

# plt.show()

# Figure for animation
fig3 = plt.figure()
fig3.suptitle('Evolution')
ax3 = fig3.add_subplot(1, 1, 1)
ax3.set_xlabel('- World X +')
ax3.set_ylabel('- World Y +')
ax3.set_xlim([-np.sum(L)-1, np.sum(L)+1])
ax3.set_ylim([-np.sum(L)-1, np.sum(L)+1])
ax3.grid(True)

# Position of intermediate joint during motion
elb_history = np.concatenate(([L[0]*np.cos(q_history[:, 0])], [L[0]*np.sin(q_history[:, 0])])).T

# Lines for representing the links and points for joints
lthick = 3
pthick = 12
link1 = ax3.plot([0, elb_history[0, 0]], [0, elb_history[0, 1]], color='k', linewidth=lthick)
link2 = ax3.plot([elb_history[0, 0], x_history[0, 0]], [elb_history[0, 1], x_history[0, 1]], color='k', linewidth=lthick)
end = ax3.scatter(x_history[0, 0], x_history[0, 1], color='k', s=pthick*m[1], zorder=2)
elb = ax3.scatter(elb_history[0, 0], elb_history[0, 1], color='k', s=pthick*m[0], zorder=2)
bse = ax3.scatter(0, 0, color='k', s=5*pthick*m[0])

# Function for updating the animation frame
def update(arg, ii=[0]):

    i = ii[0]  # don't ask...

    if np.isclose(t_arr[i], np.around(t_arr[i], 1)):
        fig3.suptitle('Evolution (Time: {})'.format(t_arr[i]), fontsize=24)

    link1[0].set_data([0, elb_history[i, 0]], [0, elb_history[i, 1]])
    link2[0].set_data([elb_history[i, 0], x_history[i, 0]], [elb_history[i, 1], x_history[i, 1]])
    end.set_offsets((x_history[i, 0], x_history[i, 1]))
    elb.set_offsets((elb_history[i, 0], elb_history[i, 1]))

    ii[0] += int(1 / (dt * framerate))
    if ii[0] >= len(t_arr):
        print("Resetting animation!")
        ii[0] = 0

    return [link1, link2, end, elb]

# Run animation
animation = ani.FuncAnimation(fig3, func=update, interval=dt*1000)
print("\nRemember to keep the diplay window aspect ratio square.\n")
plt.show()
