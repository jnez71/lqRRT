"""
The main class for lqRRT.

Create an instance of Planner and then call update_plan to generate a plan internal to the instance.
To get the state or effort at some time t, use the functions get_state(t) and get_effort(t).

"""

################################################# DEPENDENCIES

from __future__ import division
import time

import numpy as np
import numpy.linalg as npl
from scipy.interpolate import interp1d

from tree import Tree
from constraints import Constraints

################################################# PRIMARY CLASS

class Planner:
	"""
	To initialize, provide...

	dynamics: Function that returns the next state given
			  the current state x and the current effort u,
			  and timestep dt. That is, xnext = dynamics(x, u, dt).

	lqr: Function that returns the local LQR cost-to-go matrix S
		 and policy matrix K as a tuple of arrays (S, K) where S
		 solves the local Riccati equation and K is the associated
		 feedback gain matrix. That is, (S, K) = lqr(x).

	constraints: Instance of the Constraints class that defines
				 feasibility of states & efforts, search space, etc...

	horizon: The simulation duration in seconds used to extend the tree.

	dt: The simulation timestep in seconds used to extend the tree.

	error_tol: The state error array defining convergence for the simulation.
			   If a scalar is given, all states are tried against that scalar.

	erf: Function that takes two states xgoal and x and returns the state error
		 between them. Defaults to simple subtraction xgoal - x.

	min_time: The least number of seconds that the tree will
			  grow for. That is, even if a feasible plan is found
			  before min_time, it will keep growing until min_time
			  is reached and then give the best of the plans.

	max_time: The max number of seconds that the tree will grow for.
			  That is, if there are still no feasible plans after this
			  amount of time, the plan_reached_goal flag will remain False
			  and the plan that gets closest to the goal is given.

	max_nodes: If the tree reaches this number of nodes but no path is
			   found, the plan_reached_goal flag will remain False and the
			   plan that gets closest to the goal is given.

	goal0: The initial goal state. If left as None, update_plan
		   cannot be run. Use set_goal to set the goal at any time.
		   Be sure to update the plan after setting a new goal.

	system_time: Function that returns the real-world system time.
				 Defaults to the Python time library's time().

	"""
	def __init__(self, dynamics, lqr, constraints,
				 horizon=10, dt=0.05, error_tol=0.05, erf=np.subtract,
				 min_time=0.5, max_time=1, max_nodes=1E5,
				 goal0=None, system_time=time.time):

		self.set_system(dynamics, lqr, constraints, erf)
		
		self.set_resolution(horizon, dt, error_tol)

		self.set_runtime(min_time, max_time, max_nodes)

		self.set_goal(goal0)

		if hasattr(system_time, '__call__'):
			self.systime = system_time
		else:
			raise ValueError("Expected system_time to be a function.")

#################################################

	def update_plan(self, x0, sampling_bias=0.2, xrand_gen=None,
					finish_on_goal=True, reset_tree=True):
		"""
		A new tree is grown from the seed x0 in an attempt to plan
		a path to the goal. The returned path can be accessed with
		the interpolator functions get_state(t) and get_effort(t).

		After min_time seconds, the best available path from x0 to
		the current goal is returned and the functions get_state(t)
		and get_effort(t) are modified to interpolate this new path.

		If no path was found yet, the search continues until max_time or
		until the node limit is breached. After the limit, a warning is
		printed and the path that gets nearest to the goal is used instead.

		The sampling_bias is the fraction of the time the goal is sampled.
		It can be a scalar from 0 (none of the time) to 1 (all of the time)
		or a list of scalars corresponding to each state dimension.

		Alternatively, you can give a function xrand_gen which takes the current
		successful path (self.x_seq, None if no path found yet) and outputs
		the random sample state. (Giving this will disregard sampling_bias).

		If finish_on_goal is set to True, once the plan makes it to the goal
		region (goal plus buffer), it will attempt to steer one more path
		directly into the exact goal. Can fail for nonholonomic systems.

		If reset_tree is made False, the tree is not reset before growing
		continues. THE SEED WILL NOT CHANGE TO X0. Just don't do this.

		"""
		# Safety first!
		x0 = np.array(x0, dtype=np.float64)
		if self.goal is None:
			print("No goal has been set yet!")
			self.get_state = lambda t: x0
			self.get_effort = lambda t: np.zeros(self.ncontrols)
			return None

		# Reset the tree if told to (or if no tree exists yet)
		if reset_tree or self.tree is None:
			self.tree = Tree(x0, self.lqr(x0))
			self.x_seq = None
			self.u_seq = None
			self.t_seq = None

		# If not given an xrand_gen function, make the standard one
		if xrand_gen is None:

			# Properly cast the given sampling bias
			if sampling_bias is None:
				sampling_bias = [0] * self.nstates
			elif hasattr(sampling_bias, '__contains__'):
				if len(sampling_bias) != self.nstates:
					raise ValueError("Expected sampling_bias to be scalar or have same length as state.")
			else:
				sampling_bias = [sampling_bias] * self.nstates

			# If we are already in the goal region, just sample the goal
			if self._in_goal(x0):
				sampling_bias = [1] * self.nstates

			# Error-sized hyperbox plus buffer
			sampling_spans = 2*np.abs(self.goal - x0) + self.constraints.search_buffer_spans

			# The standard xrand_gen is centered at goal, offset by search_buffer mean,
			# and expanded to a hyperbox of size sampling_spans
			def xrand_gen(x_seq):
				xrand = self.goal + sampling_spans*(np.random.sample(self.nstates)-0.5) + self.constraints.search_buffer_offsets
				for i, choice in enumerate(np.greater(sampling_bias, np.random.sample())): #<<< should make this more pythonic
					if choice:
						xrand[i] = self.goal[i]
				return xrand

		# Loop managers
		print("\n...planning...\n")
		self.plan_reached_goal = False
		self.T = np.inf
		time_elapsed = 0
		time_start = self.systime()

		# Planning loop!
		while True:

			# Random sample state
			xrand = xrand_gen(np.array(self.x_seq))

			# The "nearest" node to xrand has the least cost-to-go of all nodes
			nearestID = np.argmin(self._costs_to_go(xrand))

			# Candidate extension to the tree
			xnew_seq, unew_seq = self._steer(nearestID, xrand, force_arrive=False)

			# If steer produced any feasible results, extend tree
			if len(xnew_seq) > 1:

				# Add the new node to the tree
				xnew = np.copy(xnew_seq[-1])
				self.tree.add_node(nearestID, xnew, self.lqr(xnew), xnew_seq, unew_seq)

				# Check if the newest node reached the goal region
				if self._in_goal(xnew):

					# Raise flag
					self.plan_reached_goal = True

					# Climb tree to construct sequence of states for this path
					node_seq = self.tree.climb(self.tree.size-1)
					x_seq, u_seq = self.tree.trajectory(node_seq)

					# Expected time to complete this plan
					T = len(x_seq) * self.dt

					# Retain this plan if it is faster than the previous one
					if T < self.T:
						self.T = T
						self.node_seq = node_seq
						self.x_seq = x_seq
						self.u_seq = u_seq
						self.t_seq = np.arange(len(self.x_seq)) * self.dt
						print("Found plan at elapsed time: {} s".format(np.round(time_elapsed, 6)))

			# Check if we should stop planning
			time_elapsed = self.systime() - time_start

			# Success
			if self.plan_reached_goal and time_elapsed >= self.min_time:
				if finish_on_goal:
					# Add exact goal to the tree
					xgoal_seq, ugoal_seq = self._steer(self.node_seq[-1], self.goal, force_arrive=True)
					self.tree.add_node(self.node_seq[-1], self.goal, None, xgoal_seq, ugoal_seq)
					# Tack it on to the plan too
					self.node_seq.append(self.tree.size-1)
					self.x_seq.extend(xgoal_seq)
					self.u_seq.extend(ugoal_seq)
					self.T = len(self.x_seq) * self.dt
					self.t_seq = np.arange(len(self.x_seq)) * self.dt
				# Over and out!
				print("\nSuccess!\nTree size: {0}\nETA: {1} s".format(self.tree.size, np.round(self.T, 2)))
				self._prepare_interpolators()
				break

			# Failure (kinda)
			elif (time_elapsed >= self.max_time or self.tree.size > self.max_nodes) and not self.plan_reached_goal:
				# Climb tree to construct sequence of nodes from seed to closest-node-to-goal
				closestID = np.argmin(self._costs_to_go(self.goal))
				self.node_seq = self.tree.climb(closestID)
				# Construct plan
				self.x_seq, self.u_seq = self.tree.trajectory(self.node_seq)
				self.T = len(self.x_seq) * self.dt
				self.t_seq = np.arange(len(self.x_seq)) * self.dt
				# Over and out!
				print("Didn't reach goal.\nTree size: {0}\nETA: {1} s".format(self.tree.size, np.round(self.T, 2)))
				self._prepare_interpolators()
				break

#################################################

	def _costs_to_go(self, x):
		"""
		Returns an array of costs to go to x for each node in the
		current tree, in the same ordering as the nodes. This cost
		is  (v-x).T * S * (v-x)  for each node state v where S is
		found by LQR about x, not v.

		"""
		S = self.lqr(x)[0]
		diffs = self.tree.state - x
		return np.sum(np.tensordot(diffs, S, axes=1) * diffs, axis=1)

#################################################

	def _steer(self, ID, xtar, force_arrive=False):  #<<< need to numpy this function for final speedup!
		"""
		Starting from the given node ID's state, the system dynamics are
		forward simulated using the local LQR policy toward xtar.

		If the state updates into an infeasible condition, the simulation
		is finished and the path returned is half what was generated.

		If the error magnitude between the sim state and xtar falls below
		self.error_tol at any time, the simulation is also finished.

		If force_arrive is set to True, then the simulation isn't finished
		until error_tol is achieved or until a physical timeout. If it is
		False, then the simulation will stop after self.horizon sim seconds.

		Returns the sequences of states and efforts. Note that the initial
		state is not included in the returned trajectory (to avoid tree overlap).

		"""
		# Set up
		K = np.copy(self.tree.lqr[ID][1])
		x = np.copy(self.tree.state[ID])
		x_seq = []; u_seq = []
		
		# Management
		i = 0; elapsed_time = 0
		start_time = self.systime()
		
		# Simulate
		while True:

			# Compute effort using local LQR policy
			e = self.erf(np.copy(xtar), np.copy(x))
			u = K.dot(e)
			
			# Step forward dynamics
			x = self.dynamics(np.copy(x), np.copy(u), self.dt)

			# Check for feasibility
			if not self.constraints.is_feasible(x, u):
				x_seq = x_seq[:len(x_seq)//2]
				u_seq = u_seq[:len(u_seq)//2]
				break

			# Get next control policy
			K = self.lqr(x)[1]

			# Record
			x_seq.append(x)
			u_seq.append(u)

			# Error based finish criteria #<<< get rid of slowdowns
			if np.all(np.less_equal(np.abs(e), self.error_tol)):
				break

			# Time based finish criteria
			if force_arrive:
				elapsed_time = self.systime() - start_time
				if elapsed_time > self.min_time:
					break
			else:
				i += 1
				if i > self.horizon_iters:
					break

		return (x_seq, u_seq)

#################################################

	def _in_goal(self, x):
		"""
		Returns True if some state x is in the goal region.

		"""
		return all(goal_span[0] < v < goal_span[1] for goal_span, v in zip(self.goal_region, x))

#################################################

	def _prepare_interpolators(self):
		"""
		Updates the interpolator functions the user calls
		to interpolate the current plan.

		"""
		if len(self.x_seq) == 1:
			self.get_state = lambda t: self.x_seq[0]
			self.get_effort = lambda t: np.zeros(self.ncontrols)
		else:
			self.get_state = interp1d(self.t_seq, np.array(self.x_seq), axis=0, assume_sorted=True,
									  bounds_error=False, fill_value=self.x_seq[-1][:])
			self.get_effort = interp1d(self.t_seq, np.array(self.u_seq), axis=0, assume_sorted=True,
									  bounds_error=False, fill_value=self.u_seq[-1][:])

#################################################

	def set_goal(self, goal):
		"""
		Modifies the goal state and region.
		Be sure to update the plan after modifying the goal.

		"""
		if goal is None:
			self.goal = None
		else:
			if len(goal) == self.nstates:
				self.goal = np.array(goal, dtype=np.float64)
			else:
				raise ValueError("The goal state must have same dimensionality as state space.")
			
			goal_region = []
			for i, buff in enumerate(self.constraints.goal_buffer):
				goal_region.append((self.goal[i]-buff, self.goal[i]+buff))
			
			self.goal_region = goal_region
			self.plan_reached_goal = False

#################################################

	def set_runtime(self, min_time=None, max_time=None, max_nodes=None):
		"""
		See class docstring for argument definitions.
		Arguments not given are not modified.

		"""
		if min_time is not None:
			self.min_time = min_time

		if max_time is not None:
			self.max_time = max_time

		if self.min_time >= self.max_time:
			raise ValueError("The min_time must be strictly less than the max_time.")

		if max_nodes is not None:
			self.max_nodes = max_nodes

#################################################

	def set_resolution(self, horizon=None, dt=None, error_tol=None):
		"""
		See class docstring for argument definitions.
		Arguments not given are not modified.

		"""
		if horizon is not None:
			self.horizon = horizon

		if dt is not None:
			self.dt = dt

		if error_tol is not None:
			if np.array(error_tol).shape in [(), (self.nstates,)]:
				self.error_tol = error_tol
			else:
				raise ValueError("Shape of error_tol must be scalar or length of state.")

		self.horizon_iters = self.horizon / self.dt

#################################################

	def set_system(self, dynamics=None, lqr=None, constraints=None, erf=None):
		"""
		See class docstring for argument definitions.
		Arguments not given are not modified.
		If dynamics gets modified, so must lqr (and vis versa).
		Calling this function resets the tree and plan.

		"""
		if dynamics is not None or lqr is not None:
			if hasattr(dynamics, '__call__'):
				self.dynamics = dynamics
			else:
				raise ValueError("Expected dynamics to be a function.")
			if hasattr(lqr, '__call__'):
				self.lqr = lqr
			else:
				raise ValueError("Expected lqr to be a function.")

		if constraints is not None:
			if isinstance(constraints, Constraints):
				self.constraints = constraints
				self.nstates = self.constraints.nstates
				self.ncontrols = self.constraints.ncontrols
			else:
				raise ValueError("Expected constraints to be an instance of the Constraints class.")

		if erf is not None:
			if hasattr(erf, '__call__'):
				self.erf = erf
			else:
				raise ValueError("Expected erf to be a function.")

		self.plan_reached_goal = False
		self.tree = None
		self.x_seq = None
		self.u_seq = None
		self.t_seq = None

#################################################

	def visualize(self, dx, dy):
		"""
		Plots the (dx,dy)-cross-section of the current tree,
		and highlights the current plan's trajectory.
		For example, dx=0, dy=1 plots the states #0 and #1.

		"""
		if hasattr(self, 'node_seq'):
			self.tree.visualize(dx, dy, node_seq=self.node_seq)
		else:
			print("There is no plan to visualize!")
