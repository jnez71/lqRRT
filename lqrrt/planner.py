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
		 feedback gain matrix. That is, (S, K) = lqr(x, u).

	constraints: Instance of the Constraints class that defines
				 feasibility of states & efforts, goal region, etc...

	horizon: The simulation duration in seconds used to extend the tree.
			 If you set this to 0, a pseudo-reachability-guided heuristic is used.

	dt: The simulation timestep in seconds used to extend the tree.

	FPR: Failed Path Retention factor. When a path is grown and found to be infeasible,
		 this is the fraction of the path that is retained up to that infeasible point.

	error_tol: The state error array or scalar defining controller convergence.

	erf: Function that takes two states xgoal and x and returns the state error
		 between them. Defaults to simple subtraction xgoal - x. This is useful
		 if your state includes a quaternion or heading.

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

	printing: Bool that specifies if internal stuff should be printed.

	"""
	def __init__(self, dynamics, lqr, constraints,
				 horizon, dt=0.05, FPR=0.5,
				 error_tol=0.05, erf=np.subtract,
				 min_time=0.5, max_time=1, max_nodes=1E5,
				 goal0=None, system_time=time.time, printing=True):

		self.set_system(dynamics, lqr, constraints, erf)
		
		self.set_resolution(horizon, dt, FPR, error_tol)

		self.set_runtime(min_time, max_time, max_nodes)

		self.set_goal(goal0)

		self.set_system_time(system_time)

		self.printing = printing
		self.killed = False

#################################################

	def update_plan(self, x0, sample_space, goal_bias=0,
					xrand_gen=None, finish_on_goal=False,
					specific_time=None):
		"""
		A new tree is grown from the seed x0 in an attempt to plan
		a path to the goal. The returned path can be accessed with
		the interpolator functions get_state(t) and get_effort(t).

		The tree is motivated by uniform random samples in the over
		the given sample_space. The sample_space is a list of n tuples
		where n is the number of states; [(min1, max1), (min2, max2)...].

		The goal_bias is the fraction of the time the goal is sampled.
		It can be a scalar from 0 (none of the time) to 1 (all the time)
		or a list of scalars corresponding to each state dimension.

		Alternatively, you can give a function xrand_gen which takes the
		current planner instance (self) and outputs the random sample state.
		Doing this will override both sample_space and goal_bias, which you
		can set to None only if you provide an xrand_gen.

		After min_time seconds, the fastest available path from x0 to
		the current goal is returned and the functions get_state(t)
		and get_effort(t) are modified to interpolate this new path.

		If no path was found yet, the search continues until max_time or
		until the node limit is breached. After the limit, a warning is
		printed and the path that gets nearest to the goal is used instead.

		If finish_on_goal is set to True, once the plan makes it to the goal
		region (goal plus buffer), it will attempt to steer one more path
		directly into the exact goal. Can fail for nonholonomic systems.

		If you want this update_plan to plan for some specific amount of
		time (instead of using the global min_time and max_time), pass it
		in as specific_time in seconds.

		This function returns True if it finished fully, or False if it was
		haulted. It can hault if it is killed or if the tree exceeds max_nodes,
		or if no goal has been set yet.

		"""
		# Safety first!
		x0 = np.array(x0, dtype=np.float64)
		if self.goal is None:
			print("No goal has been set yet!")
			self.get_state = lambda t: x0
			self.get_effort = lambda t: np.zeros(self.ncontrols)
			return False

		# Store timing
		if specific_time is None:
			min_time = self.min_time
			max_time = self.max_time
		else:
			min_time = specific_time
			max_time = specific_time

		# Reset the tree
		self.tree = Tree(x0, self.lqr(x0, np.zeros(self.ncontrols)))

		# If not given an xrand_gen function, make the standard one
		if xrand_gen is None:

			# Properly cast the given goal bias
			if goal_bias is None:
				goal_bias = [0] * self.nstates
			elif hasattr(goal_bias, '__contains__'):
				if len(goal_bias) != self.nstates:
					raise ValueError("Expected goal_bias to be scalar or have same length as state.")
			else:
				goal_bias = [goal_bias] * self.nstates

			# Properly cast the given sample space and extract statistics
			sample_space = np.array(sample_space, dtype=np.float64)
			if sample_space.shape != (self.nstates, 2):
				raise ValueError("Expected sample_space to be list of nstates tuples.")
			sampling_centers = np.mean(sample_space, axis=1)
			sampling_spans = np.diff(sample_space).flatten()

			# Standard sampling
			def xrand_gen(planner):
				while time_elapsed < max_time:
					xrand = sampling_centers + sampling_spans*(np.random.sample(self.nstates)-0.5)
					for i, choice in enumerate(np.greater(goal_bias, np.random.sample())): #<<< should make this more pythonic
						if choice:
							xrand[i] = self.goal[i]
					if self.constraints.is_feasible(xrand, np.zeros(self.ncontrols)):
						return xrand
				return self.goal

		# Otherwise, use given sampling function
		else:
			if not hasattr(xrand_gen, '__call__'):
				raise ValueError("Expected xrand_gen to be None or a function.")

		# If we are in the goal already, just get to the center
		if self._in_goal(x0):
			# print("\n...planning to goal center...\n")
			self.plan_reached_goal = True
			self.x_seq, self.u_seq = self._steer(0, self.goal, force_arrive=True)
			self.x_seq.append(self.goal)
			self.u_seq.append(np.zeros(self.ncontrols))
			self.t_seq = np.arange(len(self.x_seq)) * self.dt
			self.T = self.t_seq[-1] + self.dt
			self.node_seq = self.tree.climb(self.tree.size-1)
			self._prepare_interpolators()
			return True

		# Loop managers
		if self.printing:
			print("\n...planning...")
		self.plan_reached_goal = False
		self.T = np.inf
		time_elapsed = 0
		time_start = self.systime()

		# Planning loop!
		while True:

			# Random sample state
			xrand = xrand_gen(self)

			# The "nearest" node to xrand has the least cost-to-go of all nodes
			nearestID = np.argmin(self._costs_to_go(xrand))

			# Candidate extension to the tree
			xnew_seq, unew_seq = self._steer(nearestID, xrand, force_arrive=False)

			# If steer produced any feasible results, extend tree
			if len(xnew_seq) > 0:

				# Add the new node to the tree
				xnew = np.copy(xnew_seq[-1])
				self.tree.add_node(nearestID, xnew, self.lqr(xnew, np.copy(unew_seq[-1])), xnew_seq, unew_seq)

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
						if self.printing:
							print("Found plan at elapsed time: {} s".format(np.round(time_elapsed, 6)))

			# Check if we should stop planning
			time_elapsed = self.systime() - time_start

			# Success
			if self.plan_reached_goal and time_elapsed >= min_time:
				if finish_on_goal:
					# Steer to exact goal
					xgoal_seq, ugoal_seq = self._steer(self.node_seq[-1], self.goal, force_arrive=True)
					# If it works, tack it onto the plan
					if len(xgoal_seq) > 0:
						self.tree.add_node(self.node_seq[-1], self.goal, None, xgoal_seq, ugoal_seq)
						self.node_seq.append(self.tree.size-1)
						self.x_seq.extend(xgoal_seq)
						self.u_seq.extend(ugoal_seq)
						self.t_seq = np.arange(len(self.x_seq)) * self.dt
				# Over and out!
				if self.printing:
					print("Tree size: {0}\nETA: {1} s".format(self.tree.size, np.round(self.T, 2)))
				self._prepare_interpolators()
				break

			# Failure (kinda)
			elif ((time_elapsed >= max_time or self.tree.size > self.max_nodes) and not self.plan_reached_goal) or self.killed:
				# Find node that has the most potential to steer to the goal
				Sgoal = self.lqr(self.goal, np.zeros(self.ncontrols))[0]
				for i, g in enumerate(self.constraints.goal_buffer):
					if np.isinf(g):
						Sgoal[:, i] = 0
				goaldiffs = self.tree.state - self.goal
				closestIDs = np.argsort(np.sum(np.tensordot(goaldiffs, Sgoal, axes=1) * goaldiffs, axis=1))
				best_dist = np.inf
				for i, ID in enumerate(closestIDs[:np.ceil(0.5*self.tree.size)]):
					xcheck_seq, ucheck_seq = self._steer(ID, self.goal, force_arrive=False)
					if len(xcheck_seq) > 1:
						diff = xcheck_seq[-1] - self.goal
						dist = diff.dot(Sgoal.dot(diff))
						if dist < best_dist:
							best_dist = dist
							best_ID = ID
							best_xcheck_seq = xcheck_seq
							best_ucheck_seq = ucheck_seq
				if not np.isinf(best_dist):
					self.tree.add_node(best_ID, best_xcheck_seq[-1], None, best_xcheck_seq, best_ucheck_seq)
					self.node_seq = self.tree.climb(self.tree.size-1)
				else:
					self.node_seq = self.tree.climb(closestIDs[0])
				# Construct plan
				self.x_seq, self.u_seq = self.tree.trajectory(self.node_seq)
				self.T = len(self.x_seq) * self.dt
				self.t_seq = np.arange(len(self.x_seq)) * self.dt
				# Over and out!
				if self.printing:
					print("Didn't reach goal.\nTree size: {0}\nETA: {1} s".format(self.tree.size, np.round(self.T, 2)))
				self._prepare_interpolators()
				break

		if self.killed or self.tree.size > self.max_nodes:
			if self.printing:
				print("Plan update terminated abruptly!")
			self.killed = False
			return False
		else:
			return True

#################################################

	def _costs_to_go(self, x):
		"""
		Returns an array of costs to go to x for each node in the
		current tree, in the same ordering as the nodes. This cost
		is  (v-x).T * S * (v-x)  for each node state v where S is
		found by LQR about x, not v.

		"""
		S = self.lqr(x, np.zeros(self.ncontrols))[0]
		diffs = self.tree.state - x
		return np.sum(np.tensordot(diffs, S, axes=1) * diffs, axis=1)

#################################################

	def _steer(self, ID, xtar, force_arrive=False):  #<<< need to numpy this function for final speedup!
		"""
		Starting from the given node ID's state, the system dynamics are
		forward simulated using the local LQR policy toward xtar.

		If the state updates into an infeasible condition, the simulation
		is finished and the path returned is half what was generated.

		If force_arrive is set to True, then the simulation isn't finished
		until xtar is achieved or until a physical timeout.

		If it is False, then the simulation will stop after self.horizon sim
		seconds or if the error drops below some reasonable self.error_tol.

		Returns the sequences of states and efforts. Note that the initial
		state is not included in the returned trajectory (to avoid tree overlap).

		"""
		# Set up
		K = np.copy(self.tree.lqr[ID][1])
		x = np.copy(self.tree.state[ID])
		x_seq = []; u_seq = []
		last_emag = np.inf
		
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
				x_seq = x_seq[:int(self.FPR * len(x_seq))]
				u_seq = u_seq[:int(self.FPR * len(u_seq))]
				break

			# Check force-arrive finish criteria
			if force_arrive:

				# Physical time limit
				elapsed_time = self.systime() - start_time
				if elapsed_time > self.min_time/2:
					print("(exact goal-convergence timed-out)")
					break
				
				# Definite convergence criteria
				if np.allclose(x, xtar, rtol=1E-4, atol=1E-4):
					break
			
			# or check lenient-arrive finish criteria
			else:
				i += 1
				emag = np.abs(e)
				
				# "Reachability" heuristic
				if self.rfactor:
					if np.all(emag >= last_emag):
						x_seq = []; u_seq = []
						self.horizon_iters = int(np.clip(self.horizon_iters/self.rfactor, 1, 1/self.dt))
						break
					if i == self.horizon_iters:
						self.horizon_iters = int(np.clip(self.rfactor*self.horizon_iters, 1, 1/self.dt))
					last_emag = emag
				
				# Horizon, or tolerable convergence criteria
				if i > self.horizon_iters or np.all(emag <= self.error_tol):
					break

			# Record
			x_seq.append(x)
			u_seq.append(u)

			# Get next control policy
			K = self.lqr(x, u)[1]

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

		if self.min_time > self.max_time:
			raise ValueError("The min_time must be less than or equal to the max_time.")

		if max_nodes is not None:
			self.max_nodes = max_nodes

#################################################

	def set_resolution(self, horizon=None, dt=None, FPR=None, error_tol=None):
		"""
		See class docstring for argument definitions.
		Arguments not given are not modified.

		"""
		if horizon is not None:
			self.horizon = horizon

		if dt is not None:
			self.dt = dt

		if FPR is not None:
			self.FPR = FPR

		if error_tol is not None:
			if np.shape(error_tol) in [(), (self.nstates,)]:
				self.error_tol = np.abs(error_tol).astype(np.float64)
			else:
				raise ValueError("Shape of error_tol must be scalar or length of state.")

		if self.horizon >= self.dt:
			self.horizon_iters = int(self.horizon / self.dt)
			self.rfactor = 0
		elif self.horizon == 0:
			self.horizon_iters = 1
			self.rfactor = int(2)
		else:
			raise ValueError("The horizon must be at least as big as dt.")

#################################################

	def set_system(self, dynamics=None, lqr=None, constraints=None, erf=None):
		"""
		See class docstring for argument definitions.
		Arguments not given are not modified.
		If dynamics gets modified, so must lqr (and vis versa).

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

#################################################

	def set_system_time(self, system_time):
		"""
		Sets the system time function, which gets called when the current physical time is needed.

		"""
		if hasattr(system_time, '__call__'):
			self.systime = system_time
		else:
			raise ValueError("Expected system_time to be a function.")

#################################################

	def kill_update(self):
		"""
		Raises a flag that will cause an abrupt termination of the update_plan routine.

		"""
		self.killed = True

#################################################

	def unkill(self):
		"""
		Lowers the kill_update flag. Do this if you made a mistake.

		"""
		self.killed = False

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
