#!/usr/bin/env python
"""
Node that subscribes to the current boat state (Odometry message),
and the world-frame ogrid (OccupancyGrid message). It publishes the
REFerence trajectory that moves to the goal as an Odometry message, as
well as the path and tree as PoseArray messages. An action is provided
for moving the reference to some goal (Move.action).

"""
from __future__ import division
import numpy as np
import numpy.linalg as npl
from scipy.interpolate import interp1d

import rospy
import actionlib
import tf.transformations as trns

from std_msgs.msg import Bool, Int64, Float64, String
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid

from behaviors import params, car, boat, particle, escape
from lqrrt_ros_demo.msg import MoveAction, MoveFeedback, MoveResult


#<<< NEED TO FINISH OGRID DENSITY AND CIRCLE MOVES


################################################# INITIALIZATIONS

class LQRRT_Node(object):

	def __init__(self, odom_topic='/odom', ogrid_topic='/ogrid', ref_topic='/ref',
				 move_topic='/move_to', path_topic='/path', tree_topic='/tree'):
		"""
		Initialize with topic names.

		"""
		# Oncers
		self.revisit_period = 0.05  # s
		self.fudge_factor = 0.98
		self.rostime = lambda: rospy.Time.now().to_sec()
		self.state = None

		# Set-up planners
		self.behaviors_list = [car, boat, particle, escape]
		for behavior in self.behaviors_list:
			behavior.planner.set_system(erf=self.erf)
			behavior.planner.set_runtime(sys_time=self.rostime)
			behavior.planner.constraints.set_feasibility_function(self.is_feasible)

		# Reset all management
		self.reset()

		# Subscribers
		rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
		rospy.Subscriber(ogrid_topic, OccupancyGrid, self.ogrid_cb)

		# Publishers
		self.ref_pub = rospy.Publisher(ref_topic, Odometry, queue_size=1)
		self.path_pub = rospy.Publisher(path_topic, PoseArray, queue_size=3)
		self.tree_pub = rospy.Publisher(tree_topic, PoseArray, queue_size=3)
		self.goal_pub = rospy.Publisher('/lqrrt/goal', PoseStamped, queue_size=3)

		# Actions
		self.move_server = actionlib.SimpleActionServer(move_topic, MoveAction, execute_cb=self.move_cb, auto_start=False)
		self.move_server.start()

		# Timers
		rospy.Timer(rospy.Duration(self.revisit_period), self.publish_ref)
		rospy.Timer(rospy.Duration(self.revisit_period), self.action_check)


	def reset(self):
		"""
		Resets management variables.

		"""
		# Timing stuff
		self.busy = False
		self.next_runtime = None
		self.time_till_issue = None
		self.last_update_time = None

		# Objective stuff
		self.goal = None
		self.ogrid = None
		self.tracking = False

		# Behavior stuff
		self.move_type = None
		self.behavior = None
		self.enroute_behavior = None
		self.goal_bias = None
		self.stuck = False
		self.circle_dir = None

		# Plan stuff
		self.get_ref = None
		self.x_seq = None
		self.u_seq = None
		self.tree = None
		self.next_seed = None

		# Unkill all planners
		for behavior in self.behaviors_list:
			behavior.planner.unkill()

################################################# ACTION

	def move_cb(self, msg):
		"""

		"""
		# Make sure odom is publishing (well, at least once)
		if self.state is None:
			print("Cannot plan until odom is received!\n")
			self.move_server.set_aborted(MoveResult(String('odom')))
			return False

		# Reset the whole planner system for safety
		self.reset()

		# Check given move_type
		if msg.move_type.data in ['hold', 'drive', 'skid', 'circle']:
			print("Preparing: {}".format(msg.move_type.data))
			self.move_type = msg.move_type.data
		else:
			print("Unsupported move_type: '{}'\n".format(msg.move_type.data))
			self.move_server.set_aborted(MoveResult(String('move_type')))
			return False

		# Check given focus, skid case
		if self.move_type == 'skid':
			if msg.focus.z == 0:
				print("Holding orientation.")
				boat.focus = None
			else:
				boat.focus = np.array([msg.focus.x, msg.focus.y])
				print("Focused on: {}".format(boat.focus[:2]))

		# Check given focus, circle case
		elif self.move_type == 'circle':
			boat.focus = np.array([msg.focus.x, msg.focus.y])
			if msg.focus.z >= 0:
				print("Focused on: {}, counterclockwise".format(boat.focus[:2]))
				self.circle_dir = 1
			else:
				print("Focused on: {}, clockwise".format(boat.focus[:2]))
				self.circle_dir = -1
		
		# Check given focus, all other cases
		else:
			boat.focus = None

		# Station keeping
		if self.move_type == 'hold':
			self.set_goal(self.state)
			self.get_ref = lambda t: self.goal
			self.move_server.set_succeeded(MoveResult())
			print("Done!\n")
			return True

		# Circling
		elif self.move_type == 'circle':
			print("Circle moves are not implemented yet!\n")
			self.move_server.set_aborted(MoveResult(String('patience')))
			return False

		# More typical moves
		else:
			# Give desired pose to everyone who needs it
			self.set_goal(self.unpack_pose(msg.goal))

			# Standard driving
			if self.move_type == 'drive':

				# Find the heading that points to the goal
				p_err = self.goal[:2] - self.state[:2]
				h_goal = np.arctan2(p_err[1], p_err[0])

				# If we aren't within a cone of that heading, construct rotation
				if abs(self.angle_diff(h_goal, self.state[2])) > params.pointshoot_tol:
					dt_rot = np.clip(params.dt, 1E-6, 0.001)
					x_seq_rot, T_rot, rot_success = self.rotation_move(self.state, h_goal, params.pointshoot_tol, dt_rot)

					# If rotation failed, switch to skid
					if not rot_success:
						print("Something is preventing me from rotating completely!\nSwitching to skid.")
						self.move_type = 'skid'

					# Begin interpolating rotation move
					self.get_ref = interp1d(np.arange(len(x_seq_rot))*dt_rot, np.array(x_seq_rot), axis=0, assume_sorted=True,
											bounds_error=False, fill_value=x_seq_rot[-1][:])

				# Start tree-chaining with the end of the rotation move
				self.last_update_time = self.rostime()
				self.next_seed = np.copy(x_seq_rot[-1])
				self.next_runtime = np.clip(T_rot, params.basic_duration, 2*np.pi/params.velmax_pos_plan[2])

			# Translate or look-at move
			elif self.move_type == 'skid':
				self.next_seed = np.copy(self.state)
				self.next_runtime = params.basic_duration

			# (debug)
			assert self.next_seed is not None
			assert self.next_runtime is not None

			# Begin tree-chaining loop
			while True:
				self.tree_chain()

				# Print feedback
				print("Move Details\n----")
				print("Behavior: {}".format(self.behavior.__name__))
				print("Goal bias: {}".format(self.goal_bias))
				print("Tree size: {}".format(self.tree.size))
				print("ETA: {}\n".format(self.next_runtime))

				# Check if action goal is complete
				if np.all(np.abs(self.erf(self.goal, self.state)) <= params.real_tol):
					break
				else:
					rospy.sleep(self.revisit_period)

			# Over and out!
			remain = np.copy(self.goal)
			self.get_ref = lambda t: remain
			self.move_server.set_succeeded(MoveResult())
			print("Done!\n")
			return True

################################################# MOVE HELPERS

	def rotation_move(self, x, h, tol, dt=0.01):
		"""

		"""
		# Set-up
		x = np.array(x, dtype=np.float64)
		xg = np.copy(x); xg[2] = h
		x_seq = []; T = 0; i = 0
		u = np.zeros(3)

		# Simulate rotation move
		while True:

			# Stop if pose is infeasible
			if not self.is_feasible(x, np.zeros(3)) and len(x_seq):
				return (x_seq, T, False)
			else:
				x_seq.append(x)

			# Keep rotating towards goal until tolerance is met
			e = self.erf(xg, x)
			if abs(e[2]) <= tol:
				return (x_seq, T, True)

			# Step
			u = boat.lqr(x, u)[1].dot(e)
			x = boat.dynamics(x, u, dt)
			T += dt
			i += 1


	def circle_move(self):
		"""

		"""
		pass

################################################# WHERE IT HAPPENS

	def tree_chain(self):
		"""

		"""
		# Make sure we are not currently in an update
		if self.busy:
			return

		# Thread locking, lol
		self.busy = True

		# No crisis
		if self.time_till_issue is None:
			self.behavior = self.select_behavior()
			self.goal_bias = self.select_bias()

		# Distant crisis
		elif self.time_till_issue >= params.basic_duration:
			self.behavior = self.select_behavior()
			self.goal_bias = self.select_bias()
			self.next_runtime = params.basic_duration
			self.next_seed = self.get_ref(self.next_runtime + self.rostime() - self.last_update_time)
		
		# Immediate crisis
		else:
			self.behavior = escape
			self.goal_bias = 0
			self.next_runtime = self.time_till_issue
			self.next_seed = self.get_ref(self.next_runtime + self.rostime() - self.last_update_time)

		# (debug)
		if self.next_runtime == None:
			assert self.stuck and self.time_till_issue is None

		# Update plan
		clean_update = self.behavior.planner.update_plan(x0=self.next_seed,
														 sample_space=self.behavior.gen_ss(self.next_seed, self.goal),
														 goal_bias=self.goal_bias,
														 specific_time=self.next_runtime)

		# Finish up
		if clean_update:
			if self.behavior.planner.tree.size == 1 and self.next_runtime > params.dt:
				print("I think we're stuck...hm")
				self.stuck = True
			else:
				self.stuck = False
			self.last_update_time = self.rostime()
			self.get_ref = self.behavior.planner.get_state
			self.x_seq = self.behavior.planner.x_seq
			self.u_seq = self.behavior.planner.u_seq
			self.tree = self.behavior.planner.tree
			self.next_runtime = self.fudge_factor * self.behavior.planner.T
			self.next_seed = self.get_ref(self.next_runtime)
			self.enroute_behavior = self.behavior

		# Make sure all planners are actually unkilled
		for behavior in self.behaviors_list:
			behavior.planner.unkill()

		# Unlocking, lol
		self.busy = False

		# Visualizers
		self.publish_tree()
		self.publish_path()

################################################# DECISIONS

	def select_behavior(self):
		"""

		"""
		# Are we stuck?
		if self.stuck:
			self.next_runtime = None
			return escape

		# Are we driving?
		if self.move_type == 'drive':
			# All clear?
			if npl.norm(self.goal[:2] - self.next_seed[:2]) < params.free_radius:
				return boat
			else:
				return car

		# Are we skidding?
		if self.move_type == 'skid':
			# Got a focus?
			if boat.focus is None:
				return particle
			else:
				return boat

		# (debug)
		raise ValueError("Indeterminant behavior configuration.")


	def select_bias(self):
		"""

		"""
		#<<< use ogrid
		return [0.5, 0.5, 0, 0, 0, 0]

################################################# VERIFICATIONS

	def is_feasible(self, x, u):
		"""

		"""
		# Reject going too fast
		for i, v in enumerate(x[3:]):
			if v > params.velmax_pos_plan[i] or v < params.velmax_neg_plan[i]:
				return False

		# If there's no ogrid yet, anywhere is valid
		if self.ogrid is None:
			return True

		# Body to world
		c, s = np.cos(x[2]), np.sin(x[2])
		R = np.array([[c, -s],
					  [s,  c]])

		# Vehicle points in world frame
		points = x[:2] + R.dot(params.vps).T

		# Check for collision
		indicies = (self.ogrid_cpm * (points - self.ogrid_origin)).astype(np.int64)
		try:
			grid_values = self.ogrid[indicies[:, 1], indicies[:, 0]]
		except IndexError:
			print("WOAH NELLY! Search exceeded ogrid size.")
			return False

		# Assuming anything greater than 90 is a hit
		return np.all(grid_values < 90)


	def reevaluate_plan(self):
		"""

		"""
		# Make sure we are not already fixing the plan
		if self.time_till_issue is not None:
			return

		# Make sure a plan exists
		if self.enroute_behavior is None:
			return

		# Time since last update
		iters_passed = int((self.rostime() - self.last_update_time) / params.dt)

		# Check that all points in the plan are still feasible
		p_seq = np.copy(self.x_seq[iters_passed:])
		p_seq[:, 3:] = 0
		for i, (x, u) in enumerate(zip(p_seq, [np.zeros(3)]*len(p_seq))):
			if not self.is_feasible(x, u):
				print("Found collision on current path!")
				self.time_till_issue = i*params.dt
				self.behavior.planner.kill_update()
				return

		# If we are escaping, check if we have a clear path again
		if self.enroute_behavior.__name__ == 'escape':
			start = self.get_ref(self.rostime() - self.last_update_time)
			xline = np.arange(start[0], self.goal[0], params.boat_width)
			yline = np.linspace(start[1], self.goal[1], len(xline))
			sline = np.vstack((xline, yline, np.zeros((4, len(xline))))).T
			checks = []
			for x in sline:
				checks.append(self.is_feasible(x, np.zeros(3)))
			if np.all(checks):
				print("Done escaping!")
				self.time_till_issue = self.basic_duration
				self.behavior.planner.kill_update()
				return

		# No concerns
		self.time_till_issue = None


	def action_check(self, *args):
		"""

		"""
		if not self.move_server.is_active():
			return

		if self.move_server.is_preempt_requested():
			self.move_server.set_preempted()
			print("Action preempted!")
			self.reset()

		if self.enroute_behavior is not None and self.tree is not None and self.tracking is not None and \
		   self.next_runtime is not None and self.last_update_time is not None:
			self.move_server.publish_feedback(MoveFeedback(String(self.enroute_behavior.__name__),
														   Int64(self.tree.size),
														   Bool(self.tracking),
														   Float64(self.next_runtime - (self.rostime() - self.last_update_time))))

################################################# LIL MATH DOERS

	def set_goal(self, x):
		"""
		Gives a goal state x out to everyone who needs it.

		"""
		self.goal = np.copy(x)
		for behavior in self.behaviors_list:
			behavior.planner.set_goal(self.goal)
		self.goal_pub.publish(self.pack_posestamped(np.copy(self.goal), rospy.Time.now()))


	def erf(self, xgoal, x):
		"""
		Returns error e given two states xgoal and x.
		Angle differences are taken properly on SO2.

		"""
		e = np.subtract(xgoal, x)
		e[2] = self.angle_diff(xgoal[2], x[2])
		return e


	def angle_diff(self, agoal, a):
		"""
		Takes an angle difference properly on SO2.

		"""
		c = np.cos(a)
		s = np.sin(a)
		cg = np.cos(agoal)
		sg = np.sin(agoal)
		return np.arctan2(sg*c - cg*s, cg*c + sg*s)

################################################# PUBDUBS

	def publish_ref(self, *args):
		"""

		"""
		# Make sure a plan exists
		if self.get_ref is None:
			return

		# Time since last update
		if self.last_update_time is None:
			T = self.rostime()
		else:
			T = self.rostime() - self.last_update_time

		# Publish interpolated reference
		self.ref_pub.publish(self.pack_odom(self.get_ref(T), rospy.Time.now()))


	def publish_path(self):
		"""

		"""
		# Make sure a plan exists
		if self.x_seq is None:
			return

		# Construct pose array and publish
		pose_list = []
		stamp = rospy.Time.now()
		for x in self.x_seq:
			pose_list.append(self.pack_pose(x))
		if len(pose_list):
			msg = PoseArray(poses=pose_list)
			msg.header.frame_id = self.world_frame_id
			self.path_pub.publish(msg)


	def publish_tree(self):
		"""

		"""
		# Make sure a plan exists
		if self.tree is None:
			return

		# Construct pose array and publish
		pose_list = []
		stamp = rospy.Time.now()
		for ID in xrange(self.tree.size):
			x = self.tree.state[ID]
			pose_list.append(self.pack_pose(x))
		if len(pose_list):
			msg = PoseArray(poses=pose_list)
			msg.header.frame_id = self.world_frame_id
			self.tree_pub.publish(msg)

################################################# SUBSCRUBS

	def ogrid_cb(self, msg):
		"""
		Expects an OccupancyGrid message.
		Stores the ogrid array and origin vector.
		Reevaluates the current plan since the ogrid changed.

		"""
		self.ogrid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
		self.ogrid_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
		self.ogrid_cpm = 1 / msg.info.resolution
		self.reevaluate_plan()


	def odom_cb(self, msg):
		"""
		Expects an Odometry message.
		Stores the current state of the vehicle tracking the plan.
		Reference frame information is also recorded.
		Determines if the vehicle is tracking well.

		"""
		self.world_frame_id = msg.header.frame_id
		self.body_frame_id = msg.child_frame_id
		self.state = self.unpack_odom(msg)
		if self.get_ref is not None and self.last_update_time is not None:
			if np.all(np.abs(self.erf(self.get_ref(self.rostime() - self.last_update_time), self.state)) < 2*np.array(params.real_tol)):
				self.tracking = True
			else:
				self.tracking = False

################################################# CONVERTERS

	def unpack_pose(self, msg):
		"""
		Converts a Pose message into a state vector with zero velocity.

		"""
		q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
		return np.array([msg.position.x, msg.position.y, trns.euler_from_quaternion(q)[2], 0, 0, 0])


	def unpack_odom(self, msg):
		"""
		Converts an Odometry message into a state vector.

		"""
		q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
		return np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, trns.euler_from_quaternion(q)[2],
						 msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])


	def pack_pose(self, state):
		"""
		Converts the positional part of a state vector into a Pose message.

		"""
		msg = Pose()
		msg.position.x, msg.position.y = state[:2]
		msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = trns.quaternion_from_euler(0, 0, state[2])
		return msg


	def pack_posestamped(self, state, stamp):
		"""
		Converts the positional part of a state vector into
		a PoseStamped message with a given header timestamp.

		"""
		msg = PoseStamped()
		msg.header.stamp = stamp
		msg.header.frame_id = self.world_frame_id
		msg.pose.position.x, msg.pose.position.y = state[:2]
		msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = trns.quaternion_from_euler(0, 0, state[2])
		return msg


	def pack_odom(self, state, stamp):
		"""
		Converts a state vector into an Odometry message
		with a given header timestamp.

		"""
		msg = Odometry()
		msg.header.stamp = stamp
		msg.header.frame_id = self.world_frame_id
		msg.child_frame_id = self.body_frame_id
		msg.pose.pose.position.x, msg.pose.pose.position.y = state[:2]
		msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = trns.quaternion_from_euler(0, 0, state[2])
		msg.twist.twist.linear.x, msg.twist.twist.linear.y = state[3:5]
		msg.twist.twist.angular.z = state[5]
		return msg

################################################# NODE

if __name__ == "__main__":
	rospy.init_node("LQRRT")
	print("")
	better_than_Astar = LQRRT_Node()
	rospy.spin()
