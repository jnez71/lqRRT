#!/usr/bin/env python
"""
Node that subscribes to a goal (PoseStamped message), the current boat
state (Odometry message), and the world-frame ogrid (OccupancyGrid
message). It publishes the trajectory that heads to the goal as an
Odometry message, as well as the path and tree as PoseArray messages.
It also publishes a Bool message for if the goal is currently achieved.

"""
from __future__ import division
import numpy as np
import numpy.linalg as npl
import behaviors

import rospy
import tf.transformations as trns
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid

################################################# INITIALIZATION

class LQRRT_Node(object):
	"""
	Basic functionality is described in the lqrrt_node.py docstring.

	"""
	def __init__(self, odom_topic='/odom', ogrid_topic='/ogrid', goal_topic='/lqrrt_goal',
				 planstate_topic='/traj', path_topic='/path', tree_topic='/tree', done_topic='/move_complete'):
		"""
		Initialize with topic names.
		First three are for subscribing, last four are for publishing.

		"""
		# Behavior stuff
		self.free_radius = npl.norm(behaviors.fcar.goal_buffer[:2])
		self.stuck = False
		self.collision_warning = False

		# Timing stuff
		self.busy = False
		self.reevaluating = False
		self.time_remaining = 0
		self.get_rostime = lambda: rospy.Time.now().to_sec()
		self.last_update_time = self.get_rostime()
		self.fudge_factor = 0.95

		# Start-up stuff
		self.x_seq = None
		self.ogrid = None
		self.waiting_for_odom = True
		self.give_odom_warning = True
		self.odom_topic = odom_topic
		self.good_update = True

		# Set-up planners
		self.behavior = None
		self.get_planstate = lambda t: np.copy(self.state)
		self.behaviors_list = [behaviors.direct, behaviors.boat, behaviors.fcar, behaviors.escape]
		for behavior in self.behaviors_list:
			behavior.planner.set_system_time(self.get_rostime)
			behavior.planner.constraints.set_feasibility_function(self.is_feasible)

		# Subscribers
		rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
		rospy.Subscriber(ogrid_topic, OccupancyGrid, self.ogrid_cb)
		rospy.Subscriber(goal_topic, PoseStamped, self.goal_cb)

		# Publishers
		self.planstate_pub = rospy.Publisher(planstate_topic, Odometry, queue_size=1)
		self.path_pub = rospy.Publisher(path_topic, PoseArray, queue_size=1)
		self.tree_pub = rospy.Publisher(tree_topic, PoseArray, queue_size=1)
		self.done_pub = rospy.Publisher(done_topic, Bool, queue_size=1)

		# Revisit timers
		rospy.Timer(rospy.Duration(0.05), self.replan)
		rospy.Timer(rospy.Duration(0.05), self.publish_planstate)
		# rospy.Timer(rospy.Duration(2), self.publish_path)
		# rospy.Timer(rospy.Duration(2), self.publish_tree)

################################################# PROCESSING ROUTINES

	def replan(self, *args):
		"""

		"""
		# Make sure first odom was received
		if self.waiting_for_odom:
			if self.give_odom_warning:
				print("(waiting for {})".format(self.odom_topic))
				self.give_odom_warning = False
			return

		# Make sure we are not currently in an update
		if self.busy:
			return

		# Thread locking, lol
		self.busy = True

		# Select which behavior to use
		self.behavior = self.select_behavior()

		# Clip the duration of the update
		if self.time_remaining < self.behavior.min_time or np.isinf(self.time_remaining):
			self.time_remaining = self.behavior.min_time
			self.next_seed = self.get_planstate(self.time_remaining + (self.get_rostime() - self.last_update_time))

		# Update plan
		print("\nBehavior: {}".format(self.behavior.__name__))
		self.good_update = self.behavior.planner.update_plan(x0=self.next_seed,
															 sample_space=self.behavior.gen_ss(self.next_seed, self.goal),
															 goal_bias=self.behavior.goal_bias,
															 specific_time=self.time_remaining)

		# Finish up
		if self.good_update:
			if self.behavior.planner.tree.size == 1 and not self.behavior.planner.plan_reached_goal:
				self.stuck = True
				print("I think we're stuck...hm")
			else:
				self.stuck = False
			self.time_remaining = self.fudge_factor * self.behavior.planner.T
			self.get_planstate = self.behavior.planner.get_state
			self.x_seq = self.behavior.planner.x_seq
			self.u_seq = self.behavior.planner.u_seq
			self.tree = self.behavior.planner.tree
			self.next_seed = self.get_planstate(self.time_remaining)
			self.last_update_time = self.get_rostime()
			self.good_update = True
			self.reevaluating = False
		else:
			self.stuck = False
			self.good_update = False
			self.time_remaining = 0

		# Make sure all planners are actually unkilled
		for behavior in self.behaviors_list:
			behavior.planner.unkill()

		# Unlocking, lol
		self.busy = False

		# Visualizers
		self.publish_tree()
		self.publish_path()


	def select_behavior(self):
		"""

		"""
		# Is a collision about to happen?
		if self.collision_warning:
			self.collision_warning = False
			return behaviors.direct

		# Are we stuck?
		if self.stuck:
			return behaviors.escape

		# Are we in the freedom region where all is safe?
		err = self.goal[:2] - self.next_seed[:2]
		if npl.norm(err) < self.free_radius:
			return behaviors.direct

		# Are we pointed toward the goal?
		hvec = np.array([np.cos(self.next_seed[2]), np.sin(self.next_seed[2])])
		if err.dot(hvec) > 0:
			return behaviors.fcar

		# Otherwise
		return behaviors.boat


	def is_feasible(self, x, u):
		"""

		"""
		# Reject going too fast
		if self.behavior.__name__ == 'behaviors.fcar':
			for i, v in enumerate(x[3:]):
				if v > behaviors.params.velmax_pos_plan[i] or v < behaviors.params.velmax_neg_plan[i]:
					return False

		# If there's no ogrid yet, anywhere is valid
		if self.ogrid is None:
			return True

		# Body to world
		c, s = np.cos(x[2]), np.sin(x[2])
		R = np.array([[c, -s],
					  [s,  c]])

		# Vehicle points in world frame
		points = x[:2] + R.dot(behaviors.params.vps).T

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
		if self.reevaluating:
			return

		# Make sure a plan exists
		if self.x_seq is None:
			return

		# Time since last update
		iters_passed = int((self.get_rostime() - self.last_update_time) / behaviors.params.dt)

		# Check that all points in the plan are still feasible
		for i, (x, u) in enumerate(zip(self.x_seq[iters_passed:], self.u_seq[iters_passed:])):
			if not self.is_feasible(x, u):
				self.reevaluating = True
				if i*behaviors.params.dt < behaviors.fcar.min_time:
					self.collision_warning = True
				self.behavior.planner.kill_update()
				print("Found collision on current path!")
				return

################################################# PUBLISHING ROUTINES

	def publish_planstate(self, *args):
		"""

		"""
		# Make sure a plan exists
		if self.x_seq is None:
			return

		# Time since last update
		T = self.get_rostime() - self.last_update_time

		# Publish interpolated planstate
		self.planstate_pub.publish(self.pack_odom(self.get_planstate(T), rospy.Time.now()))


	def publish_path(self, *args):
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


	def publish_tree(self, *args):
		"""

		"""
		# Make sure a plan exists
		if self.x_seq is None:
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

################################################# SUBSCRIBER CALLBACKS

	def goal_cb(self, msg):
		"""
		Expects a PoseStamped message.
		Stores the goal state and gives it to all planners.
		Then resets execution of the current planner's updating.

		"""
		self.goal = self.unpack_pose(msg)
		for behavior in self.behaviors_list:
			behavior.planner.set_goal(self.goal)
		if self.x_seq is not None:
			self.behavior.planner.kill_update()
			print("Got new goal!")


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
		Publishes the plan completion status.
		Reference frame information is also recorded.
		Also sets the first seed and goal to the initial state.

		"""
		self.world_frame_id = msg.header.frame_id
		self.body_frame_id = msg.child_frame_id
		self.state = self.unpack_odom(msg)
		if self.waiting_for_odom:
			self.next_seed = np.copy(self.state)
			self.goal_cb(self.pack_posestamped(self.state, rospy.Time.now()))
			self.waiting_for_odom = False
		if np.all(np.abs(behaviors.params.erf(self.goal, self.state)) <= behaviors.params.real_tol):
			self.done_pub.publish(Bool(data=True))
		else:
			self.done_pub.publish(Bool(data=False))

################################################# MESSAGE CONVERSIONS

	def unpack_pose(self, msg):
		"""
		Converts a PoseStamped message into a state vector with zero velocity.

		"""
		q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
		return np.array([msg.pose.position.x, msg.pose.position.y, trns.euler_from_quaternion(q)[2], 0, 0, 0])


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

################################################# ROS NODE

if __name__ == "__main__":
	rospy.init_node("LQRRT")
	better_than_Astar = LQRRT_Node()
	rospy.spin()
