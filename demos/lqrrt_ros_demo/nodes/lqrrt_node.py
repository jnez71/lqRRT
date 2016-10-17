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

from geometry_msgs.msg import PointStamped, Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid

from behaviors import params, car, boat, escape
from lqrrt_ros_demo.msg import MoveAction, MoveFeedback, MoveResult

################################################# INITIALIZATIONS

class LQRRT_Node(object):

    def __init__(self, odom_topic='/odom', ogrid_topic='/ogrid', ref_topic='/lqrrt/ref',
                 move_topic='/move_to', path_topic='/lqrrt/path', tree_topic='/lqrrt/tree'):
        """
        Initialize with topic names.

        """
        # One-time initializations
        self.revisit_period = 0.05  # s
        self.fudge_factor = 0.9
        self.rostime = lambda: rospy.Time.now().to_sec()
        self.ogrid = None
        self.state = None
        self.tracking = None
        self.busy = False
        self.done = True

        # Set-up planners
        self.behaviors_list = [car, boat, escape]
        for behavior in self.behaviors_list:
            behavior.planner.set_system(erf=self.erf)
            behavior.planner.set_runtime(sys_time=self.rostime)
            behavior.planner.constraints.set_feasibility_function(self.is_feasible)

        # Initialize resetable stuff
        self.reset()

        # Subscribers
        rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        rospy.Subscriber(ogrid_topic, OccupancyGrid, self.ogrid_cb)
        rospy.sleep(0.5)

        # Publishers
        self.ref_pub = rospy.Publisher(ref_topic, Odometry, queue_size=1)
        self.path_pub = rospy.Publisher(path_topic, PoseArray, queue_size=3)
        self.tree_pub = rospy.Publisher(tree_topic, PoseArray, queue_size=3)
        self.goal_pub = rospy.Publisher('/lqrrt/goal', PoseStamped, queue_size=3)
        self.focus_pub = rospy.Publisher('/lqrrt/focus', PointStamped, queue_size=3)

        # Actions
        self.move_server = actionlib.SimpleActionServer(move_topic, MoveAction, execute_cb=self.move_cb, auto_start=False)
        self.move_server.start()
        rospy.sleep(1)

        # Timers
        rospy.Timer(rospy.Duration(self.revisit_period), self.publish_ref)
        rospy.Timer(rospy.Duration(self.revisit_period), self.action_check)


    def reset(self):
        """
        Resets variables that should definitely be cleared before a new action begins.

        """
        # Internal plan
        self.goal = None
        self.get_ref = None
        self.x_seq = None
        self.u_seq = None
        self.tree = None

        # Behavior control
        self.move_type = None
        self.behavior = None
        self.enroute_behavior = None
        self.goal_bias = None
        self.stuck = False
        self.stuck_count = 0

        # Planning control
        self.last_update_time = None
        self.next_runtime = None
        self.next_seed = None
        self.time_till_issue = None
        self.preempted = False

        # Unkill all planners
        for behavior in self.behaviors_list:
            behavior.planner.unkill()

################################################# ACTION

    def move_cb(self, msg):
        """
        Callback for the Move action.

        """
        # Main callback flag
        self.done = False

        # Make sure odom is publishing (well, at least once)
        if self.state is None:
            print("Cannot plan until odom is received!\n")
            self.move_server.set_aborted(MoveResult('odom'))
            self.done = True
            return False
        else:
            print("="*50)

        # Reset the planner system for safety
        self.reset()

        # Give desired pose to everyone who needs it
        self.set_goal(self.unpack_pose(msg.goal))

        # Check given move_type
        if msg.move_type in ['hold', 'drive', 'skid', 'circle']:
            print("Preparing: {}".format(msg.move_type))
            self.move_type = msg.move_type
        else:
            print("Unsupported move_type: '{}'\n".format(msg.move_type))
            self.move_server.set_aborted(MoveResult('move_type'))
            self.done = True
            return False

        # Check given focus
        if self.move_type == 'skid':
            if msg.focus.z == 0:
                boat.focus = None
                self.focus_pub.publish(self.pack_pointstamped([1E6, 1E6, 1E6], rospy.Time.now()))
            else:
                boat.focus = np.array([msg.focus.x, msg.focus.y, 0])
                focus_vec = boat.focus[:2] - self.goal[:2]
                focus_goal = np.copy(self.goal)
                focus_goal[2] = np.arctan2(focus_vec[1], focus_vec[0])
                self.set_goal(focus_goal)
                self.focus_pub.publish(self.pack_pointstamped(boat.focus, rospy.Time.now()))
                print("Focused on: {}".format(boat.focus[:2]))
        elif self.move_type == 'circle':
            boat.focus = np.array([msg.focus.x, msg.focus.y, msg.focus.z])
            self.focus_pub.publish(self.pack_pointstamped(boat.focus, rospy.Time.now()))
            if boat.focus[2] >= 0:
                print("Focused on: {}, counterclockwise".format(boat.focus[:2]))
            else:
                print("Focused on: {}, clockwise".format(boat.focus[:2]))
        else:
            boat.focus = None
            self.focus_pub.publish(self.pack_pointstamped([1E6, 1E6, 1E6], rospy.Time.now()))

        # Station keeping
        if self.move_type == 'hold':
            self.set_goal(self.state)
            self.last_update_time = self.rostime()
            self.get_ref = lambda t: self.goal
            self.move_server.set_succeeded(MoveResult())
            print("\nDone!\n")
            self.done = True
            return True

        # Circling
        elif self.move_type == 'circle':
            print("Circle moves are not implemented yet!\n")
            self.move_server.set_aborted(MoveResult('patience'))
            self.done = True
            return False

        # Standard driving
        elif self.move_type == 'drive':

            # Find the heading that points to the goal
            p_err = self.goal[:2] - self.state[:2]
            h_goal = np.arctan2(p_err[1], p_err[0])

            # If we aren't within a cone of that heading and the goal is far away, construct rotation
            if abs(self.angle_diff(h_goal, self.state[2])) > params.pointshoot_tol and npl.norm(p_err) > params.free_radius:
                dt_rot = np.clip(params.dt, 1E-6, 0.01)
                x_seq_rot, T_rot, rot_success = self.rotation_move(self.state, h_goal, params.pointshoot_tol, dt_rot)
                print("Rotating towards goal (duration: {})".format(np.round(T_rot, 2)))

                # If rotation failed, switch to skid
                if not rot_success:
                    print("Cannot rotate completely! Switching move_type to skid.")
                    self.move_type = 'skid'

                # Begin interpolating rotation move
                self.last_update_time = self.rostime()
                self.get_ref = interp1d(np.arange(len(x_seq_rot))*dt_rot, np.array(x_seq_rot), axis=0,
                                        assume_sorted=True, bounds_error=False, fill_value=x_seq_rot[-1][:])

                # Start tree-chaining with the end of the rotation move
                self.next_runtime = np.clip(T_rot, params.basic_duration, 2*np.pi/params.velmax_pos[2])
                self.next_seed = np.copy(x_seq_rot[-1])

            else:
                self.next_runtime = params.basic_duration
                self.next_seed = np.copy(self.state)

        # Translate or look-at move
        elif self.move_type == 'skid':
            self.next_runtime = params.basic_duration
            self.next_seed = np.copy(self.state)

        # (debug)
        assert self.next_seed is not None
        assert self.next_runtime is not None
        move_number = 1

        # Begin tree-chaining loop
        while not rospy.is_shutdown():
            self.tree_chain()

            # Print feedback
            if not self.preempted:
                print("\nMove {}\n----".format(move_number))
                print("Behavior: {}".format(self.enroute_behavior.__name__[10:]))
                print("Reached goal region: {}".format(self.enroute_behavior.planner.plan_reached_goal))
                print("Goal bias: {}".format(np.round(self.goal_bias, 2)))
                print("Tree size: {}".format(self.tree.size))
                print("Move duration: {}".format(np.round(self.next_runtime, 1)))
            move_number += 1

            # Check if action goal is complete
            if np.all(np.abs(self.erf(self.goal, self.state)) <= params.real_tol):
                break

            # Check for abrupt termination
            if self.preempted:
                print("\nTerminated.")
                self.done = True
                return False

        # Over and out!
        remain = np.copy(self.goal)
        self.get_ref = lambda t: remain
        self.move_server.set_succeeded(MoveResult())
        print("\nDone!\n")
        self.done = True
        return True

################################################# WHERE IT HAPPENS

    def tree_chain(self):
        """
        Plans an lqRRT and sets things up to chain along
        another lqRRT when called again.

        """
        # Make sure we are not currently in an update
        if self.busy:
            return

        # Thread locking, lol
        self.busy = True

        # No issue
        if self.time_till_issue is None:
            self.behavior = self.select_behavior()
            self.goal_bias = self.select_bias()
            if self.next_runtime < params.basic_duration and self.last_update_time is not None and not self.stuck:
                self.next_runtime = params.basic_duration
                self.next_seed = self.get_ref(self.next_runtime + self.rostime() - self.last_update_time)

        # Distant issue
        elif self.time_till_issue > 2*params.basic_duration:
            self.behavior = self.select_behavior()
            self.goal_bias = self.select_bias()
            self.next_runtime = params.basic_duration
            self.next_seed = self.get_ref(self.next_runtime + self.rostime() - self.last_update_time)

        # Immediate issue
        else:
            self.behavior = escape
            self.goal_bias = 0
            self.next_runtime = self.time_till_issue/2
            self.next_seed = self.get_ref(self.next_runtime + self.rostime() - self.last_update_time)

        # (debug)
        if self.next_runtime == None:
            assert self.stuck and self.time_till_issue is None and self.behavior is escape and self.goal_bias == 0

        # Update plan
        clean_update = self.behavior.planner.update_plan(x0=self.next_seed,
                                                         sample_space=self.behavior.gen_ss(self.next_seed, self.goal),
                                                         goal_bias=self.goal_bias,
                                                         specific_time=self.next_runtime)

        # Update finished properly
        if clean_update:

            # We might be stuck if tree is oddly small
            if self.behavior.planner.tree.size <= params.stuck_threshold and \
               not self.behavior.planner.plan_reached_goal and \
               npl.norm(self.goal[:2] - self.state[:2]) > params.free_radius:

                # Increase stuck count towards threshold
                self.stuck_count += 1
                if self.stuck_count > params.stuck_threshold:
                    print("\nI think we're stuck...")
                    self.stuck = True
                    self.stuck_count = 0
            else:
                self.stuck = False
                self.stuck_count = 0

            # Cash-in new goods
            self.enroute_behavior = self.behavior
            self.x_seq = np.copy(self.behavior.planner.x_seq)
            self.u_seq = np.copy(self.behavior.planner.u_seq)
            self.tree = self.behavior.planner.tree
            self.last_update_time = self.rostime()
            self.get_ref = self.behavior.planner.get_state
            self.next_runtime = self.behavior.planner.T
            if self.next_runtime > params.basic_duration:
                self.next_runtime *= self.fudge_factor
            self.next_seed = self.get_ref(self.next_runtime)
            self.time_till_issue = None

            # Visualizers
            self.publish_tree()
            self.publish_path()

        else:
            print("Update cancelled.")

        # Make sure all planners are actually unkilled
        for behavior in self.behaviors_list:
            behavior.planner.unkill()

        # Unlocking, lol
        self.busy = False

################################################# DECISIONS

    def select_behavior(self):
        """
        Chooses the behavior for a given move.

        """
        # Are we stuck?
        if self.stuck:
            self.next_runtime = None
            return escape

        # Positional error norm of next_seed
        dist = npl.norm(self.goal[:2] - self.next_seed[:2])

        # Are we driving?
        if self.move_type == 'drive':
            # All clear?
            if dist < params.free_radius:
                return boat
            else:
                return car

        # Are we skidding?
        if self.move_type == 'skid':
            return boat

        # (debug)
        raise ValueError("Indeterminant behavior configuration.")


    def select_bias(self):
        """
        Chooses the goal bias for a given move.

        """
        # Escaping means maximal exploration
        if self.behavior is escape:
            return 0

        # Use ogrid to find good bias
        if self.ogrid is not None and self.next_seed is not None:
            xline = np.arange(self.next_seed[0], self.goal[0], params.vps_spacing)
            yline = np.linspace(self.next_seed[1], self.goal[1], len(xline))
            sline = np.vstack((xline, yline, np.zeros((4, len(xline))))).T
            frees = 0; total = 0
            for x in sline:
                if self.is_feasible(x, np.zeros(3)):
                    frees += 1
                total += 1
            if total != 0:
                b = np.clip(frees / total, 0.2, 0.9)
            else:
                b = 1
        else:
            b = 1

        # For boating, no focus means hold goal orientation
        if self.behavior is boat:
            if npl.norm(self.goal[:2] - self.next_seed[:2]) < params.free_radius:
                return [1, 1, 1, 0.1, 0.1, 0]
            else:
                return [b, b, 1, 0, 0, 1]

        # For car-ing, just don't bias too much
        if self.behavior is car:
            bc = np.clip(b, 0, 0.75)
            return [bc, bc, 0, 0, 0.5, 0]

        # (debug)
        raise ValueError("Indeterminant behavior configuration.")

################################################# VERIFICATIONS

    def is_feasible(self, x, u):
        """
        Given a state x and effort u, returns a bool
        that is only True if that (x, u) is feasible.

        """
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
        Iterates through the current plan re-checking for
        feasibility using the newest ogrid data.

        """
        # Make sure we are not already fixing the plan
        if self.time_till_issue is not None:
            return

        # Make sure a plan exists
        if self.last_update_time is None or self.x_seq is None:
            return

        # Timesteps since last update
        iters_passed = int((self.rostime() - self.last_update_time) / params.dt)

        # Check that all points in the plan are still feasible
        p_seq = np.copy(self.x_seq[iters_passed:])
        if len(p_seq):
            p_seq[:, 3:] = 0
            for i, (x, u) in enumerate(zip(p_seq, [np.zeros(3)]*len(p_seq))):
                if not self.is_feasible(x, u):
                    self.time_till_issue = i*params.dt
                    for behavior in self.behaviors_list:
                        behavior.planner.kill_update()
                    print("\nFound collision on current path!\nTime till collision: {}".format(self.time_till_issue))
                    return

        # If we are escaping, check if we have a clear path again
        if self.enroute_behavior is escape:
            start = self.get_ref(self.rostime() - self.last_update_time)
            xline = np.arange(start[0], self.goal[0], params.vps_spacing)
            yline = np.linspace(start[1], self.goal[1], len(xline))
            sline = np.vstack((xline, yline, np.zeros((4, len(xline))))).T
            checks = []
            for x in sline:
                checks.append(self.is_feasible(x, np.zeros(3)))
            if np.all(checks):
                self.time_till_issue = 1E2
                for behavior in self.behaviors_list:
                    behavior.planner.kill_update()
                self.stuck = False
                self.stuck_count = 0
                print("\nDone escaping!")
                return

        # No concerns
        self.time_till_issue = None


    def action_check(self, *args):
        """
        Manages action preempting.

        """
        if not self.move_server.is_active():
            return

        if self.move_server.is_preempt_requested() or (rospy.is_shutdown() and self.busy):
            self.move_server.set_preempted()
            self.preempted = True
            print("\nAction preempted!")
            if self.behavior is not None:
                print("Killing planners.")
                for behavior in self.behaviors_list:
                    behavior.planner.kill_update()
                while not self.done:
                    rospy.sleep(0.1)
            print("\n")
            self.reset()

        if self.enroute_behavior is not None and self.tree is not None and self.tracking is not None and \
           self.next_runtime is not None and self.last_update_time is not None:
            self.move_server.publish_feedback(MoveFeedback(self.enroute_behavior.__name__[10:],
                                                           self.tree.size,
                                                           self.tracking,
                                                           self.next_runtime - (self.rostime() - self.last_update_time)))

################################################# LIL MATH DOERS

    def rotation_move(self, x, h, tol, dt=0.01):
        """
        Returns a state sequence, total time, and success bool for
        a simple rotate in place move. Success is False if the move
        becomes infeasible before the state heading x[2] is within
        the goal heading h +- tol. Simulation timestep is dt.

        """
        # Set-up
        x = np.array(x, dtype=np.float64)
        xg = np.copy(x); xg[2] = h
        x_seq = []; T = 0; i = 0
        u = np.zeros(3)

        # Simulate rotation move
        while not rospy.is_shutdown():

            # Stop if pose is infeasible
            if not self.is_feasible(np.concatenate((x[:3], np.zeros(3))), np.zeros(3)) and len(x_seq):
                return (x_seq, T, False)
            else:
                x_seq.append(x)

            # Keep rotating towards goal until tolerance is met
            e = self.erf(xg, x)
            if abs(e[2]) <= tol:
                return (x_seq, T, True)

            # Step
            u = 3*boat.lqr(x, u)[1].dot(e)
            x = boat.dynamics(x, u, dt)
            T += dt
            i += 1


    def circle_move(self):
        """

        """
        pass
        #<<<


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

    def set_goal(self, x):
        """
        Gives a goal state x out to everyone who needs it.

        """
        self.goal = np.copy(x)
        for behavior in self.behaviors_list:
            behavior.planner.set_goal(self.goal)
        self.goal_pub.publish(self.pack_posestamped(np.copy(self.goal), rospy.Time.now()))


    def publish_ref(self, *args):
        """
        Publishes the reference trajectory as an Odometry message.

        """
        # Make sure a plan exists
        if self.get_ref is None:
            return

        # Time since last update
        T = self.rostime() - self.last_update_time

        # Publish interpolated reference
        self.ref_pub.publish(self.pack_odom(self.get_ref(T), rospy.Time.now()))


    def publish_path(self):
        """
        Publishes all tree-node poses along the current path as a PoseArray.

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
        Publishes all tree-node poses as a PoseArray.

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


    def pack_pointstamped(self, point, stamp):
        """
        Converts a point vector into a PointStamped
        message with a given header timestamp.

        """
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame_id
        msg.point.x, msg.point.y = point[:2]
        msg.point.z = 0
        return msg

################################################# NODE

if __name__ == "__main__":
    rospy.init_node("LQRRT")
    print("")
    better_than_Astar = LQRRT_Node()
    rospy.spin()
