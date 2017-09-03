#!/usr/bin/env python
"""
Example of a ROS node that uses lqRRT for a big boat.

This node subscribes to the current boat state (Odometry message),
and the world-frame ogrid (OccupancyGrid message). It publishes the
REFerence trajectory that moves to the goal, as an Odometry message.
It provides an action for moving said reference to that goal (Move.action).

"""
from __future__ import division
import numpy as np
import numpy.linalg as npl
import cv2  # for occupancy grid analysis

import rospy
import actionlib
import tf.transformations as trns

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point32, PointStamped, Pose, PoseArray, \
                              PoseStamped, WrenchStamped, PolygonStamped

from behaviors import params, car, boat, escape
from lqrrt_ros.msg import MoveAction, MoveFeedback, MoveResult

# Check scipy version for assume_sorted argument in interp1d
import scipy.interpolate
if int(scipy.__version__.split('.')[1]) < 16:
    def interp1d(*args, **kwargs):
        kwargs.pop('assume_sorted', None)
        return scipy.interpolate.interp1d(*args, **kwargs)
else:
    interp1d = scipy.interpolate.interp1d

################################################# INITIALIZATIONS

class LQRRT_Node(object):

    def __init__(self, odom_topic, ref_topic, move_topic, path_topic, tree_topic,
                 goal_topic, focus_topic, effort_topic, ogrid_topic, ogrid_threshold):
        """
        Initialize with topic names and ogrid threshold as applicable.
        Defaults are generated at the ROS params level.

        """
        # One-time initializations
        self.revisit_period = 0.05  # s
        self.ogrid = None
        self.ogrid_threshold = float(ogrid_threshold)
        self.state = None
        self.tracking = None
        self.done = True

        # Lil helpers
        self.rostime = lambda: rospy.Time.now().to_sec()
        self.intup = lambda arr: tuple(np.array(arr, dtype=np.int64))
        self.get_hood = lambda img, row, col: img[row-1:row+2, col-1:col+2]

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
        self.goal_pub = rospy.Publisher(goal_topic, PoseStamped, queue_size=3)
        self.focus_pub = rospy.Publisher(focus_topic, PointStamped, queue_size=3)
        self.eff_pub = rospy.Publisher(effort_topic, WrenchStamped, queue_size=3)
        self.sampspace_pub = rospy.Publisher(sampspace_topic, PolygonStamped, queue_size=3)
        self.guide_pub = rospy.Publisher(guide_topic, PointStamped, queue_size=3)

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
        self.get_eff = None
        self.x_seq = None
        self.u_seq = None
        self.tree = None
        self.dt = params.dt

        # Behavior control
        self.move_type = None
        self.behavior = None
        self.enroute_behavior = None
        self.goal_bias = None
        self.sample_space = None
        self.guide = None
        self.speed_factor = np.array(1)

        # Planning control
        self.last_update_time = None
        self.next_runtime = None
        self.next_seed = None
        self.move_count = 0
        self.initial_plan_time = params.basic_duration
        self.blind = False

        # Issue control
        self.stuck = False
        self.stuck_count = 0
        self.fail_count = 0
        self.time_till_issue = None
        self.failure_reason = ''
        self.preempted = False
        self.unreachable = False
        self.collided = False

        # Unkill all planners
        self.lock_tree = False
        for behavior in self.behaviors_list:
            behavior.planner.unkill()

################################################# ACTION

    def move_cb(self, msg):
        """
        Callback for the Move action.

        """
        # Start clean
        self.done = False
        self.reset()
        print("="*50)

        # Make sure odom is publishing (well, at least once)
        if self.state is None:
            print("Cannot plan until odom is received!\n")
            self.move_server.set_aborted(MoveResult('odom'))
            self.done = True
            return False

        # Give desired pose to everyone who needs it
        self.set_goal(self.unpack_pose(msg.goal))

        # Check given move_type
        if msg.move_type in ['hold', 'drive', 'drive!', 'skid', 'spiral', 'bypass']:
            if msg.blind:
                self.blind = True
                print("Preparing: blind {}".format(msg.move_type))
            else:
                print("Preparing: {}".format(msg.move_type))
            self.move_type = msg.move_type
        else:
            print("Unsupported move_type: '{}'\n".format(msg.move_type))
            self.move_server.set_aborted(MoveResult('move_type'))
            self.done = True
            return False

        # Station keeping move
        if self.move_type == 'hold':
            self.set_goal(self.state)
            self.last_update_time = self.rostime()
            remain = np.copy(self.goal)
            self.get_ref = lambda t: remain
            print("\nDone!\n")
            self.move_server.set_succeeded(MoveResult(self.failure_reason))
            self.done = True
            return True

        # Bypass move
        if self.move_type == 'bypass':
            if self.is_feasible(self.goal, np.zeros(3)) or self.blind:
                self.last_update_time = self.rostime()
                remain = np.copy(self.goal)
                self.get_ref = lambda t: remain
                print("\nDone!\n")
                self.move_server.set_succeeded(MoveResult(self.failure_reason))
                self.done = True
                return True
            else:
                print("\nGoal infeasible! Bypass move rejected.\n")
                self.move_server.set_aborted(MoveResult('occupied'))
                self.done = True
                return False

        # Make sure we are not already in a collided state
        if not self.is_feasible(self.state, np.zeros(3)) and not self.blind:
            print("\nCan't move. Already collided.\n")
            self.move_server.set_aborted(MoveResult('collided'))
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
        elif self.move_type == 'spiral':
            boat.focus = np.array([msg.focus.x, msg.focus.y, msg.focus.z])
            self.focus_pub.publish(self.pack_pointstamped(boat.focus, rospy.Time.now()))
            if boat.focus[2] >= 0:
                print("Focused on: {}, counterclockwise".format(boat.focus[:2]))
                if boat.focus[2] == 0:
                    boat.focus[2] = 1
            else:
                print("Focused on: {}, clockwise".format(boat.focus[:2]))
        else:
            boat.focus = None
            self.focus_pub.publish(self.pack_pointstamped([1E6, 1E6, 1E6], rospy.Time.now()))

        # Store the initial planning time, if specified
        if msg.initial_plan_time > 0:
            print("Initial plan time: {}".format(msg.initial_plan_time))
            self.initial_plan_time = msg.initial_plan_time

        # Apply the speed factor, if specified
        msg.speed_factor = np.array(msg.speed_factor).flatten()
        if len(msg.speed_factor) == 0:
            pass
        elif len(msg.speed_factor) not in [1, 3]:
            print("(WARNING: expected speed_factor to be a scalar or in the form [x, y, h])")
        else:
            for i, sf in enumerate(msg.speed_factor):
                if np.isclose(sf, 0) or sf < 0:
                    msg.speed_factor[i] = 1
            if not np.all(msg.speed_factor == 1):
                self.speed_factor = np.copy(msg.speed_factor)
                print("Speed_factor: {}".format(self.speed_factor))
                if np.any(self.speed_factor < 1) and self.dt > 0.05:
                    self.dt *= np.min(self.speed_factor)
                    print("(using smaller timestep: {} s)".format(self.dt))
        for behavior in self.behaviors_list:
            behavior.planner.dt = self.dt
            behavior.velmax_pos = self.speed_factor * params.velmax_pos
            behavior.velmax_neg = self.speed_factor * params.velmax_neg
            behavior.D_pos = params.D_pos / self.speed_factor
            behavior.D_neg = params.D_neg / self.speed_factor

        # Spiraling
        if self.move_type == 'spiral':
            if np.isclose(npl.norm(boat.focus[:2] - self.state[:2]), 0):
                print("\nCan't perform a spiral of zero initial radius! Sorry.")
                print("\nTerminated.\n")
                self.move_server.set_aborted(MoveResult('degen'))
                self.done = True
                return False
            spr_results = self.spiral_move(c=boat.focus[:2],
                                           x=self.state,
                                           N=abs(boat.focus[2]),
                                           direc=np.sign(boat.focus[2]),
                                           mpr=msg.goal.position.z/(2*np.pi))
            x_seq_spr, T_spr, spr_success = spr_results
            if len(x_seq_spr) and spr_success:
                self.x_seq = x_seq_spr
                self.set_goal(x_seq_spr[-1][:])
                self.get_ref = interp1d(np.arange(len(x_seq_spr))*self.dt, np.array(x_seq_spr), axis=0,
                                        assume_sorted=True, bounds_error=False, fill_value=x_seq_spr[-1][:])
                self.last_update_time = self.rostime()
                self.next_runtime = T_spr
                self.lock_tree = True
            else:
                print("\nDesired spiral path is blocked!")
                print("\nTerminated.\n")
                self.move_server.set_aborted(MoveResult('blocked'))
                self.done = True
                return False

        # Standard driving
        elif self.move_type in ['drive', 'drive!']:

            # Find the heading that points to the goal
            p_err = self.goal[:2] - self.state[:2]
            h_goal = np.arctan2(p_err[1], p_err[0])

            # If we aren't within a cone of that heading and the goal is far away, construct rotation
            if abs(self.angle_diff(h_goal, self.state[2])) > params.pointshoot_tol and npl.norm(p_err) > params.free_radius and self.move_type != 'drive!':
                x_seq_rot, T_rot, rot_success, u_seq_rot = self.rotation_move(self.state, h_goal, params.pointshoot_tol)
                print("\nRotating towards goal (duration: {})".format(np.round(T_rot, 2)))
                if not rot_success:
                    print("\n(cannot rotate completely!)")

                # Begin interpolating rotation move
                self.last_update_time = self.rostime()
                if len(x_seq_rot):
                    self.get_ref = interp1d(np.arange(len(x_seq_rot))*self.dt, np.array(x_seq_rot), axis=0,
                                            assume_sorted=True, bounds_error=False, fill_value=x_seq_rot[-1][:])
                    self.get_eff = interp1d(np.arange(len(u_seq_rot))*self.dt, np.array(u_seq_rot), axis=0,
                                            assume_sorted=True, bounds_error=False, fill_value=u_seq_rot[-1][:])
                    self.next_runtime = np.clip(T_rot, params.basic_duration, 2*np.pi/params.velmax_pos[2])
                    self.next_seed = np.copy(x_seq_rot[-1])
                else:
                    self.next_runtime = params.basic_duration
                    self.next_seed = np.copy(self.state)

            else:
                self.next_runtime = params.basic_duration
                self.next_seed = np.copy(self.state)
                if self.move_type == 'drive!':
                    self.move_type = 'drive'

        # Translate or look-at move
        elif self.move_type == 'skid':
            self.next_runtime = params.basic_duration
            self.next_seed = np.copy(self.state)

        # (debug)
        if not self.lock_tree:
            assert self.next_seed is not None
            assert self.next_runtime is not None

        # Begin tree-chaining loop
        while not rospy.is_shutdown():
            clean_update = self.tree_chain()
            self.move_count += 1

            # Print feedback
            if clean_update and not (self.preempted or self.unreachable or self.stuck or self.lock_tree):
                print("\nMove {}\n----".format(self.move_count))
                print("Behavior: {}".format(self.enroute_behavior.__name__[10:]))
                print("Reached goal region: {}".format(self.enroute_behavior.planner.plan_reached_goal))
                print("Goal bias: {}".format(np.round(self.goal_bias, 2)))
                print("Tree size: {}".format(self.tree.size))
                print("Move duration: {}".format(np.round(self.next_runtime, 1)))
            # elif not self.lock_tree:
            #     print("\nIssue Status\n----")
            #     print("Stuck: {}".format(self.stuck))
            #     print("Collided: {}".format(self.collided))
            #     print("Unreachable: {}".format(self.unreachable))
            #     print("Preempted: {}".format(self.preempted))
            #     print("Tree size: {}".format(self.tree.size))
            #     print("Move duration: {}".format(np.round(self.next_runtime, 1)))

            # Check if action goal is complete
            if self.move_type == 'spiral':
                self.publish_path()
                if self.time_till_issue is not None:
                    self.unreachable = True
                    self.failure_reason = 'blocked'
                if (self.rostime() - self.last_update_time) >= T_spr:
                    self.done = True
            elif np.all(np.abs(self.erf(self.goal, self.state)) <= params.real_tol):
                remain = np.copy(self.goal)
                self.get_ref = lambda t: remain
                self.get_eff = lambda t: np.zeros(3)
                self.done = True
            if self.done:
                print("\nDone!\n")
                self.move_server.set_succeeded(MoveResult(self.failure_reason))
                return True

            # Check for abrupt termination or unreachability
            if self.preempted or self.unreachable:
                remain = np.copy(np.concatenate((self.state[:3], np.zeros(3))))
                self.get_ref = lambda t: remain
                self.get_eff = lambda t: np.zeros(3)
                print("\nTerminated.\n")
                self.move_server.set_aborted(MoveResult(self.failure_reason))
                self.done = True
                return False

################################################# WHERE IT HAPPENS

    def tree_chain(self):
        """
        Plans an lqRRT and sets things up to chain along
        another lqRRT when called again.

        """
        # Make sure we are not currently in an update or a collided state
        if self.lock_tree:
            rospy.sleep(0.01)
            return False

        # Thread locking, lol
        self.lock_tree = True

        # No issue
        if self.time_till_issue is None:
            if self.next_runtime < params.basic_duration and self.last_update_time is not None and not self.stuck:
                self.next_runtime = params.basic_duration
                self.next_seed = self.get_ref(self.next_runtime + self.rostime() - self.last_update_time)
            elif self.stuck:
                self.next_runtime = None
            self.behavior = self.select_behavior()
            self.goal_bias, self.sample_space, self.guide, self.pruning = self.select_exploration()

        # Distant issue
        elif self.time_till_issue > 2*params.basic_duration:
            self.next_runtime = params.basic_duration
            self.next_seed = self.get_ref(self.next_runtime + self.rostime() - self.last_update_time)
            self.behavior = self.select_behavior()
            self.goal_bias, self.sample_space, self.guide, self.pruning = self.select_exploration()

        # Immediate issue
        else:
            self.last_update_time = self.rostime()
            remain = np.copy(self.state)
            self.get_ref = lambda t: remain
            self.get_eff = lambda t: np.zeros(3)
            self.next_runtime = params.basic_duration
            self.next_seed = remain
            self.behavior = self.select_behavior()
            self.goal_bias, self.sample_space, self.guide, self.pruning = self.select_exploration()

        # Special first-move case
        if self.move_count == 0:
            if self.get_ref is not None:
                if self.initial_plan_time > self.next_runtime:
                    self.next_runtime = self.initial_plan_time
            else:
                self.next_runtime = self.initial_plan_time

        # (debug)
        if self.stuck and self.time_till_issue is None:
            assert self.next_runtime is None

        # Update plan
        clean_update = self.behavior.planner.update_plan(x0=self.next_seed,
                                                         sample_space=self.sample_space,
                                                         goal_bias=self.goal_bias,
                                                         guide=self.guide,
                                                         pruning=self.pruning,
                                                         specific_time=self.next_runtime)

        # Update finished properly
        if clean_update:
            # We might be stuck if tree is oddly small
            if self.behavior.planner.tree.size == 1:
                # Increase stuck count towards threshold
                self.stuck_count += 1
                if self.stuck_count > params.stuck_threshold:
                    self.fail_count += 1
                    if self.fail_count > params.fail_threshold:
                        print("\nNot making any progress.\nGoal may be unreachable.")
                        self.unreachable = True
                        self.failure_reason = 'unreachable'
                    else:
                        print("\nI think we're stuck...")
                        self.stuck = True
                        self.stuck_count = 0
                else:
                    self.stuck = False
            else:
                self.stuck = False
                self.stuck_count = 0
                self.fail_count = 0

            # Cash-in new goods
            self.x_seq = np.copy(self.behavior.planner.x_seq)
            self.u_seq = np.copy(self.behavior.planner.u_seq)
            self.tree = self.behavior.planner.tree
            self.last_update_time = self.rostime()
            self.get_ref = self.behavior.planner.get_state
            self.get_eff = self.behavior.planner.get_effort
            self.next_runtime = self.behavior.planner.T
            if self.next_runtime > params.basic_duration:
                self.next_runtime *= params.fudge_factor
            self.next_seed = self.get_ref(self.next_runtime)
            self.enroute_behavior = self.behavior
            self.time_till_issue = None

            # Visualizers
            self.publish_tree()
            self.publish_path()
            self.publish_expl()

        # Make sure all planners are actually unkilled
        for behavior in self.behaviors_list:
            behavior.planner.unkill()

        # Unlocking, lol
        self.lock_tree = False

        return clean_update

################################################# DECISIONS

    def select_behavior(self):
        """
        Chooses the behavior for the current move.

        """
        # Are we stuck?
        if self.stuck:
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


    def select_exploration(self):
        """
        Chooses the goal bias, sample space, guide point, and pruning
        choice for the current move and behavior.

        """
        # Escaping means maximal exploration
        if self.behavior is escape:
            if self.guide is None:
                gs = np.copy(self.goal)
                vec = self.goal[:2] - self.next_seed[:2]
                dist = npl.norm(vec)
                if dist < 2*params.free_radius:
                    gs[:2] = vec + 2*params.free_radius*(vec/dist)
            else:
                gs = np.copy(self.guide)
            return(0, escape.gen_ss(self.next_seed, self.goal), gs, True)

        # Analyze ogrid to find good bias and sample space buffer
        if self.ogrid is not None and not self.blind and self.next_seed is not None:

            # Get opencv-ready image from current ogrid (255 is occupied, 0 is clear)
            occ_img = 255*np.greater(self.ogrid, self.ogrid_threshold).astype(np.uint8)

            # Dilate the image
            boat_pix = int(self.ogrid_cpm*params.boat_width)
            boat_pix += boat_pix%2
            occ_img_dial = cv2.dilate(occ_img, np.ones((boat_pix, boat_pix), np.uint8))

            # Construct the initial sample space and get bounds in pixel coordinates
            ss = self.behavior.gen_ss(self.next_seed, self.goal)
            pmin = self.intup(self.ogrid_cpm * ([ss[0][0], ss[1][0]] - self.ogrid_origin))
            pmax = self.intup(self.ogrid_cpm * ([ss[0][1], ss[1][1]] - self.ogrid_origin))

            # Other quantities in pixel coordinates
            seed = self.intup(self.ogrid_cpm * (self.next_seed[:2] - self.ogrid_origin))
            goal = self.intup(self.ogrid_cpm * (self.goal[:2] - self.ogrid_origin))
            step = int(self.ogrid_cpm * params.ss_step)

            # Make sure seed and goal are physically meaningful
            try:
                occ_img_dial[seed[1], seed[0]]
                occ_img_dial[goal[1], goal[0]]
            except IndexError:
                print("Goal and/or seed out of bounds of occupancy grid!")
                return(0, escape.gen_ss(self.next_seed, self.goal, 1), np.copy(self.goal), False)

            # Initializations
            found_entry = False
            push = [0, 0, 0, 0]
            npush = 0
            gs = np.copy(self.goal)
            last_offsets = [pmin[0], pmin[1]]
            ss_img = np.copy(occ_img_dial[pmin[1]:pmax[1], pmin[0]:pmax[0]])
            ss_goal = self.intup(np.subtract(goal, last_offsets))
            ss_seed = self.intup(np.subtract(seed, last_offsets))

            # Iteratively determine how much to push out the sample space
            while np.all(np.less_equal(push, len(occ_img_dial))):
                if found_entry:
                    break

                # Find dividing boundary points
                if np.any(np.equal(ss_img.shape, 0)):
                    print("\nOccupancy grid analysis failed... Please report this!")
                    return(0.5, self.behavior.gen_ss(self.next_seed, self.goal), np.copy(self.goal), False)
                bpts = self.boundary_analysis(ss_img, ss_seed, ss_goal)

                # If already connected, no expansion necessary
                if bpts == 'connected':
                    found_entry = True
                    break

                # If impossible to connect, no sense in expanding
                if bpts == 'isolated':
                    break

                # Otherwise, prepare to push ss based on boundary intercepts
                push_xmin = False
                push_xmax = False
                push_ymin = False
                push_ymax = False

                # Buffered boundary imensions
                row_min = 1; col_min = 1
                row_max = ss_img.shape[0] - 2
                col_max = ss_img.shape[1] - 2

                # Classify boundary points
                for (row, col) in bpts:
                    if col == col_min:  # left
                        push_xmin = True
                        if row == row_min:  # top left
                            push_ymin = True
                        elif row == row_max:  # bottom left
                            push_ymax = True
                    elif col == col_max:  # right
                        push_xmax = True
                        if row == row_min:  # top right
                            push_ymin = True
                        elif row == row_max:  # bottom left
                            push_ymax = True
                    elif row == row_min:  # top
                        push_ymin = True
                    elif row == row_max:  # bottom
                        push_ymax = True

                    # Push accordingly
                    if push_xmin:
                        push[0] += step
                        npush += 1
                    if push_xmax:
                        push[1] += step
                        npush += 1
                    if push_ymin:
                        push[2] += step
                        npush += 1
                    if push_ymax:
                        push[3] += step
                        npush += 1

                    # Get image cropped to sample space and offset points of interest
                    offset_x = (pmin[0]-push[0], pmax[0]+push[1])
                    offset_y = (pmin[1]-push[2], pmax[1]+push[3])
                    ss_img = np.copy(occ_img_dial[offset_y[0]:offset_y[1], offset_x[0]:offset_x[1]])
                    if np.any(np.equal(ss_img.shape, 0)):
                        print("\nOccupancy grid analysis failed... Please report this!")
                        return(0.5, self.behavior.gen_ss(self.next_seed, self.goal), np.copy(self.goal), False)
                    ss_goal = self.intup(np.subtract(goal, [offset_x[0], offset_y[0]]))
                    ss_seed = self.intup(np.subtract(seed, [offset_x[0], offset_y[0]]))
                    test_flood = np.copy(ss_img)
                    cv2.floodFill(test_flood, np.zeros((test_flood.shape[0]+2, test_flood.shape[1]+2), np.uint8), ss_goal, 69)
                    if test_flood[ss_seed[1], ss_seed[0]] == 69:
                        gs[:2] = (np.add([col, row], [last_offsets[0], last_offsets[1]]).astype(np.float64) / self.ogrid_cpm) + self.ogrid_origin
                        found_entry = True
                        break

                # Used for remembering the previous sample space coordinates
                last_offsets = [offset_x[0], offset_y[0]]

            # If we expanded to the limit and found no entry (and goal is unoccupied), goal is infeasible
            if not found_entry and self.is_feasible(self.goal, np.zeros(3)):
                print("\nGoal is unreachable!")
                self.unreachable = True
                self.failure_reason = 'unreachable'
                for behavior in self.behaviors_list:
                    behavior.planner.kill_update()
                return(0, escape.gen_ss(self.next_seed, self.goal, 1), np.copy(self.goal), False)

            # Apply push in real coordinates
            push = np.array(push, dtype=np.float64) / self.ogrid_cpm
            if npush > 0:
                push += params.boat_length
            ss = self.behavior.gen_ss(self.next_seed, self.goal, push + [params.ss_start]*4)

            # Select bias based on density of ogrid in sample space
            if ss_img.size:
                free_ratio = len(np.argwhere(ss_img == 0)) / ss_img.size
                b = np.clip(free_ratio - 0.05*npush, 0, 1)
                if b < 0.9:
                    b -= 0.2
            else:
                b = 1
        else:
            b = 1
            gs = np.copy(self.goal)
            ss = self.behavior.gen_ss(self.next_seed, self.goal)

        # For boating
        if self.behavior is boat:
            if npl.norm(self.goal[:2] - self.next_seed[:2]) < params.free_radius:
                return([1, 1, 1, 0, 0, 0], ss, gs, False)
            else:
                if boat.focus is None:
                    return([b, b, 1, 0, 0, 1], ss, gs, True)
                else:
                    return([b, b, 0, 0, 0, 0], ss, gs, True)

        # For car-ing
        if self.behavior is car:
            return([b, b, 0, 0, 0.5, 0], ss, gs, True)

        # (debug)
        raise ValueError("Indeterminant behavior configuration.")

################################################# VERIFICATIONS

    def is_feasible(self, x, u):
        """
        Given a state x and effort u, returns a bool
        that is only True if that (x, u) is feasible.

        """
        # If there's no ogrid yet or it's a blind move, anywhere is valid
        if self.ogrid is None or self.blind:
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
            return False

        # Greater than threshold is a hit
        return np.all(grid_values < self.ogrid_threshold)


    def reevaluate_plan(self):
        """
        Iterates through the current plan re-checking for
        feasibility using the newest ogrid data, looking for
        a clear path if we are escaping, and checking that
        the goal is still feasible.

        """
        # Make sure a plan exists and that we aren't currently fixing it
        last_update_time = self.last_update_time
        x_seq = np.copy(self.x_seq)
        if last_update_time is None or not np.any(x_seq) \
           or self.done or self.time_till_issue is not None:
            return

        # Timesteps since last update
        iters_passed = int((self.rostime() - last_update_time) / self.dt)

        # Make sure that the goal pose is still feasible
        if not self.is_feasible(self.goal, np.zeros(3)) and self.move_type != 'spiral':
            print("\nThe given goal is occupied!\nGoing nearby instead.")
            self.time_till_issue = np.inf
            self.failure_reason = 'occupied'
            p_err = self.goal[:2] - self.state[:2]
            p_err_mag = npl.norm(p_err)
            if p_err_mag <= npl.norm(params.real_tol[:2]):
                print("...looks like I am already nearby.")
                self.set_goal(self.state)
                return
            npoints = p_err_mag / (params.boat_length/2)
            xline = np.linspace(self.goal[0], self.state[0], npoints)
            yline = np.linspace(self.goal[1], self.state[1], npoints)
            hline = [np.arctan2(p_err[1], p_err[0])] * npoints
            sline = np.vstack((xline, yline, hline, np.zeros((3, npoints)))).T
            found_free_goal = False
            for i, x in enumerate(sline[:-1]):
                if self.is_feasible(x, np.zeros(3)):
                    self.set_goal(sline[i+1])
                    found_free_goal = True
                    break
            if not found_free_goal:
                print("\nCould not find a reasonable free goal!")
                self.unreachable = True
                self.failure_reason = 'unreachable'
                return
            if boat.focus is not None and self.move_type == 'skid':
                focus_vec = boat.focus[:2] - self.goal[:2]
                focus_goal = np.copy(self.goal)
                focus_goal[2] = np.arctan2(focus_vec[1], focus_vec[0])
                if self.is_feasible(focus_goal, np.zeros(3)):
                    self.set_goal(focus_goal)
                else:
                    print("Cancelling focus.")
                    boat.focus = None
            for behavior in self.behaviors_list:
                behavior.planner.kill_update()
            return

        # Check that the next reeval_time seconds in the current plan are still feasible
        p_seq = np.copy(x_seq[iters_passed:min([len(x_seq), int(iters_passed+(params.reeval_time/self.dt)), params.reeval_limit])])
        if len(p_seq):
            p_seq[:, 3:] = 0
            for i, (x, u) in enumerate(zip(p_seq, [np.zeros(3)]*len(p_seq))):
                if not self.is_feasible(x, u):
                    time_till_collision = i*self.dt
                    if time_till_collision <= params.collision_threshold:
                        if not self.collided:
                            print("\nCollided! (*seppuku*)")
                            print("But we cannot throw away our shot!")
                            self.collided = True
                            self.failure_reason = 'collided'
                    else:
                        print("\nFound collision on current path!\nTime till collision: {}".format(time_till_collision))
                        self.time_till_issue = time_till_collision
                        for behavior in self.behaviors_list:
                            behavior.planner.kill_update()
                    return
        if self.collided:
            print("\nNo longer collided! (*unseppuku*)")
            self.collided = False
            if self.failure_reason == 'collided':
                self.failure_reason = ''

        # If we are escaping, check if we have a clear path again
        if self.enroute_behavior is escape:
            start = self.get_ref(self.rostime() - last_update_time)
            p_err = self.goal[:2] - start[:2]
            if npl.norm(p_err) > params.free_radius:
                npoints = npl.norm(p_err) / params.vps_spacing
                xline = np.linspace(start[0], self.goal[0], npoints)
                yline = np.linspace(start[1], self.goal[1], npoints)
                hline = [np.arctan2(p_err[1], p_err[0])] * npoints
                sline = np.vstack((xline, yline, hline, np.zeros((3, npoints)))).T
                checks = []
                for x in sline:
                    checks.append(self.is_feasible(x, np.zeros(3)))
                if np.all(checks):
                    self.time_till_issue = np.inf
                    self.move_type = 'drive'
                    for behavior in self.behaviors_list:
                        behavior.planner.kill_update()
                    self.stuck = False
                    self.stuck_count = 0
                    print("\nClear path found!")
                    return

        # No concerns
        self.time_till_issue = None


    def action_check(self, *args):
        """
        Manages action preempting.

        """
        if self.preempted or not self.move_server.is_active():
            return

        if self.move_server.is_preempt_requested() or (rospy.is_shutdown() and self.lock_tree):
            self.preempted = True
            self.failure_reason = 'preempted'
            print("\nAction preempted!")
            if self.behavior is not None:
                for behavior in self.behaviors_list:
                    behavior.planner.kill_update()
                while not self.done:
                    rospy.sleep(0.1)
            return

        next_runtime = self.next_runtime
        last_update_time = self.last_update_time
        if next_runtime is not None and last_update_time is not None:
            time_till_next_branch = next_runtime - (self.rostime() - last_update_time)
            if self.move_type == 'spiral':
                self.move_server.publish_feedback(MoveFeedback('boat',
                                                               0,
                                                               self.tracking,
                                                               self.erf(self.goal, self.get_ref(self.rostime() - last_update_time))[:3],
                                                               time_till_next_branch))
            elif self.enroute_behavior is not None and self.tree is not None and self.tracking is not None:
                self.move_server.publish_feedback(MoveFeedback(self.enroute_behavior.__name__[10:],
                                                               self.tree.size,
                                                               self.tracking,
                                                               self.erf(self.goal, self.get_ref(self.rostime() - last_update_time))[:3],
                                                               time_till_next_branch))

################################################# LIL MATH DOERS

    def rotation_move(self, x, h, tol):
        """
        Returns a state sequence, total time, success bool and effort
        sequence for a simple rotate in place move. Success is False if
        the move becomes infeasible before the state heading x[2] is within
        the goal heading h+-tol. Give x (start state), and h (goal heading).

        """
        # Set-up
        x = np.array(x, dtype=np.float64)
        xg = np.copy(x); xg[2] = h
        x_seq = []; u_seq = []
        T = 0; i = 0
        u = np.zeros(3)

        # Simulate rotation move
        while not rospy.is_shutdown():

            # Stop if pose is infeasible
            if not self.is_feasible(np.concatenate((x[:3], np.zeros(3))), np.zeros(3)) and len(x_seq):
                portion = params.FPR*len(x_seq)
                return (x_seq[:int(portion)], T-portion*self.dt, False, u_seq[:int(portion)])
            else:
                x_seq.append(x)
                u_seq.append(u)

            # Keep rotating towards goal until tolerance is met
            e = self.erf(xg, x)
            if abs(e[2]) <= tol:
                return (x_seq, T, True, u_seq)

            # Step
            u = 3*boat.lqr(x, u)[1].dot(e)
            x = boat.dynamics(x, u, self.dt)
            T += self.dt
            i += 1


    def spiral_move(self, x, c, N, direc, mpr):
        """
        Returns a state sequence, total time, and success bool for a spiral
        move. Success is False if the move becomes infeasible at any point.
        Give x (initial state), c (center), N (number of revolutions), direc
        (+- z axis), and mpr (meters expansion per radian).

        """
        # Set-up
        if direc < 0:
            mpr *= -1
        x = np.array(x, dtype=np.float64)
        c = np.array(c, dtype=np.float64)
        vt = 0; ang = 0; t = 0
        x_seq = []

        # Radial vector
        r = x[:2] - c
        rmag = npl.norm(r)
        rdir = r / np.clip(rmag, 1E-6, rmag)

        # Tangential velocity and acceleration
        if direc >= 0:
            ht = np.arctan2(rdir[0], -rdir[1])
        else:
            ht = np.arctan2(-rdir[0], rdir[1])
        h0 = self.angle_diff(x[2], ht)
        R_h0 = np.array([[np.cos(h0), -np.sin(h0)], [np.sin(h0), np.cos(h0)]])
        vtmax = abs(R_h0.dot(self.speed_factor*params.velmax_pos[:2])[0])
        atmax = abs(R_h0.dot([params.Fx_max, params.Fy_max])[0] / params.m)
        magic_ratio = 0.5 / 5

        # Simulate spiral move
        anglim = abs(2*np.pi*N)
        while abs(ang) < anglim and not rospy.is_shutdown():

            # Rotation for this instant
            radfactor = np.clip(magic_ratio*rmag, 0, 1)
            vt = np.clip(vt + radfactor*atmax*self.dt, 0, radfactor*vtmax)
            w = direc*vt/np.clip(rmag, 1E-6, rmag)
            dang = w*self.dt
            R = np.array([[np.cos(dang), -np.sin(dang)], [np.sin(dang), np.cos(dang)]])

            # Update
            t += self.dt
            ang += dang
            rdir = R.dot(rdir)
            rmag += mpr*dang

            # Record
            x[:2] = c + rmag*rdir
            x[2] = x[2] + dang
            x[3:5] = R_h0.T.dot([vt, mpr*w])
            x[5] = w
            if self.is_feasible(x, np.zeros(3)):
                x_seq.append(np.copy(x))
            else:
                return x_seq, t, False

        return x_seq, t, True


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


    def boundary_analysis(self, img, seed, goal):
        """
        Returns a list of the two boundary points of the contour dividing seed from
        goal in the occupancy image img. If the seed and goal are connected, returns
        'connected' or if they are terminally isolated, returns 'isolated'. Make sure
        seed and goal are intups and in the same pixel coordinates as img, and that
        occupied pixels have value 255 in img.

        """
        # Set-up
        img = np.copy(img)
        fc1 = 1; fc2 = 2
        bpts = []

        # If goal can flood to seed then done
        flood_goal = np.copy(img)
        cv2.floodFill(flood_goal, np.zeros((flood_goal.shape[0]+2, flood_goal.shape[1]+2), np.uint8), goal, fc1)
        if flood_goal[seed[1], seed[0]] == fc1:
            return 'connected'
        
        # Filter out the dividing boundary
        flood_goal_thresh = fc1*np.equal(flood_goal, fc1).astype(np.uint8)
        flood_seed = np.copy(flood_goal_thresh)
        cv2.floodFill(flood_seed, np.zeros((flood_seed.shape[0]+2, flood_seed.shape[1]+2), np.uint8), seed, fc2)
        flood_seed_thresh = fc2*np.equal(flood_seed, fc2).astype(np.uint8)

        # Buffered boundaries and dimensions
        left = img[1:-1, 1]
        right = img[1:-1, -2]
        top = img[1, 2:-2]
        bottom = img[-2, 2:-2]
        row_min = 1; col_min = 1
        row_max = img.shape[0] - 2
        col_max = img.shape[1] - 2

        # Buffered boundary points that were occupied, in boundary coordinates
        left_cands = np.argwhere(np.equal(left, 255))
        right_cands = np.argwhere(np.equal(right, 255))
        top_cands = np.argwhere(np.equal(top, 255))
        bottom_cands = np.argwhere(np.equal(bottom, 255))

        # Convert to original coordinates
        left_cands = np.hstack((left_cands+1, col_min*np.ones_like(left_cands)))
        right_cands = np.hstack((right_cands+1, col_max*np.ones_like(right_cands)))
        top_cands = np.hstack((row_min*np.ones_like(top_cands), top_cands+2))
        bottom_cands = np.hstack((row_max*np.ones_like(bottom_cands), bottom_cands+2))
        cands = np.vstack((left_cands, right_cands, top_cands, bottom_cands))

        # Iterate through candidates and store the dividing boundary points
        for (row, col) in cands:
            hood = self.get_hood(flood_seed_thresh, row, col)
            if np.any(hood == fc2) and np.any(hood == 0):
                bpts.append([row, col])

        # If they are not connected but there are no boundary points,
        # they are forever isolated
        if len(bpts) == 0:
            return 'isolated'
        else:
            return bpts

################################################# PUBDUBS

    def set_goal(self, x):
        """
        Gives a goal state x out to everyone who needs it and echos
        it as a PoseStamped message.

        """
        self.goal = np.copy(x)
        for behavior in self.behaviors_list:
            behavior.planner.set_goal(self.goal)
        self.goal_pub.publish(self.pack_posestamped(np.copy(self.goal), rospy.Time.now()))


    def publish_ref(self, *args):
        """
        Publishes the reference trajectory as an Odometry message,
        and reference effort as a WrenchStamped message.

        """
        # Make sure a plan exists
        last_update_time = self.last_update_time
        if self.get_ref is None or last_update_time is None:
            return

        # Time since last update
        T = self.rostime() - last_update_time
        stamp = rospy.Time.now()

        # Publish interpolated reference
        self.ref_pub.publish(self.pack_odom(self.get_ref(T), stamp))

        # Not really necessary, but for fun-and-profit also publish the planner's effort wrench
        if self.get_eff is not None:
            self.eff_pub.publish(self.pack_wrenchstamped(self.get_eff(T), stamp))


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


    def publish_expl(self):
        """
        Publishes sample space as a PolygonStamped and
        the guide point as a PointStamped.

        """
        # Make sure a plan exists
        if self.sample_space is None or self.guide is None:
            return

        # Construct and publish
        point_list = [Point32(self.sample_space[0][0], self.sample_space[1][0], 0),
                      Point32(self.sample_space[0][1], self.sample_space[1][0], 0),
                      Point32(self.sample_space[0][1], self.sample_space[1][1], 0),
                      Point32(self.sample_space[0][0], self.sample_space[1][1], 0)]
        ss_msg = PolygonStamped()
        ss_msg.header.frame_id = self.world_frame_id
        ss_msg.polygon.points = point_list
        self.sampspace_pub.publish(ss_msg)
        self.guide_pub.publish(self.pack_pointstamped(self.guide[:2], rospy.Time.now()))

################################################# SUBSCRUBS

    def ogrid_cb(self, msg):
        """
        Expects an OccupancyGrid message.
        Stores the ogrid array and origin vector.
        Reevaluates the current plan since the ogrid changed.

        """
        start = self.rostime()
        self.ogrid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.ogrid_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.ogrid_cpm = 1 / msg.info.resolution
        try:
            self.reevaluate_plan()
        except:
            print("\n(WARNING: something went wrong in reevaluate_plan)\n")
        elapsed = abs(self.rostime() - start)
        if elapsed > 1:
            print("\n(WARNING: ogrid callback is taking {} seconds)\n".format(np.round(elapsed, 2)))


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
        last_update_time = self.last_update_time
        if self.get_ref is not None and last_update_time is not None:
            if np.all(np.abs(self.erf(self.get_ref(self.rostime() - last_update_time), self.state)) < 2*np.array(params.real_tol)):
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


    def pack_wrenchstamped(self, effort, stamp):
        """
        Converts an effort vector into a WrenchStamped message
        with a given header timestamp.

        """
        msg = WrenchStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.body_frame_id
        msg.wrench.force.x, msg.wrench.force.y = effort[:2]
        msg.wrench.torque.z = effort[2]
        return msg

################################################# NODE

if __name__ == "__main__":
    rospy.init_node("lqrrt_node")
    print("")

    move_topic = rospy.get_param("~move_topic", "/move_to")
    odom_topic = rospy.get_param("~odom_topic", "/odom")
    ogrid_topic = rospy.get_param("~ogrid_topic", "/ogrid")
    ogrid_threshold = rospy.get_param("~ogrid_threshold", "90")
    ref_topic = rospy.get_param("~ref_topic", "/lqrrt/ref")
    path_topic = rospy.get_param("~path_topic", "/lqrrt/path")
    tree_topic = rospy.get_param("~tree_topic", "/lqrrt/tree")
    goal_topic = rospy.get_param("~goal_topic", "/lqrrt/goal")
    focus_topic = rospy.get_param("~focus_topic", "/lqrrt/focus")
    effort_topic = rospy.get_param("~effort_topic", "/lqrrt/effort")
    sampspace_topic = rospy.get_param("~sampspace_topic", "/lqrrt/sampspace")
    guide_topic = rospy.get_param("~guide_topic", "/lqrrt/guide")

    better_than_Astar = LQRRT_Node(odom_topic, ref_topic, move_topic,
                                   path_topic, tree_topic, goal_topic,
                                   focus_topic, effort_topic, ogrid_topic,
                                   ogrid_threshold)

    rospy.spin()
