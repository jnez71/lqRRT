#!/usr/bin/env python
"""
For now, just publishes a fake odom that perfectly tracks the plan.
Eventually will run a basic controller and sim to show tracking.
Two PoseStamped messages are also published to allow for displaying
the size of the boat in rviz.

"""
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

rospy.init_node("odom_gen")
odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
boatfront_pub = rospy.Publisher('/boatfront', PoseStamped, queue_size=1)
boatback_pub = rospy.Publisher('/boatback', PoseStamped, queue_size=1)

odom = Odometry()

def odom_gen(*args):
	global odom
	o = odom
	o.header.frame_id = '/world'
	o.child_frame_id = '/body'
	odom_pub.publish(o)

def planstate_cb(msg):
	global odom
	boatfront_pub.publish(PoseStamped(pose=msg.pose.pose, header=msg.header))
	boatback_pub.publish(PoseStamped(pose=msg.pose.pose, header=msg.header))
	odom = msg

rospy.Subscriber('/ref', Odometry, planstate_cb)

rospy.Timer(rospy.Duration(0.05), odom_gen)

rospy.spin()
