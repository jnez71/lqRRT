#!/usr/bin/env python
"""
For now, just publishes a fake odom.
Eventually will run a basic controller and sim to show tracking.

"""
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

rospy.init_node("odom_gen")
odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
boatfront_pub = rospy.Publisher('/boatfront', PoseStamped, queue_size=1)
boatback_pub = rospy.Publisher('/boatback', PoseStamped, queue_size=1)

def odom_gen(*args):
	o = Odometry()
	o.header.frame_id = '/world'
	o.child_frame_id = '/body'
	odom_pub.publish(o)

def planstate_cb(msg):
	boatfront_pub.publish(PoseStamped(pose=msg.pose.pose, header=msg.header))
	boatback_pub.publish(PoseStamped(pose=msg.pose.pose, header=msg.header))

rospy.Subscriber('/traj', Odometry, planstate_cb)

rospy.Timer(rospy.Duration(0.05), odom_gen)

rospy.spin()
