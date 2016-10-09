#!/usr/bin/env python
"""
For now, just publishes a fake odom.
Eventually will run a basic controller and sim to show tracking.

"""
import rospy
from nav_msgs.msg import Odometry

rospy.init_node("odom_gen")

odom_pub = rospy.Publisher('/odom', Odometry, queue_size=0)

def odom_gen(*args):
	o = Odometry()
	o.header.frame_id = '/world'
	o.child_frame_id = '/body'
	odom_pub.publish(o)

rospy.Timer(rospy.Duration(0.05), odom_gen)

rospy.spin()
