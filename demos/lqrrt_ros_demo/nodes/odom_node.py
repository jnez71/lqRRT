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
vehicle_pub = rospy.Publisher('/vehicle', PoseStamped, queue_size=1)

odom = Odometry()

def odom_gen(*args):
    global odom
    o = odom
    o.header.frame_id = '/world'
    o.child_frame_id = '/body'
    odom_pub.publish(o)

def ref_cb(msg):
    global odom
    vehicle_pub.publish(PoseStamped(pose=msg.pose.pose, header=msg.header))
    odom = msg

rospy.Subscriber('/lqrrt/ref', Odometry, ref_cb)

rospy.Timer(rospy.Duration(0.05), odom_gen)

rospy.spin()
