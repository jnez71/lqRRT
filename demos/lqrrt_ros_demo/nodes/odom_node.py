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
import tf

rospy.init_node("odom_gen")

odom_topic = rospy.get_param("/lqrrt_node/odom_topic", "/odom")
odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=1)
vehicle_pub = rospy.Publisher('/vehicle', PoseStamped, queue_size=1)

body_world_tf = tf.TransformBroadcaster()
odom = Odometry()

def odom_gen(*args):
    global odom
    o = odom
    o.header.frame_id = '/world'
    o.child_frame_id = '/body'
    odom_pub.publish(o)
    body_world_tf.sendTransform((o.pose.pose.position.x, o.pose.pose.position.y, 0),
                                (o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w),
                                rospy.Time.now(),
                                '/body',
                                '/world')

def ref_cb(msg):
    global odom
    vehicle_pub.publish(PoseStamped(pose=msg.pose.pose, header=msg.header))
    odom = msg

ref_topic = rospy.get_param("/lqrrt_node/ref_topic", "/lqrrt/ref")
rospy.Subscriber(ref_topic, Odometry, ref_cb)

rospy.Timer(rospy.Duration(0.05), odom_gen)

rospy.spin()
