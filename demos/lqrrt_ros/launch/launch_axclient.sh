#!/bin/bash
# Sometimes the action client starts before the server
# This should fix that
python -c "import rospy; import actionlib; import lqrrt_ros.msg; rospy.init_node('wait_for_action'); actionlib.SimpleActionClient('/move_to', lqrrt_ros.msg.MoveAction).wait_for_server()"
rosrun actionlib axclient.py /move_to
