# lqRRT
Kinodynamic RRT Implementation

To install the Python package, navigate to this folder and do:  
`sudo python setup.py build && sudo python setup.py install`

To install the ROS example package, first install the Python package and then copy the folder named lqrrt_ros_demo (found in the lqRRT/demos folder) into the src folder of your catkin workspace. Finally, catkin_make.

Feel free to take the lqrrt.rviz file out of that folder and put it in your home directory's .rviz folder. When running the demo, just rosrun the three nodes in the nodes folder, and then for interfacing with the action server do:  
`rosrun actionlib axclient.py /move_to`

Enjoy!  
-Jason Nezvadovitz

