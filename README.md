# lqRRT
Kinodynamic RRT Implementation

To install the Python package, navigate to this folder and do:  
`sudo python setup.py build && sudo python setup.py install`

To install the ROS example package, first install the Python package and then copy the folder named lqrrt_ros_demo (found in the lqRRT/demos folder) into the src folder of your catkin workspace. Finally, catkin_make.

Feel free to take the lqrrt.rviz file out of that folder and put it in your home directory's .rviz folder. To run the simulation/demo: `roslaunch lqrrt_ros_demo lqrrt_sim.launch`

Enjoy!  
-Jason Nezvadovitz

