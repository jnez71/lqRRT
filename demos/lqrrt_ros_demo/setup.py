## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['lqrrt_ros_demo', 'behaviors']
    #package_dir={'': 'src'},
    #requires=[],
)

setup(**setup_args)
