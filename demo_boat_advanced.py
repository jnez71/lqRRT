"""
Demo of using the lqrrt planner for boats.
This is an extension of the intermediate demo criteria but
rather than exploring once all the way to the goal, we show
how you can chain short-distance dense-explorations in realtime.

State:   [x, y, h, vx, vy, vh]  (m, m, rad, m/s, m/s, rad/s)
Effort:  [ux, uy, uh]           (N, N, N*m)

Pose states are world-frame.
Twist states and wrench are body-frame.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl
import lqrrt

################################################# PHYSICAL PARAMETERS
