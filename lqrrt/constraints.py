"""
Class for lqrrt constraints.

An instance of this class must be given to an lqrrt planner
to fully define the search problem.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl

################################################# PRIMARY CLASS

class Constraints:
    """
    To initialize, provide...

    nstates: The dimensionality of the state space.

    ncontrols: The dimensionality of the effort space.

    goal_buffer: Half-edge lengths of box defining goal region.

    is_feasible: Function that takes a state and effort and returns a bool.

    """

    def __init__(self, nstates, ncontrols, goal_buffer, is_feasible):
        self.nstates = nstates
        self.ncontrols = ncontrols
        self.set_buffers(goal_buffer)
        self.set_feasibility_function(is_feasible)

#################################################

    def set_buffers(self, goal_buffer=None):
        """
        See class docstring for argument definitions.
        Arguments not given are not modified.

        """
        if goal_buffer is not None:
            if len(goal_buffer) == self.nstates:
                self.goal_buffer = np.abs(goal_buffer).astype(np.float64)
            else:
                raise ValueError("The goal_buffer must have same dimensionality as state.")

#################################################

    def set_feasibility_function(self, is_feasible):
        """
        See class docstring for argument definitions.

        """
        if hasattr(is_feasible, '__call__'):
            self.is_feasible = is_feasible
        else:
            raise ValueError("Expected is_feasible to be a function.")
