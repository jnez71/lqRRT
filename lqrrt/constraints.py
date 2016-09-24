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

	search_buffer: Additional span of the sampling space. List of tuples.

	is_feasible: Function that takes a state and effort and returns a bool.

	"""

	def __init__(self, nstates, ncontrols, goal_buffer, search_buffer, is_feasible):
		self.nstates = nstates
		self.ncontrols = ncontrols
		self.set_buffers(goal_buffer, search_buffer)
		self.set_feasibility_function(is_feasible)

#################################################

	def set_buffers(self, goal_buffer=None, search_buffer=None):
		"""
		See class docstring for argument definitions.
		Arguments not given are not modified.

		"""
		if goal_buffer is not None:
			if len(goal_buffer) == self.nstates:
				self.goal_buffer = np.array(goal_buffer, dtype=np.float64)
			else:
				raise ValueError("The goal_buffer must have same dimensionality as state.")
		
		if search_buffer is not None:
			if len(search_buffer) == self.nstates:
				self.search_buffer = np.array(search_buffer, dtype=np.float64)
				self.search_buffer_spans = np.diff(self.search_buffer).flatten()
				self.search_buffer_offsets = np.mean(self.search_buffer, axis=1)
			else:
				raise ValueError("The search_buffer must have same dimensionality as state.")

#################################################

	def set_feasibility_function(self, is_feasible):
		"""
		See class docstring for argument definitions.

		"""
		if hasattr(is_feasible, '__call__'):
			self.is_feasible = is_feasible
		else:
			raise ValueError("Expected is_feasible to be a function.")
