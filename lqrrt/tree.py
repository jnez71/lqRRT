"""
Class for an lqrrt tree.
Each node in the tree has:

Managers
- ID: integer (never specified by user)
- pID: parent ID integer

Values
- state: array of state values
- lqr: tuple of local cost-to-go and policy arrays (S, K)

Edges
- x_seq: list of state arrays from parent state to node state
- u_seq: list of effort arrays from parent state to node state

The ID corresponds to when the node was added (i.e. the third node
added has an ID of 3). The seed node always has ID 0, and pID -1.
The ID of the last node added to the tree is always mytree.size-1,
where mytree.size is the number of nodes in the tree.

To get the value of some feature of a node, do
        <tree_instance>.<feature>[<ID>]
For example, mytree.state[3] will return the state of node ID3 in
mytree. Or, mytree.x_seq[6][3] will return the fourth state vector on
the edge connecting node ID6 to its parent (starting from the parent).
IF YOU PULL OUT AN ARRAY LIKE IN THOSE EXAMPLES, IT PASSES BY REFERENCE.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np

################################################# PRIMARY CLASS

class Tree:
    """
    To initialize, provide...

    seed_state: An array of the state of the seed node.

    seed_lqr: The local LQR-optimal cost-to-go array S
              and policy array K as a tuple (S, K). That
              is, S solves the local Riccati equation and
              K = (R^-1)*(B^T)*(S) for effort jacobian B.

    """
    def __init__(self, seed_state, seed_lqr):

        # Store number of states
        self.nstates = len(seed_state)

        # Store number of controls
        try:
            self.ncontrols = seed_lqr[1].shape[0]
        except:
            print("\nThe given seed_lqr is not consistent.")
            print("Continuing, assuming you don't care about the lqr or effort features...\n")
            self.ncontrols = 1

        # Initialize state array
        self.state = np.array(seed_state, dtype=np.float64).reshape(1, self.nstates)

        # Initialize all other feature lists
        self.pID = [-1]
        self.lqr = [seed_lqr]
        self.x_seq = [[seed_state]]
        self.u_seq = [[np.zeros(self.ncontrols)]]

        # Initialize number of nodes
        self.size = 1

#################################################

    def add_node(self, pID, state, lqr, x_seq, u_seq):
        """
        Adds a node to the tree with the given features.

        """
        # Make sure the desired parent exists
        if pID >= self.size or pID < 0:
            raise ValueError("The given parent ID, {}, doesn't exist.".format(pID))

        # Update state array
        self.state = np.vstack((self.state, state))

        # Append all other feature lists
        self.pID.append(pID)
        self.lqr.append(lqr)
        self.x_seq.append(x_seq)
        self.u_seq.append(u_seq)

        # Increment node count
        self.size += 1

#################################################

    def climb(self, ID):
        """
        Returns a list of node IDs that connect the seed
        to the node with the given ID. The first element
        in the list is always 0 (seed) and the last is
        always ID (the given climb destination).

        """
        # Make sure the desired end node exists
        if ID >= self.size or ID < 0:
            raise ValueError("The given ID, {}, doesn't exist.".format(ID))

        # Follow parents backward and then reverse list
        IDs = []
        while ID != -1:
            IDs.append(ID)
            ID = self.pID[ID]
        return IDs[::-1]

#################################################

    def trajectory(self, IDs):
        """
        Given a list of node IDs, the full sequence of
        states and efforts to go from IDs[0] to IDs[-1]
        are returned as a tuple (x_seq_full, u_seq_full).

        """
        x_seq_full = []; u_seq_full = []
        for ID in IDs:
            x_seq_full.extend(self.x_seq[ID])
            u_seq_full.extend(self.u_seq[ID])
        return (x_seq_full, u_seq_full)

#################################################

    def visualize(self, dx, dy, node_seq=None):
        """
        Plots the (dx,dy)-cross-section of the current tree,
        and highlights the path given by the list, node_seq.
        For example, dx=0, dy=1 plots the states #0 and #1.

        """
        print("\n...now plotting...")
        from matplotlib import pyplot as plt

        fig = plt.figure()
        fig.suptitle('Tree')
        plt.axis('equal')

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('- State {} +'.format(dx))
        ax.set_ylabel('- State {} +'.format(dy))
        ax.grid(True)

        if node_seq is None:
            node_seq = []

        if self.size > 1:
            for ID in xrange(self.size):
                x_seq = np.array(self.x_seq[ID])
                if ID in node_seq:
                    ax.plot((x_seq[:, dx]), (x_seq[:, dy]), color='r', zorder=2)
                else:
                    ax.plot((x_seq[:, dx]), (x_seq[:, dy]), color='0.75', zorder=1)

        ax.scatter(self.state[0, dx], self.state[0, dy], color='b', s=48)
        if len(node_seq):
            ax.scatter(self.state[node_seq[-1], dx], self.state[node_seq[-1], dy], color='r', s=48)

        print("Done! Close window to continue.\n")
        plt.show()
