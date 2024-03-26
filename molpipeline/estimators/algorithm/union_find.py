"""Union find algorithm."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class UnionFindNode:
    """Union find node.

    A UnionFindNode is a node in a union find data structure, also called disjoint-set data structure.
    It stores a collection of non-overlapping sets and provides operations for merging these sets.
    It can be used to determine connected components in a graph.
    """

    def __init__(self) -> None:
        """Initialize union find node."""
        # initially, each node is its own parent
        self.parent = self
        # connected component number for each node. Needed at the end when connected components are getting extracted.
        self.connected_component_number = -1

    def find(self) -> UnionFindNode:
        """Find the root node.

        Returns
        -------
        Self
            Root node.
        """
        if self.parent is not self:
            self.parent = self.parent.find()
        return self.parent

    def union(self, other: UnionFindNode) -> UnionFindNode:
        """Union two nodes.

        Parameters
        ----------
        other: Self
            Other node.

        Returns
        -------
        UnionFindNode
            Root node of set or self.
        """
        # get the root nodes of the connected components of both nodes.
        # Additionally, compress the paths to the root nodes by setting others' parent to the root node.
        if self is self.parent:
            # this node is a parent. Meaning the root of the connected component. Let's overwrite it.
            self.parent = other.find()
        elif self is other:
            # this node is the other node. They are identical and therefore already in the same set.
            return self
        else:
            # add other node to this node's parent (and in addition to this connected component)
            self.parent = self.parent.union(other)
        return self.parent

    @staticmethod
    def get_connected_components(
        union_find_nodes: list[UnionFindNode],
    ) -> tuple[int, npt.NDArray[np.int32]]:
        """Get connected components from a union find node list.

        Parameters
        ----------
        union_find_nodes: list[UnionFindNode]
            List of union find nodes.

        Returns
        -------
        tuple[int, np.ndarray[int]]
            Number of connected components and connected component labels.
        """
        # results
        connected_components_counter = 0
        connected_components_array = np.empty(len(union_find_nodes), dtype=np.int32)

        for i, node in enumerate(union_find_nodes):
            root_parent = node.find()
            if root_parent.connected_component_number == -1:
                # found root node of a connected component. Annotate it with a connected component number.
                root_parent.connected_component_number = connected_components_counter
                connected_components_counter += 1
            connected_components_array[i] = root_parent.connected_component_number
        return connected_components_counter, connected_components_array
