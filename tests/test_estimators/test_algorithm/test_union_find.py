"""Test union find algorithm."""

import unittest

import numpy as np

from molpipeline.estimators.algorithm.union_find import UnionFindNode


class TestUnionFind(unittest.TestCase):
    """Test the UnionFindNode class."""

    def test_union_find(self) -> None:
        """Test the union find algorithm."""
        uf_nodes = [UnionFindNode() for _ in range(10)]
        nof_cc, cc_labels = UnionFindNode.get_connected_components(uf_nodes)

        self.assertTrue(np.equal(cc_labels, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).all())
        self.assertEqual(nof_cc, 10)

        uf_nodes = [UnionFindNode() for _ in range(10)]
        # one cc
        uf_nodes[0].union(uf_nodes[1])
        uf_nodes[0].union(uf_nodes[7])
        # second cc
        uf_nodes[2].union(uf_nodes[3])
        uf_nodes[2].union(uf_nodes[4])
        uf_nodes[3].union(uf_nodes[9])

        cc_num, cc_labels = UnionFindNode.get_connected_components(uf_nodes)
        self.assertTrue(
            np.equal(
                cc_labels,
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    3,
                    0,
                    4,
                    1,
                ],
            ).all()
        )
        self.assertEqual(cc_num, 5)
