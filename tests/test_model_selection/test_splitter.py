import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.model_selection import GroupShuffleSplit

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.model_selection.splitter import GroupShuffleSplit
from molpipeline.mol2any.mol2bool import MolToBool


class TestSampleBasedGroupShuffleSplit(unittest.TestCase):
    """Unit test for the functionality of the pipeline class."""

    def test_group_shuffle_split_default_test_size(self) -> None:
        """Test the group shuffle split using the number samples in the groups for balancing train and test set."""

        for train_size, exp_train, exp_test in [(None, 8, 2), (7, 7, 3), (0.7, 7, 3)]:

            # Check that the default value has the expected behavior, i.e. 0.2 if both
            # unspecified or complement train_size unless both are specified.
            X = np.ones(10)
            y = np.ones(10)
            groups = range(10)

            for group_by in ["size", "number"]:

                X_train, X_test = next(
                    GroupShuffleSplit(train_size=train_size, group_by=group_by).split(
                        X, y, groups
                    )
                )

                self.assertEqual(len(X_train), exp_train)
                self.assertEqual(len(X_test), exp_test)

    def test_group_shuffle_split_default_test_size(self) -> None:
        test_groups = (
            np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
            np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]),
            np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]),
            [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            ["1", "1", "1", "1", "2", "2", "2", "3", "3", "3", "3", "3"],
        )

        for groups_i in test_groups:
            X = y = np.ones(len(groups_i))
            n_splits = 6
            test_size = 1.0 / 3
            for group_by in ["size", "number"]:
                slo = GroupShuffleSplit(
                    n_splits, test_size=test_size, random_state=0, group_by=group_by
                )

                # Make sure the repr works
                repr(slo)

                # Test that the length is correct
                assert slo.get_n_splits(X, y, groups=groups_i) == n_splits

                l_unique = np.unique(groups_i)
                l = np.asarray(groups_i)

                for train, test in slo.split(X, y, groups=groups_i):
                    # First test: no train group is in the test set and vice versa
                    l_train_unique = np.unique(l[train])
                    l_test_unique = np.unique(l[test])
                    assert not np.any(np.isin(l[train], l_test_unique))
                    assert not np.any(np.isin(l[test], l_train_unique))

                    # Second test: train and test add up to all the data
                    assert l[train].size + l[test].size == l.size

                    # Third test: train and test are disjoint
                    assert_array_equal(np.intersect1d(train, test), [])

                    # Fourth test:
                    # unique train and test groups are correct, +- 1 for rounding error
                    assert (
                        abs(len(l_test_unique) - round(test_size * len(l_unique))) <= 1
                    )
                    assert (
                        abs(
                            len(l_train_unique)
                            - round((1.0 - test_size) * len(l_unique))
                        )
                        <= 1
                    )
