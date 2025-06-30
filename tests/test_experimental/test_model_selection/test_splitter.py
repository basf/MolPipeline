"""Tests for splitters."""

import unittest
from typing import get_args

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.model_selection import GroupShuffleSplit as SklearnGroupShuffleSplit

from molpipeline.experimental.model_selection.splitter import (
    GroupShuffleSplit,
    SplitModeOption,
)

_TEST_GROUPS = (
    np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]),
    np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]),
    np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]),
    [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    ["1", "1", "1", "1", "2", "2", "2", "3", "3", "3", "3", "3"],
)


class TestGroupShuffleSplit(unittest.TestCase):
    """Unit test for the functionality of the pipeline class."""

    def test_splitting_produces_expected_sizes_when_data_allows_it(self) -> None:
        """Test the default behavior has the expected outcome.

        Inspired from sklearn test
        https://github.com/scikit-learn/scikit-learn/blob/812ff67e6725a8ca207a37f5ed4bfeafc5d1265d/sklearn/model_selection/tests/test_split.py#L789
        """
        for train_size, exp_train, exp_test in [(None, 8, 2), (7, 7, 3), (0.7, 7, 3)]:
            # Check that the default value has the expected behavior, i.e. 0.2 if both
            # unspecified or complement train_size unless both are specified.
            X = np.ones(10)  # pylint: disable=invalid-name
            y = np.ones(10)
            groups = range(10)
            for split_mode in get_args(SplitModeOption):
                split_generator = GroupShuffleSplit(
                    train_size=train_size, split_mode=split_mode
                ).split(X, y, groups)
                X_train, X_test = next(split_generator)  # pylint: disable=invalid-name
                self.assertEqual(len(X_train), exp_train)
                self.assertEqual(len(X_test), exp_test)

    def test_invalid_split_mode(self) -> None:
        """Test that an invalid split mode raises a ValueError."""
        # test invalid constructor argument
        with self.assertRaises(ValueError):
            GroupShuffleSplit(
                1,
                test_size=0.3,
                split_mode="Not a valid option.",  # type: ignore[arg-type]
                random_state=0,
            )

        # test overwrite split_mode raises
        with self.assertRaises(AssertionError):
            split_generator = GroupShuffleSplit(train_size=0.7, split_mode="samples")
            split_generator.split_mode = "Not a valid option."  # type: ignore[assignment]
            next(split_generator.split(X=[1, 2, 3], y=[1, 2, 3], groups=[1, 2, 3]))

    def test_different_input(self) -> None:
        """Test that the splitter works with different input types.

        This test inspired from sklearn. [1]

        References
        ----------
        [1] https://github.com/scikit-learn/scikit-learn/blob/812ff67e6725a8ca207a37f5ed4bfeafc5d1265d/sklearn/model_selection/tests/test_split.py#L1010

        """
        for groups_i in _TEST_GROUPS:
            X = y = np.ones(len(groups_i))  # pylint: disable=invalid-name
            n_splits = 6
            test_size = 1.0 / 3
            for split_mode in get_args(SplitModeOption):
                gss = GroupShuffleSplit(
                    n_splits, test_size=test_size, random_state=0, split_mode=split_mode
                )

                # Make sure the repr works
                repr(gss)

                # Test that the length is correct
                self.assertEqual(gss.get_n_splits(X, y, groups=groups_i), n_splits)

                l_unique = np.unique(groups_i)
                groups_i_array = np.asarray(groups_i)

                for train, test in gss.split(X, y, groups=groups_i):
                    # First test: no train group is in the test set and vice versa
                    l_train_unique = np.unique(groups_i_array[train])
                    l_test_unique = np.unique(groups_i_array[test])
                    self.assertFalse(
                        np.any(np.isin(groups_i_array[train], l_test_unique))
                    )
                    self.assertFalse(
                        np.any(np.isin(groups_i_array[test], l_train_unique))
                    )

                    # Second test: train and test add up to all the data
                    self.assertEqual(
                        groups_i_array[train].size + groups_i_array[test].size,
                        groups_i_array.size,
                    )

                    # Third test: train and test are disjoint
                    assert_array_equal(np.intersect1d(train, test), [])

                    # Fourth test:
                    # unique train and test groups are correct, +- 1 for rounding error
                    self.assertLessEqual(
                        abs(len(l_test_unique) - round(test_size * len(l_unique))), 1
                    )
                    self.assertLessEqual(
                        abs(
                            len(l_train_unique)
                            - round((1.0 - test_size) * len(l_unique))
                        ),
                        1,
                    )

    def test_compare_to_sklearn_implementation(self) -> None:
        """Test that MolPipelines implementation produces the same results as sklearn's for split_mode='groups'."""
        random_seed = 987
        for groups_i in _TEST_GROUPS:
            X = y = np.ones(len(groups_i))  # pylint: disable=invalid-name
            n_splits = 6
            test_size = 1.0 / 3
            gss_molpipeline = GroupShuffleSplit(
                n_splits,
                test_size=test_size,
                random_state=random_seed,
                split_mode="groups",
            )
            gss_sklearn = SklearnGroupShuffleSplit(
                n_splits,
                test_size=test_size,
                random_state=random_seed,
            )

            for (train_mp, test_mp), (train_sk, test_sk) in zip(
                gss_molpipeline.split(X, y, groups=groups_i),
                gss_sklearn.split(X, y, groups=groups_i),
            ):
                # test that for the same seed the exact same splits are produced
                assert_array_equal(train_mp, train_sk)
                assert_array_equal(test_mp, test_sk)
