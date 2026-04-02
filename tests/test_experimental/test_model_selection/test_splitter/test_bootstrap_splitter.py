"""Unit tests for BootstrapSplit splitter."""

import unittest

import numpy as np

from molpipeline.experimental.model_selection.splitter.bootstrap_splitter import (
    BootstrapSplit,
)


class TestBootstrapSplit(unittest.TestCase):
    """Tests for BootstrapSplit."""

    def test_invalid_max_samples(self) -> None:
        """Verify that max_samples cannot be set with invalid value."""
        invalid_values = [-0.5, 0.0, 1.5]
        for value in invalid_values:
            with (
                self.subTest(max_samples=value),
                self.assertRaisesRegex(
                    ValueError,
                    r"If max_samples is a float, it must be in the range \(0.0, 1.0].",
                ),
            ):
                BootstrapSplit(n_splits=3, max_samples=1.5)

    def test_get_n_splits(self) -> None:
        """Verify get_n_splits returns the configured number of splits."""
        splitter = BootstrapSplit(n_splits=4, random_state=0)
        self.assertEqual(splitter.get_n_splits(X=np.ones(3)), 4)

    def test_split_properties(self) -> None:
        """Verify train/test indices satisfy bootstrap properties."""
        n_samples = 20
        n_splits = 5
        x = np.arange(n_samples)
        splitter = BootstrapSplit(n_splits=n_splits, random_state=7)

        splits = list(splitter.split(X=x))
        self.assertEqual(len(splits), n_splits)

        all_indices = np.arange(n_samples)
        for train_indices, test_indices in splits:
            # Train set is sampled with replacement and has same size as input.
            self.assertEqual(train_indices.shape[0], n_samples)

            # Test set is exactly the complement of sampled train indices.
            expected_test = np.setdiff1d(all_indices, train_indices)
            self.assertGreater(len(expected_test), 0)  # Test set is non-empty
            self.assertTrue(np.array_equal(test_indices, expected_test))
            # Test that test_indices does not contain repetitions
            self.assertEqual(test_indices.shape[0], np.unique(test_indices).shape[0])

            # Test all samples are covered by either train or test sets.
            unique_train = np.unique(train_indices)
            unique_test = np.unique(test_indices)
            self.assertEqual(np.union1d(unique_train, unique_test).shape[0], n_samples)

    def test_split_reproducible_and_expected_for_seed(self) -> None:
        """Verify split output is deterministic for a fixed random_state."""
        x = np.arange(6)
        n_splits = 5
        splitter_a = BootstrapSplit(n_splits=n_splits, random_state=123)
        splitter_b = BootstrapSplit(n_splits=n_splits, random_state=123)
        splits_a = list(splitter_a.split(X=x))
        splits_b = list(splitter_b.split(X=x))

        self.assertEqual(len(splits_a), n_splits)
        self.assertEqual(len(splits_b), n_splits)

        self.assertEqual(len(splits_a), len(splits_b))
        for split_a_i, split_b_i in zip(splits_a, splits_b, strict=True):
            train_a_i, test_a_i = split_a_i
            train_b_i, test_b_i = split_b_i
            self.assertTrue(np.array_equal(train_a_i, train_b_i))
            self.assertTrue(np.array_equal(test_a_i, test_b_i))

    def test_no_random_state(self) -> None:
        """Verify that a None is a valid random state."""
        n_splits = 5
        splitter = BootstrapSplit(n_splits=n_splits, random_state=None)
        splits = list(splitter.split(np.ones(30)))
        self.assertEqual(len(splits), n_splits)

    def test_split_with_integer_max_samples(self) -> None:
        """Verify integer max_samples controls sampled population and draw size."""
        max_samples = 7
        splitter = BootstrapSplit(n_splits=3, max_samples=max_samples, random_state=17)

        for train_indices, _test_indices in splitter.split(X=np.arange(20)):
            self.assertEqual(train_indices.shape[0], max_samples)

        sample_size = 5
        for train_indices, _test_indices in splitter.split(X=np.arange(sample_size)):
            self.assertEqual(train_indices.shape[0], sample_size)

    def test_split_with_float_max_samples(self) -> None:
        """Verify float max_samples uses the corresponding fraction of samples."""
        x = np.arange(25)
        max_samples = 0.4
        expected_n_samples = int(len(x) * max_samples)
        splitter = BootstrapSplit(n_splits=4, max_samples=max_samples, random_state=9)
        for train_indices, _test_indices in splitter.split(X=x):
            self.assertEqual(train_indices.shape[0], expected_n_samples)


if __name__ == "__main__":
    unittest.main()
