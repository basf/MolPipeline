"""Unit tests for the DataRepetitionSplit."""

import unittest

import numpy as np

from molpipeline.experimental.model_selection.splitter.data_repetition_splitter import (
    DataRepetitionSplit,
)


class DataRepetitionSplitSplitter(unittest.TestCase):
    """Tests for DataRepetitionSplit."""

    def test_get_n_splits_returns_configured_value(self) -> None:
        """Verify get_n_splits returns the configured number of splits."""
        splitter = DataRepetitionSplit(n_splits=4)
        self.assertEqual(splitter.get_n_splits(X=np.ones(3)), 4)

    def test_split_returns_full_train_and_empty_test_for_each_split(self) -> None:
        """Verify split yields full training indices and no test indices."""
        n_samples = 6
        n_splits = 3
        x = np.arange(n_samples)
        splitter = DataRepetitionSplit(n_splits=n_splits)

        splits = list(splitter.split(X=x))
        self.assertEqual(len(splits), n_splits)

        expected_train = np.arange(n_samples)
        for train_indices, test_indices in splits:
            self.assertTrue(np.array_equal(train_indices, expected_train))
            self.assertEqual(test_indices.size, 0)
            self.assertEqual(test_indices.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
