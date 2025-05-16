"""Test oversamplers."""

import unittest

import numpy as np
import numpy.typing as npt

from molpipeline.estimators.samplers.oversampling import GroupRandomOversampler


class TestGroupRandomOversampler(unittest.TestCase):
    """Test oversamplers."""

    def _assert_resampling_results(
        self,
        x_matrix: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        x_resampled: npt.NDArray[np.float64],
        y_resampled: npt.NDArray[np.float64],
    ) -> None:
        """Verify invariants of resampling results.

        Parameters
        ----------
        x_matrix : npt.NDArray[np.float64]
            Original features.
        y : npt.NDArray[np.float64]
            Original target values.
        x_resampled : npt.NDArray[np.float64]
            Resampled features.
        y_resampled : npt.NDArray[np.float64]
            Resampled target values.

        """
        n_samples = x_matrix.shape[0]

        # test shapes
        self.assertEqual(x_resampled.shape[1], x_matrix.shape[1])
        self.assertEqual(x_resampled.shape[0], y_resampled.shape[0])

        y_unique, y_counts = np.unique(y, return_counts=True)
        y_resampled_unique, y_resampled_counts = np.unique(
            y_resampled,
            return_counts=True,
        )

        self.assertEqual(len(y_unique), len(y_resampled_unique))
        n_minority_original, n_majority_original = sorted(y_counts)
        n_expected_new = n_majority_original - n_minority_original

        # test the resampled data has the expected number of samples
        self.assertEqual(len(y_resampled), n_samples + n_expected_new)
        # test that the resampled data has the expected number of classes
        self.assertTrue(np.array_equal(y_unique, np.unique(y_resampled)))
        # test that the resampled data has balanced classes
        self.assertEqual(y_resampled_counts[0], y_resampled_counts[1])

        # test that all oversampled samples are copies of original ones
        u_x_matrix = np.unique(x_matrix, axis=0)
        u_x_resampled = np.unique(x_resampled, axis=0)
        self.assertTrue(np.array_equal(u_x_matrix, u_x_resampled))

    def test_oversampling(self) -> None:
        """Test oversamplers."""
        sampler = GroupRandomOversampler(random_state=12345)

        y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        x_matrix = np.zeros((y.shape[0], 2))
        groups = np.array([1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])

        (  # pylint: disable=unbalanced-tuple-unpacking
            x_resampled,
            y_resampled,
        ) = sampler.transform(  # type: ignore[misc]
            x_matrix,
            y,
            groups,
        )
        self._assert_resampling_results(x_matrix, y, x_resampled, y_resampled)

    def test_invalid_x_y_size(self) -> None:
        """Test that an error is raised when group size doesn't match sample size."""
        sampler = GroupRandomOversampler(random_state=42)

        y = np.array([0, 0, 1, 0, 1])
        # X has different length than y
        x_matrix = np.zeros((y.shape[0] - 1, 2))
        groups = np.array([1, 1, 1, 2, 2])

        with self.assertRaises(ValueError) as context:
            sampler.transform(x_matrix, y, groups)

        self.assertTrue(
            "Found input variables with inconsistent numbers of samples"
            in str(context.exception),
        )

    def test_invalid_group_size(self) -> None:
        """Test that an error is raised when group size doesn't match sample size."""
        sampler = GroupRandomOversampler(random_state=42)

        y = np.array([0, 0, 1, 0, 1])
        x_matrix = np.zeros((y.shape[0], 2))
        # Different length than X and y
        groups = np.array([1, 1, 1, 2])

        with self.assertRaises(ValueError) as context:
            sampler.transform(x_matrix, y, groups)

        self.assertTrue("Found groups with size" in str(context.exception))

    def test_non_binary_classification(self) -> None:
        """Test that an error is raised for non-binary classification problems."""
        sampler = GroupRandomOversampler(random_state=42)

        # Multi-class problem with 3 classes
        y = np.array([0, 1, 2, 0, 1, 2])
        x_matrix = np.zeros((y.shape[0], 2))
        groups = np.array([1, 1, 1, 2, 2, 2])

        with self.assertRaises(ValueError):
            sampler.transform(x_matrix, y, groups)

    def test_minority_class_in_single_group(self) -> None:
        """Test when all minority class samples appear in only one group."""
        sampler = GroupRandomOversampler(random_state=42)

        n_samples = 100
        n_minority = 10

        y = np.zeros(n_samples)
        y[:n_minority] = 1  # Minority class (class 1) is first 10 samples
        x_matrix = np.zeros((n_samples, 2))

        # All minority samples are in group 1, majority samples distributed across
        # groups.
        groups = np.ones(n_samples)
        groups[n_minority : n_minority + 30] = 2
        groups[n_minority + 30 : n_minority + 60] = 3
        groups[n_minority + 60 :] = 4

        x_resampled, y_resampled = (  # pylint: disable=unbalanced-tuple-unpacking
            sampler.transform(x_matrix, y, groups)  # type: ignore[misc]
        )
        self._assert_resampling_results(x_matrix, y, x_resampled, y_resampled)

    def test_only_one_group(self) -> None:
        """Test when all samples belong to the same group."""
        sampler = GroupRandomOversampler(random_state=42)

        y = np.array([0, 1, 0, 1, 0, 0])
        n_samples = y.shape[0]
        x_matrix = np.zeros((n_samples, 2))
        groups = np.ones(n_samples)  # all samples belong to the same group

        x_resampled, y_resampled = (  # pylint: disable=unbalanced-tuple-unpacking
            sampler.transform(x_matrix, y, groups)  # type: ignore[misc]
        )
        self._assert_resampling_results(x_matrix, y, x_resampled, y_resampled)

    def test_already_balanced(self) -> None:
        """Test when the dataset is already balanced."""
        sampler = GroupRandomOversampler(random_state=42)

        y = np.array([0, 1, 0, 1])
        n_samples = y.shape[0]
        x_matrix = np.zeros((n_samples, 2))
        groups = np.array([1, 1, 2, 2])

        x_resampled, y_resampled = (  # pylint: disable=unbalanced-tuple-unpacking
            sampler.transform(x_matrix, y, groups)  # type: ignore[misc]
        )
        self._assert_resampling_results(x_matrix, y, x_resampled, y_resampled)

    def test_deterministic_behavior(self) -> None:
        """Test that the same random_state produces consistent results."""
        y = np.array([0, 0, 0, 1, 1])
        x_matrix = np.zeros((y.shape[0], 2))
        groups = np.array([1, 1, 2, 2, 3])

        # Run with same random_state twice
        sampler1 = GroupRandomOversampler(random_state=1234)
        sampler2 = GroupRandomOversampler(random_state=1234)

        x_resampled1, y_resampled1 = (  # pylint: disable=unbalanced-tuple-unpacking
            sampler1.transform(x_matrix, y, groups)  # type: ignore[misc]
        )
        x_resampled2, y_resampled2 = (  # pylint: disable=unbalanced-tuple-unpacking
            sampler2.transform(x_matrix, y, groups)  # type: ignore[misc]
        )

        # Results should be identical
        self.assertTrue(np.array_equal(x_resampled1, x_resampled2))
        self.assertTrue(np.array_equal(y_resampled1, y_resampled2))

    def test_with_empty_groups(self) -> None:
        """Test behavior when some groups have no samples of the minority class."""
        sampler = GroupRandomOversampler(random_state=42)

        magic_value = 99

        y = np.array([0, 0, 0, 1, 1])
        x_matrix = np.zeros((y.shape[0], 2))
        x_matrix[2, :] = magic_value  # Mark data of group 3 sample
        # Group 3 has no minority samples
        groups = np.array([1, 2, 3, 1, 2])

        x_resampled, y_resampled = (  # pylint: disable=unbalanced-tuple-unpacking
            sampler.transform(x_matrix, y, groups)  # type: ignore[misc]
        )
        self._assert_resampling_results(x_matrix, y, x_resampled, y_resampled)

        # check that there is only one sample with values of 99
        self.assertEqual((x_resampled == magic_value).all(axis=1).sum(), 1)

    @staticmethod
    def _construct_test_data_inverse_probability_sampling() -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Construct test data for inverse probability sampling.

        Returns
        -------
        tuple[npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64]]
                x_matrix : Feature matrix.
                y : Target values.
                groups : Group labels for the samples.
                expected_probs : Expected sampling probabilities based on inverse group
                                 sizes.

        """
        # Create a dataset with specific group sizes
        # Group 1: 10 samples (5 minority)
        # Group 2: 20 samples (5 minority)
        # Group 3: 40 samples (5 minority)
        # The smaller groups should be sampled more frequently

        # Generate data
        n_group1, n_group2, n_group3 = 10, 20, 40
        n_minority_each = 5

        # Create dataset - majority class is 0, minority class is 1
        y = np.zeros(n_group1 + n_group2 + n_group3)
        # Set minority samples in each group
        y[:n_minority_each] = 1  # Group 1 minority
        y[n_group1 : n_group1 + n_minority_each] = 1  # Group 2 minority
        y[n_group1 + n_group2 : n_group1 + n_group2 + n_minority_each] = (
            1  # Group 3 minority
        )

        x_matrix = np.zeros((len(y), 2))

        # Assign group labels
        groups = np.zeros(len(y))
        groups[:n_group1] = 0
        groups[n_group1 : n_group1 + n_group2] = 1
        groups[n_group1 + n_group2 :] = 2

        # Calculate expected sampling probabilities based on inverse group sizes
        group_sizes = np.array([n_group1, n_group2, n_group3])
        inv_weights = 1.0 / group_sizes
        expected_probs = inv_weights / inv_weights.sum()

        return x_matrix, y, groups, expected_probs

    @staticmethod
    def _generate_observed_probs_inverse_probability_sampling(
        n_runs: int,
        x_matrix: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        groups: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate expected probabilities for the inverse_probability_sampling test.

        Parameters
        ----------
        n_runs : int
            Number of runs to average the probabilities.
        x_matrix : npt.NDArray[np.float64]
            Feature matrix.
        y : npt.NDArray[np.float64]
            Target values.
        groups : npt.NDArray[np.float64]
            Group labels for the samples.

        Returns
        -------
        npt.NDArray[np.float64]
            Observed sampling probabilities for each group.

        """
        # Track the sampled groups across multiple runs
        sampled_counts = np.zeros(3)
        rng = np.random.default_rng(1234)
        for _ in range(n_runs):
            sampler = GroupRandomOversampler(
                random_state=rng.integers(0, np.iinfo(np.int32).max),
            )
            _, _, groups_resampled = (  # pylint: disable=unbalanced-tuple-unpacking
                sampler.transform(  # type: ignore[misc]
                    x_matrix,
                    y,
                    groups,
                    return_groups=True,
                )
            )

            # Count newly added samples by group
            new_samples = groups_resampled[len(groups) :]
            unique_groups, counts = np.unique(new_samples, return_counts=True)

            for group, count in zip(unique_groups, counts, strict=True):
                sampled_counts[int(group)] += count

        # Calculate and return observed sampling probabilities
        return sampled_counts / sampled_counts.sum()

    def test_inverse_probability_sampling(self) -> None:
        """Test that samples are selected based on inverse group size probabilities."""
        x_matrix, y, groups, expected_probs = (
            self._construct_test_data_inverse_probability_sampling()
        )

        n_runs = 1000  # Large number of runs for statistical significance
        observed_probs = self._generate_observed_probs_inverse_probability_sampling(
            n_runs,
            x_matrix,
            y,
            groups,
        )

        # Check that observed probabilities are close to expected probabilities
        # Allow some tolerance for statistical variation
        for i in range(3):
            self.assertAlmostEqual(
                observed_probs[i],
                expected_probs[i],
                delta=0.05,  # 5% tolerance
                msg=f"Group {i + 1} sampling probability incorrect. "
                f"Expected ~{expected_probs[i]:.2f}, got {observed_probs[i]:.2f}",
            )

        # smaller groups should have higher sampling rates
        self.assertTrue(
            observed_probs[0] > observed_probs[1] > observed_probs[2],
            "Smaller groups should have higher sampling probabilities",
        )
