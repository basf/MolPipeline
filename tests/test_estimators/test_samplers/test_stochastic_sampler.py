"""Tests for StochasticSampler."""

import unittest
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt

from molpipeline.estimators.samplers.stochastic_filter import StochasticFilter
from molpipeline.estimators.samplers.stochastic_sampler import StochasticSampler


class _NoFilter(StochasticFilter):  # pylint: disable=too-few-public-methods
    """Mock filter that returns predefined probabilities."""

    def __init__(self, probabilities: npt.NDArray[np.float64]) -> None:
        """Create a new _NoFilter.

        Parameters
        ----------
        probabilities : npt.NDArray[np.float64]
            Predefined probabilities for the samples.

        """
        self.probabilities = probabilities

    def calculate_probabilities(
        self,
        _X: npt.NDArray[np.float64],  # noqa: N803 # pylint: disable=invalid-name
        _y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate probabilities for the samples.

        Parameters
        ----------
        _X : npt.NDArray[np.float64]
            Training data. Will be ignored, only present because of compatibility.
        _y : npt.NDArray[np.float64]
            Target values. Will be ignored, only present because of compatibility.

        Returns
        -------
        npt.NDArray[np.float64]
            Predefined probabilities for the samples.

        """
        return self.probabilities


class TestStochasticSampler(unittest.TestCase):
    """Test the StochasticSampler class."""

    def setUp(self) -> None:
        """Set up test data."""
        self.x_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])
        self.simple_filter = _NoFilter(np.array([0.1, 0.2, 0.3, 0.4]))
        self.uniform_filter = _NoFilter(np.array([0.25, 0.25, 0.25, 0.25]))

    def test_init_with_valid_parameters(self) -> None:
        """Test initialization with valid parameters."""
        sampler = StochasticSampler(
            filters=[self.simple_filter],
            n_samples=2,
            random_state=42,
        )
        self.assertEqual(sampler.n_samples, 2)
        self.assertEqual(sampler.combination_method, "product")
        self.assertEqual(len(sampler.filters), 1)
        self.assertIsInstance(
            sampler.random_state,
            np.random.RandomState,  # pylint: disable=no-member
        )

    def test_init_with_invalid_n_samples(self) -> None:
        """Test initialization with invalid n_samples."""
        with self.assertRaises(ValueError):
            StochasticSampler(filters=[self.simple_filter], n_samples=0)

        with self.assertRaises(ValueError):
            StochasticSampler(filters=[self.simple_filter], n_samples=-1)

    def test_init_with_invalid_combination_method(self) -> None:
        """Test initialization with invalid combination method."""
        with self.assertRaises(ValueError):
            StochasticSampler(
                filters=[self.simple_filter],
                n_samples=2,
                combination_method="invalid",  # type: ignore[arg-type]
            )

    def test_fit_returns_self(self) -> None:
        """Test that fit returns self."""
        sampler = StochasticSampler(filters=[self.simple_filter], n_samples=2)
        result = sampler.fit(self.x_matrix, self.y)
        self.assertIs(result, sampler)

    def test_calculate_probabilities_single_filter(self) -> None:
        """Test calculate_probabilities with a single filter."""
        sampler = StochasticSampler(
            filters=[self.simple_filter],
            n_samples=2,
            random_state=42,
        )

        probs = sampler.calculate_probabilities(self.x_matrix, self.y)

        # Should return normalized probabilities from the filter
        expected = np.array([0.1, 0.2, 0.3, 0.4]) / np.array([0.1, 0.2, 0.3, 0.4]).sum()
        self.assertTrue(np.allclose(probs, expected))
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_calculate_probabilities_multiple_filters_product(self) -> None:
        """Test calculate_probabilities w/ multiple filters and product combination."""
        filter1 = _NoFilter(np.array([0.1, 0.2, 0.3, 0.4]))
        filter2 = _NoFilter(np.array([0.4, 0.3, 0.2, 0.1]))

        sampler = StochasticSampler(
            filters=[filter1, filter2],
            n_samples=2,
            combination_method="product",
            random_state=42,
        )

        probs = sampler.calculate_probabilities(self.x_matrix, self.y)

        # Expected: product of probabilities, normalized
        expected_unnormalized = np.array([0.1 * 0.4, 0.2 * 0.3, 0.3 * 0.2, 0.4 * 0.1])
        expected = expected_unnormalized / expected_unnormalized.sum()
        self.assertTrue(np.allclose(probs, expected))
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_calculate_probabilities_multiple_filters_mean(self) -> None:
        """Test calculate_probabilities with multiple filters using mean combination."""
        filter1 = _NoFilter(np.array([0.1, 0.2, 0.3, 0.4]))
        filter2 = _NoFilter(np.array([0.4, 0.3, 0.2, 0.1]))

        sampler = StochasticSampler(
            filters=[filter1, filter2],
            n_samples=2,
            combination_method="mean",
            random_state=42,
        )

        probs = sampler.calculate_probabilities(self.x_matrix, self.y)

        # Expected: mean of probabilities, normalized
        expected_unnormalized = np.array(
            [(0.1 + 0.4) / 2, (0.2 + 0.3) / 2, (0.3 + 0.2) / 2, (0.4 + 0.1) / 2],
        )
        expected = expected_unnormalized / expected_unnormalized.sum()
        self.assertTrue(np.allclose(probs, expected))
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_calculate_probabilities_with_all_zero_probabilities(self) -> None:
        """Test calculate_probabilities with all zero probabilities."""
        zero_filter = _NoFilter(np.array([0.0, 0.0, 0.0, 0.0]))

        sampler = StochasticSampler(filters=[zero_filter], n_samples=10)

        probs = sampler.calculate_probabilities(self.x_matrix, self.y)

        # Should return uniform distribution when all probabilities are zero
        expected = np.ones(4) / 4
        self.assertTrue(np.allclose(probs, expected))
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_calculate_probabilities_with_zero_probabilities_multiple_filters(
        self,
    ) -> None:
        """Test numerical stability in combine_probabilities with zero values."""
        filter1 = _NoFilter(np.array([0.0, 0.3, 0.5, 0.0]))
        filter2 = _NoFilter(np.array([0.2, 0.0, 0.6, 0.0]))

        sampler = StochasticSampler(
            filters=[filter1, filter2],
            n_samples=2,
            combination_method="product",
        )

        probs = sampler.calculate_probabilities(self.x_matrix, self.y)

        expected_unnormalized = np.array([0.0, 0.0, 0.5 * 0.6, 0.0])
        expected = expected_unnormalized / expected_unnormalized.sum()
        self.assertTrue(np.allclose(probs, expected))
        self.assertAlmostEqual(probs.sum(), 1.0)

        # filters with all zero probabilities in product combination
        filter1 = _NoFilter(np.array([0.0, 0.3, 0.0, 0.0]))
        filter2 = _NoFilter(np.array([0.2, 0.0, 0.6, 0.0]))

        sampler = StochasticSampler(
            filters=[filter1, filter2],
            n_samples=2,
            combination_method="product",
        )

        probs = sampler.calculate_probabilities(self.x_matrix, self.y)

        # Should return uniform distribution when all probabilities are zero
        expected = np.ones(4) / 4
        self.assertTrue(np.allclose(probs, expected))
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_transform_small_probabilities_product(self) -> None:
        """Test numerical stability with very small probabilities."""
        # Create a filter with very small probabilities
        tiny_probs = np.array([1e-10, 1e-15, 1e-20, 1e-5])
        filter1 = _NoFilter(tiny_probs)
        filter2 = _NoFilter(tiny_probs)

        sampler = StochasticSampler(
            filters=[filter1, filter2],
            n_samples=10,
            random_state=42,
            combination_method="product",
        )

        # This should not raise numerical errors
        probs = sampler.calculate_probabilities(self.x_matrix, self.y)

        # Expected: product of probabilities, normalized
        expected_unnormalized = tiny_probs * tiny_probs
        expected = expected_unnormalized / expected_unnormalized.sum()
        self.assertTrue(np.allclose(probs, expected))
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_transform_small_probabilities_mean(self) -> None:
        """Test numerical stability with very small probabilities."""
        # Create a filter with very small probabilities
        tiny_probs = np.array([1e-10, 1e-15, 1e-20, 1e-5])
        filter1 = _NoFilter(tiny_probs)
        filter2 = _NoFilter(tiny_probs)

        sampler = StochasticSampler(
            filters=[filter1, filter2],
            n_samples=10,
            random_state=42,
            combination_method="mean",
        )

        # This should not raise numerical errors
        probs = sampler.calculate_probabilities(self.x_matrix, self.y)

        # Expected: product of probabilities, normalized
        expected_unnormalized = (tiny_probs + tiny_probs) / 2
        expected = expected_unnormalized / expected_unnormalized.sum()
        self.assertTrue(np.allclose(probs, expected))
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_calculate_probabilities_calls_filters(self) -> None:
        """Test that each filter's calculate_probabilities method is called."""
        mock_filter1 = mock.MagicMock(spec=StochasticFilter)
        mock_filter1.calculate_probabilities.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4],
        )

        mock_filter2 = mock.MagicMock(spec=StochasticFilter)
        mock_filter2.calculate_probabilities.return_value = np.array(
            [0.4, 0.3, 0.2, 0.1],
        )

        sampler = StochasticSampler(filters=[mock_filter1, mock_filter2], n_samples=2)

        sampler.calculate_probabilities(self.x_matrix, self.y)

        mock_filter1.calculate_probabilities.assert_called_once()
        mock_filter2.calculate_probabilities.assert_called_once()

        # Check that args passed to filter were correct
        args1, _ = mock_filter1.calculate_probabilities.call_args
        args2, _ = mock_filter2.calculate_probabilities.call_args
        self.assertTrue(np.array_equal(args1[0], self.x_matrix))
        self.assertTrue(np.array_equal(args1[1], self.y))
        self.assertTrue(np.array_equal(args2[0], self.x_matrix))
        self.assertTrue(np.array_equal(args2[1], self.y))

    @staticmethod
    def _generate_observed_probs(
        n_runs: int,
        n_samples: int,
        sampler_kwargs: dict[str, Any],
    ) -> npt.NDArray[np.float64]:
        """Generate expected probabilities for the inverse_probability_sampling test.

        Parameters
        ----------
        n_runs : int
            Number of runs to average the probabilities.
        n_samples : int
            The number of samples to generate.
        sampler_kwargs : dict[str, Any]
            Key word arguments for the sampler.

        Returns
        -------
        npt.NDArray[np.float64]
            Observed sampling probabilities for each group.

        """
        # x_matrix data will contain 50 samples with identifiers as values
        x_matrix: npt.NDArray[np.float64] = np.arange(
            n_samples,
            dtype=np.float64,
        ).reshape(-1, 1)
        y = np.ones(len(x_matrix), dtype=np.float64)

        sampled_counts = np.zeros_like(y)
        rng = np.random.default_rng(1234)
        for _ in range(n_runs):
            sampler = StochasticSampler(
                random_state=rng.integers(0, np.iinfo(np.int32).max),
                **sampler_kwargs,
            )
            x_sampled, _ = sampler.transform(
                x_matrix,
                y,
            )

            # Count newly added samples by group
            unique_samples, counts = np.unique(x_sampled, return_counts=True)

            for sample_id, count in zip(unique_samples, counts, strict=True):
                sampled_counts[int(sample_id)] += count

        # Calculate and return observed sampling probabilities
        return sampled_counts / sampled_counts.sum()

    def test_transform_with_multiple_filters_product(self) -> None:
        """Test transform with multiple filters using product combination."""
        filter1 = _NoFilter(np.array([0.1, 0.2, 0.3, 0.4]))
        filter2 = _NoFilter(np.array([0.4, 0.3, 0.2, 0.1]))

        sampler = StochasticSampler(
            filters=[filter1, filter2],
            n_samples=10,
            combination_method="product",
            random_state=42,
        )
        x_sampled, y_sampled = sampler.transform(self.x_matrix, self.y)

        self.assertEqual(x_sampled.shape, (10, 2))
        self.assertEqual(y_sampled.shape, (10,))

    def test_transform_with_multiple_filters_product_observed_freqs(self) -> None:
        """Test observed sample frequencies match expected frequencies."""
        filter1 = _NoFilter(np.array([0.5, 0.25, 0.25, 0.25]))
        filter2 = _NoFilter(np.array([0.5, 0.25, 0.25, 0.25]))

        sampler_kwargs = {
            "filters": [filter1, filter2],
            "n_samples": 100,
            "combination_method": "product",
        }

        observed_probs = self._generate_observed_probs(
            n_runs=1000,
            n_samples=4,
            sampler_kwargs=sampler_kwargs,
        )

        # the first element of the expected probs is biased by 2
        expected_probs = np.array([0.5 * 0.5, *([0.25 * 0.25] * 3)])
        expected_probs /= expected_probs.sum()
        self.assertTrue(np.allclose(expected_probs, observed_probs, rtol=0, atol=1e-2))

    def test_transform_with_multiple_filters_mean(self) -> None:
        """Test transform with multiple filters using mean combination."""
        filter1 = _NoFilter(np.array([0.1, 0.2, 0.3, 0.4]))
        filter2 = _NoFilter(np.array([0.4, 0.3, 0.2, 0.1]))

        sampler = StochasticSampler(
            filters=[filter1, filter2],
            n_samples=10,
            combination_method="mean",
            random_state=42,
        )

        x_sampled, y_sampled = sampler.transform(self.x_matrix, self.y)

        self.assertEqual(x_sampled.shape, (10, 2))
        self.assertEqual(y_sampled.shape, (10,))

    def test_transform_with_multiple_filters_mean_observed_freqs(self) -> None:
        """Test observed sample frequencies match expected frequencies."""
        filter1 = _NoFilter(np.array([0.5, 0.25, 0.25, 0.25]))
        filter2 = _NoFilter(np.array([0.5, 0.25, 0.25, 0.25]))

        sampler_kwargs = {
            "filters": [filter1, filter2],
            "n_samples": 10,
            "combination_method": "mean",
        }

        observed_probs = self._generate_observed_probs(
            n_runs=1000,
            n_samples=4,
            sampler_kwargs=sampler_kwargs,
        )

        # the first element of the expected probs is biased by 2
        expected_probs = np.array([(0.5 + 0.5) / 2, *([(0.25 + 0.25) / 2] * 3)])
        expected_probs /= expected_probs.sum()
        self.assertTrue(np.allclose(expected_probs, observed_probs, rtol=0, atol=1e-2))

    def test_transform_with_zero_probabilities(self) -> None:
        """Test transform with all zero probabilities uses uniform distribution."""
        zero_filter = _NoFilter(np.array([0.0, 0.0, 0.0, 0.0]))

        sampler = StochasticSampler(
            filters=[zero_filter],
            n_samples=100,
            random_state=42,
        )

        # This should now work without raising an error
        x_sampled, y_sampled = sampler.transform(self.x_matrix, self.y)

        self.assertEqual(x_sampled.shape, (100, 2))
        self.assertEqual(y_sampled.shape, (100,))

    def test_transform_with_zero_probabilities_product_observed_freqs(self) -> None:
        """Test observed sample frequencies match expected frequencies."""
        filter1 = _NoFilter(np.array([0.0, 0.0, 0.0, 0.0]))
        filter2 = _NoFilter(np.array([0.0, 0.0, 0.0, 0.0]))

        sampler_kwargs = {
            "filters": [filter1, filter2],
            "n_samples": 10,
            "combination_method": "product",
        }

        observed_probs = self._generate_observed_probs(
            n_runs=1000,
            n_samples=4,
            sampler_kwargs=sampler_kwargs,
        )

        # the first element of the expected probs is biased by 2
        expected_probs = np.ones(4) / 4
        self.assertTrue(np.allclose(expected_probs, observed_probs, rtol=0, atol=1e-2))

    def test_reproducibility_with_fixed_seed(self) -> None:
        """Test that results are reproducible with fixed random state."""
        sampler1 = StochasticSampler(
            filters=[self.simple_filter],
            n_samples=5,
            random_state=42,
        )
        sampler2 = StochasticSampler(
            filters=[self.simple_filter],
            n_samples=5,
            random_state=42,
        )

        x_sampled1, y_sampled1 = sampler1.transform(self.x_matrix, self.y)
        x_sampled2, y_sampled2 = sampler2.transform(self.x_matrix, self.y)

        self.assertTrue(np.array_equal(x_sampled1, x_sampled2))
        self.assertTrue(np.array_equal(y_sampled1, y_sampled2))
