"""Unit tests for Gaussian process kernels using the Tanimoto similarity."""

from __future__ import annotations

import abc
import unittest

import numpy as np
import numpy.typing as npt
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, Sum

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.kernel.gaussian_process_kernel import (
    ExponentialTanimotoKernel,
    TanimotoKernel,
)
from molpipeline.kernel.tanimoto_functions import tanimoto_similarity_sparse
from molpipeline.mol2any import MolToMorganFP


class GPKernelTestMixin(abc.ABC):
    """Test case for Gaussian process kernels using Tanimoto similarity."""

    kernel: Kernel
    feature_matrix_a: npt.NDArray[np.int_]
    feature_matrix_b: npt.NDArray[np.int_]
    smiles_list: list[str]
    label: list[int]

    def setUp(self) -> None:  # pylint: disable=invalid-name
        """Create small integer fingerprint matrices."""
        self.feature_matrix_a = np.array(
            [
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
            ],
            dtype=int,
        )
        self.feature_matrix_b = np.array(
            [
                [1, 0, 1, 1],
                [1, 1, 0, 0],
            ],
            dtype=int,
        )

        self.smiles_list = [
            "CCO",  # Ethanol
            "CCN",  # Ethylamine
            "CCC",  # Propane
        ]
        self.label = [
            1,
            1,
            0,  # Binary label (contains hetero atom or not) for testing
        ]

    def test_addition_with_rbf(self) -> None:
        """Test addition of TanimotoKernel with RBF kernel."""
        combined_kernel = self.kernel + RBF(length_scale=1.0)
        self.assertIsInstance(combined_kernel, Sum)  # type: ignore

    def build_classification_pipeline(self) -> Pipeline:
        """Build a simple pipeline for testing.

        Returns
        -------
        Pipeline
            A Pipeline with AutoToMol, MolToMorganFP, and GaussianProcessClassifier

        """
        return Pipeline(
            [
                ("smiles2mol", AutoToMol()),
                (
                    "fingerprint",
                    MolToMorganFP(radius=2, n_bits=1024, return_as="dense"),
                ),
                ("gp_classifier", GaussianProcessClassifier(kernel=self.kernel)),
            ],
        )

    def build_regression_pipeline(self) -> Pipeline:
        """Build a simple regression pipeline for testing.

        Returns
        -------
        Pipeline
            A Pipeline with AutoToMol, MolToMorganFP, and GaussianProcessRegressor

        """
        return Pipeline(
            [
                ("smiles2mol", AutoToMol()),
                (
                    "fingerprint",
                    MolToMorganFP(radius=2, n_bits=1024, return_as="dense"),
                ),
                ("gp_regressor", GaussianProcessRegressor(kernel=self.kernel)),
            ],
        )

    def test_classification_pipeline_fit_and_predict(self) -> None:
        """Test fitting the pipeline with sample data."""
        pipeline = self.build_classification_pipeline()
        pipeline.fit(self.smiles_list, self.label)
        predictions = pipeline.predict(self.smiles_list)
        proba = pipeline.predict_proba(self.smiles_list)
        self.assertEqual(len(predictions), len(self.smiles_list))  # type: ignore
        self.assertEqual(len(proba), len(self.smiles_list))  # type: ignore

    def test_regression_pipeline_fit_and_predict(self) -> None:
        """Test fitting the regression pipeline with sample data."""
        pipeline = self.build_regression_pipeline()
        pipeline.fit(self.smiles_list, self.label)
        predictions = pipeline.predict(self.smiles_list)
        self.assertEqual(len(predictions), len(self.smiles_list))  # type: ignore
        predictions, std = pipeline.predict(self.smiles_list, return_std=True)
        self.assertEqual(len(std), len(self.smiles_list))  # type: ignore


class TestTanimotoKernel(GPKernelTestMixin, unittest.TestCase):
    """Tests for TanimotoKernel."""

    def setUp(self) -> None:
        """Create small integer fingerprint matrices."""
        super().setUp()
        self.kernel = TanimotoKernel()

    def test_call_self_no_gradient(self) -> None:
        """Test similarity matrix when Y is None (self-similarity)."""
        kernel_matrix = self.kernel(self.feature_matrix_a)
        expected = tanimoto_similarity_sparse(
            self.feature_matrix_a,
            self.feature_matrix_a,
        )
        self.assertTrue(np.allclose(kernel_matrix, expected))

    def test_call_with_y_and_gradient(self) -> None:
        """Test call with external Y and gradient path."""
        kernel_matrix, grad = self.kernel(
            self.feature_matrix_a,
            self.feature_matrix_b,
            eval_gradient=True,
        )
        kernel_matrix_expected = tanimoto_similarity_sparse(
            self.feature_matrix_a,
            self.feature_matrix_b,
        )
        self.assertTrue(np.allclose(kernel_matrix, kernel_matrix_expected))
        self.assertEqual(grad.shape, (*kernel_matrix.shape, 0))

    def test_diag(self) -> None:
        """Test diag returns vector of ones."""
        d = self.kernel.diag(self.feature_matrix_a)
        np.testing.assert_array_equal(d, np.ones(self.feature_matrix_a.shape[0]))


class TestExponentialTanimotoKernel(GPKernelTestMixin, unittest.TestCase):
    """Tests for ExponentialTanimotoKernel."""

    def setUp(self) -> None:
        """Create fingerprints without zero pairwise similarity."""
        # Ensure no zero similarities: cumulative feature inclusion
        super().setUp()
        self.kernel = ExponentialTanimotoKernel(exponent=2.0)

    def test_repr(self) -> None:
        """Test string representation includes exponent."""
        rep = repr(self.kernel)
        self.assertIn("ExponentialTanimotoKernel", rep)
        self.assertIn("exponent=2", rep)

    def test_hyperparameter_exponent_property(self) -> None:
        """Test hyperparameter_exponent exposes correct bounds."""
        hp = self.kernel.hyperparameter_exponent
        self.assertEqual(hp.name, "exponent")
        self.assertTrue(np.allclose(hp.bounds, [[1e-3, 5]]))

    def test_call_no_gradient(self) -> None:
        """Test kernel matrix without gradient equals powered similarity."""
        kernel_matrix = self.kernel(self.feature_matrix_a)

        tanimoto = tanimoto_similarity_sparse(
            self.feature_matrix_a,
            self.feature_matrix_a,
        )
        kernel_matrix_expected = tanimoto**self.kernel.exponent

        self.assertTrue(np.allclose(kernel_matrix, kernel_matrix_expected))
        # Exponent effect: off-diagonal value decreases (since exponent>1)
        self.assertTrue(
            np.all(
                kernel_matrix[np.triu_indices_from(kernel_matrix, 1)]
                < tanimoto[np.triu_indices_from(tanimoto, 1)],
            ),
        )

    def test_call_with_gradient(self) -> None:
        """Test kernel matrix and gradient computation."""
        tanimoto = tanimoto_similarity_sparse(
            self.feature_matrix_a,
            self.feature_matrix_a,
        )
        kernel_matrix, grad = self.kernel(self.feature_matrix_a, eval_gradient=True)
        # Expected sim includes small diagonal epsilon (1e-9)
        expected_sim = (
            tanimoto**self.kernel.exponent
            + np.eye(self.feature_matrix_a.shape[0]) * 1e-9
        )
        self.assertTrue(np.allclose(kernel_matrix, expected_sim))
        # Gradient shape
        self.assertEqual(
            grad.shape,
            (self.feature_matrix_a.shape[0], self.feature_matrix_a.shape[0], 1),
        )
        base_grad = (tanimoto**self.kernel.exponent) * np.log10(tanimoto)
        expected_grad = base_grad + np.eye(self.feature_matrix_a.shape[0]) * 1e-9
        self.assertTrue(np.allclose(grad[:, :, 0], expected_grad))
        self.assertFalse(np.isnan(grad).any(), "Gradient contains NaN values.")

    def test_diag(self) -> None:
        """Test diag returns ones."""
        d = self.kernel.diag(self.feature_matrix_a)
        np.testing.assert_array_equal(d, np.ones(self.feature_matrix_a.shape[0]))
