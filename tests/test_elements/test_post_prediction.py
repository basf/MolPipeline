"""Test the module post_prediction.py."""

import unittest

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from molpipeline.post_prediction import PostPredictionWrapper


class TestPostPredictionWrapper(unittest.TestCase):
    """Test the PostPredictionWrapper class."""

    def test_get_params(self) -> None:
        """Test get_params method."""
        rf = RandomForestClassifier()
        rf_params = rf.get_params(deep=True)

        ppw = PostPredictionWrapper(rf)
        ppw_params = ppw.get_params(deep=True)

        wrapped_params = {}
        for key, value in ppw_params.items():
            first, _, rest = key.partition("__")
            if first == "wrapped_estimator":
                if rest == "":
                    self.assertIs(rf, value)
                else:
                    wrapped_params[rest] = value

        self.assertDictEqual(rf_params, wrapped_params)

    def test_set_params(self) -> None:
        """Test set_params method."""
        rf = RandomForestClassifier()
        ppw = PostPredictionWrapper(rf)

        ppw.set_params(wrapped_estimator__n_estimators=10)
        self.assertIsInstance(ppw.wrapped_estimator, RandomForestClassifier)
        if not isinstance(ppw.wrapped_estimator, RandomForestClassifier):
            raise TypeError("Wrapped estimator is not a RandomForestClassifier.")
        self.assertEqual(ppw.wrapped_estimator.n_estimators, 10)

        ppw_params = ppw.get_params(deep=True)
        self.assertEqual(ppw_params["wrapped_estimator__n_estimators"], 10)

    def test_fit_transform(self) -> None:
        """Test fit method."""
        rng = np.random.default_rng(20240918)
        features = rng.random((10, 5))

        pca = PCA(n_components=3)
        pca.fit(features)
        pca_transformed = pca.transform(features)

        ppw = PostPredictionWrapper(clone(pca))
        ppw.fit(features)
        ppw_transformed = ppw.transform(features)

        self.assertEqual(pca_transformed.shape, ppw_transformed.shape)
        self.assertTrue(np.allclose(pca_transformed, ppw_transformed))

    def test_inverse_transform(self) -> None:
        """Test inverse_transform method."""
        rng = np.random.default_rng(20240918)
        features = rng.random((10, 5))

        pca = PCA(n_components=3)
        pca.fit(features)
        pca_transformed = pca.transform(features)
        pca_inverse = pca.inverse_transform(pca_transformed)

        ppw = PostPredictionWrapper(clone(pca))
        ppw.fit(features)
        ppw_transformed = ppw.transform(features)
        ppw_inverse = ppw.inverse_transform(ppw_transformed)

        self.assertEqual(features.shape, ppw_inverse.shape)
        self.assertEqual(pca_inverse.shape, ppw_inverse.shape)

        self.assertTrue(np.allclose(pca_inverse, ppw_inverse))
