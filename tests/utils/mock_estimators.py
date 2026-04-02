"""Shared mock estimators used by ensemble estimator unit tests.

A normal mock cannot be used because sklearn.clone does not preserve the mocked methods.

"""

from typing import Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from typing_extensions import override


class MockEstimator(BaseEstimator):
    """A mock estimator that records fit arguments and returns fixed predictions."""

    fit_args: dict[str, npt.ArrayLike]

    def __init__(self, alpha: int = 0, beta: int = 0, gamma: int = 0) -> None:
        """Initialize the MockEstimator with dummy parameters.

        Parameters
        ----------
        alpha : int, default=0
            Dummy parameter alpha.
        beta : int, default=0
            Dummy parameter beta.
        gamma : int, default=0
            Dummy parameter gamma.

        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fit_args = {}

    def fit(
        self,
        X: npt.ArrayLike,  # noqa: N803 # pylint: disable=invalid-name
        y: npt.ArrayLike,
    ) -> Self:
        """Fit the model and record the arguments.

        Parameters
        ----------
        X : npt.ArrayLike
            Training data.
        y : npt.ArrayLike
            Target values.

        Returns
        -------
        Self
            The fitted estimator.

        """
        self.fit_args = {"X": X, "y": y}
        return self

    def predict(  # noqa: PLR6301
        self,
        X: npt.ArrayLike,  # noqa: N803 # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Return fixed predictions.

        Parameters
        ----------
        X : npt.ArrayLike
            Input data for prediction.

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted values, which are all zeros for this mock estimator.

        """
        feature_arr = np.asarray(X)
        return np.zeros(len(feature_arr), dtype=np.float64)


class MockClassifier(MockEstimator):
    """A mock classifier that records fit arguments and returns fixed predictions."""

    @override
    def predict(  # type: ignore
        self,
        X: npt.ArrayLike,  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.int64]:
        """Predicts whether the first feature is equal to 1 for each sample.

        Parameters
        ----------
        X : npt.ArrayLike
            Input data for prediction.

        Returns
        -------
        npt.NDArray[np.int64]
            Predicted class labels.

        """
        feature_arr = np.asarray(X)
        return np.array([x[0] == 1 for x in feature_arr], dtype=np.int64)

    def predict_proba(  # noqa: PLR6301
        self,
        X: npt.ArrayLike,  # noqa: N803 # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Return fake class probabilities.

        Parameters
        ----------
        X : npt.ArrayLike
            Input data for probability prediction.

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted class probabilities, where the probability of class 1 is 0.7 if
            the first feature is 1, and 0.3 otherwise.

        """
        feature_arr = np.asarray(X)
        proba = np.zeros((len(feature_arr), 2))
        equal_one = feature_arr[:, 0] == 1
        proba[equal_one, 0] = 0.3
        proba[~equal_one, 0] = 0.7
        proba[equal_one, 1] = 0.7
        proba[~equal_one, 1] = 0.3
        return proba


class MockClassiferWithFloatLabels(MockClassifier):
    """A mock classifier that returns float class labels instead of integers."""

    @override
    def predict(  # type: ignore
        self,
        X: npt.ArrayLike,  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Return fixed class predictions as floats.

        Parameters
        ----------
        X : npt.ArrayLike
            Input data for prediction.

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted class labels as floats.

        """
        return super().predict(X).astype(np.float64)


class MockClassifierWithTrueFloatLabels(MockClassifier):
    """A mock classifier that returns true float class labels instead of integers.

    True float values denote values which have float type and cannot be transformed to
    integers, such as 1.1 or 0.5. These values are invalid class labels and can raise an
    error in a meta-estimator. This mock class can be used to check if such faulty
    predictions were handled correctly.

    """

    @override
    def predict(  # type: ignore
        self,
        X: npt.ArrayLike,  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Return fixed class predictions plus 0.5 so they are not integers.

        Parameters
        ----------
        X : npt.ArrayLike
            Input data for prediction.

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted class labels as floats that cannot be interpreted as integers.

        """
        return super().predict(X).astype(np.float64) + 0.5
