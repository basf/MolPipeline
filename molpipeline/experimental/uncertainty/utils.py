"""Conformal prediction utils."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from crepes.extras import hinge, margin


class NonconformityFunctor(ABC):
    """Abstract base class for nonconformity functions.

    Simple wrapper for nonconformity functions with sklearn compatibility.
    Supports pickle serialization and parameter management.
    """

    @abstractmethod
    def __call__(
        self,
        y_prob: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute nonconformity scores.

        Parameters
        ----------
        y_prob : npt.NDArray[np.float64]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y_true : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.

        """

    @abstractmethod
    def get_name(self) -> str:
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """


class HingeNonconformity(NonconformityFunctor):
    """Hinge loss nonconformity function for classification."""

    def __call__(
        self,
        y_prob: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute hinge nonconformity scores.

        Parameters
        ----------
        y_prob : npt.NDArray[np.float64]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y_true : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.

        """
        return hinge(y_prob, classes, y_true)

    def get_name(self) -> str:  # noqa: PLR6301
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return "hinge"


class MarginNonconformity(NonconformityFunctor):
    """Margin nonconformity function for classification."""

    def __call__(
        self,
        y_prob: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute margin nonconformity scores.

        Parameters
        ----------
        y_prob : npt.NDArray[np.float64]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y_true : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.

        """
        return margin(y_prob, classes, y_true)

    def get_name(self) -> str:  # noqa: PLR6301
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return "margin"


class LogNonconformity(NonconformityFunctor):
    """Logarithmic nonconformity function for classification."""

    @staticmethod
    def log_nc(
        y_prob: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | pd.Series | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute log non-conformity scores for conformal classifiers.

        Parameters
        ----------
        y_prob : array-like of shape (n_samples, n_classes)
            predicted class probabilities
        classes : array-like of shape (n_classes,), default=None
            class names
        y_true : array-like of shape (n_samples,), default=None
            true target values

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            non-conformity scores. The shape is (n_samples, n_classes)
            if classes and y are None.

        """
        if y_true is not None and classes is not None:
            if isinstance(y_true, pd.Series):
                y_true = y_true.to_numpy()
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            class_indexes = np.fromiter(
                (class_to_idx[label] for label in y_true),
                dtype=np.intp,
                count=len(y_true),
            )
            result = -np.log(
                np.maximum(y_prob[np.arange(len(y_true)), class_indexes], 1e-10)
            )
        else:
            result = -np.log(np.maximum(y_prob, 1e-10))
        return result

    def __call__(
        self,
        y_prob: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute log nonconformity scores.

        Parameters
        ----------
        y_prob : npt.NDArray[np.float64]
            Probability predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y_true : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.

        """
        return self.log_nc(y_prob, classes, y_true)

    def get_name(self) -> str:  # noqa: PLR6301
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.
        """
        return "log"


class SVMMarginNonconformity(NonconformityFunctor):
    """SVM margin-based conformity measure for classification.

    Based on https://doi.org/10.1016/j.jbi.2019.103350
    Uses the distance to the margin boundary of the predicted class.

    The conformity score is computed as the distance between the instance
    and the margin boundary of the class under consideration. For a test
    instance, the signed distance to the separating hyperplane is d_h = y/||w||,
    and the distance to the margin boundary is d_mb = |d_h - 1/||w||.

    """

    def __call__(
        self,
        y_score: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute SVM margin-based conformity scores.

        Note: This function expects decision function values (distances to hyperplane)
        rather than probabilities. For sklearn SVC, use decision_function() output.

        Parameters
        ----------
        y_score : npt.NDArray[np.float64]
            Decision function values (signed distances to hyperplane) from SVM.
            For binary classification, shape should be (n_samples,).
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y_true : npt.NDArray[Any] | None, optional
            True labels for calibration. If provided, returns conformity scores
            for the true class. If None, returns scores for all classes.

        Raises
        ------
        ValueError
            If classes is None for multiclass nonconformity computation.

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.
            - If y_true is provided: (n_samples,) with conformity for true class
            - If y_true is None: (n_samples, n_classes) with conformity for each class
        """
        # For binary classification with decision_function output
        if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
            decision_values = y_score.flatten()

            if y_true is not None:
                if isinstance(y_true, pd.Series):
                    y_true = y_true.to_numpy()
                if classes is not None:
                    y_mapped = np.where(y_true == classes[1], 1, -1)
                else:
                    y_mapped = np.where(y_true == 1, 1, -1)
                pos_mask = y_mapped == 1
                conformity = np.empty_like(decision_values, dtype=np.float64)
                conformity[pos_mask] = decision_values[pos_mask] - 1.0
                conformity[~pos_mask] = -decision_values[~pos_mask] - 1.0
                return -conformity
            # Conformity scores for both classes (negative, positive) vectorized
            # Column 0: assuming negative class, Column 1: assuming positive class
            conformity_matrix = np.column_stack(
                (
                    -decision_values - 1.0,  # negative class margin distance
                    decision_values - 1.0,  # positive class margin distance
                ),
            )
            return -conformity_matrix
        # Multi-class case: use one-vs-rest decision values
        if y_true is not None:
            if isinstance(y_true, pd.Series):
                y_true = y_true.to_numpy()
            if classes is None:
                raise ValueError(
                    "classes parameter is required for multiclass nonconformity computation"
                )
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            class_indexes = np.fromiter(
                (class_to_idx[label] for label in y_true),
                dtype=np.intp,
                count=len(y_true),
            )
            conformity = (
                y_score[np.arange(len(y_true)), class_indexes].astype(np.float64) - 1.0
            )
            return -conformity
        conformity_matrix = y_score.astype(np.float64) - 1.0
        return -conformity_matrix

    def get_name(self) -> str:  # noqa: PLR6301
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.
        """
        return "svm_margin"


# Registry for built-in nonconformity functions
NONCONFORMITY_REGISTRY = {
    "hinge": HingeNonconformity,
    "margin": MarginNonconformity,
    "log": LogNonconformity,
    "svm_margin": SVMMarginNonconformity,
}


def create_nonconformity_function(
    nonconformity: str | NonconformityFunctor | None,
) -> NonconformityFunctor | None:
    """Create a nonconformity function wrapper.

    Parameters
    ----------
    nonconformity : str | NonconformityFunctor | None
        The nonconformity specification.

    Returns
    -------
    NonconformityFunctor | None
        The wrapped nonconformity function, or None if input is None.

    Raises
    ------
    ValueError
        If the nonconformity specification is invalid.

    """
    if nonconformity is None:
        return None

    if isinstance(nonconformity, NonconformityFunctor):
        return nonconformity

    if isinstance(nonconformity, str):
        if nonconformity in NONCONFORMITY_REGISTRY:
            return NONCONFORMITY_REGISTRY[nonconformity]()  # type: ignore[abstract]
        if nonconformity.startswith("custom_"):
            raise ValueError(
                f"Cannot restore custom function from name: {nonconformity}. "
                "Custom functions require the original function object.",
            )
        raise ValueError(
            f"Unknown nonconformity function '{nonconformity}'. "
            f"Available options: {list(NONCONFORMITY_REGISTRY.keys())}",
        )

    raise ValueError(
        f"Invalid nonconformity specification: {type(nonconformity)}. "
        "Expected str, NonconformityFunctor, or None.",
    )
