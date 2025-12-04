"""Conformal prediction utils."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from crepes.extras import hinge, margin


class NonconformityFunctor(ABC):
    """Abstract base class for nonconformity functions.

    Simple wrapper for nonconformity functions with sklearn compatibility.
    Supports pickle serialization and parameter management.

    All nonconformity functions return scores for all classes.
    Use extract_true_class_scores() to get scores for the true class only.
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
            Class labels (for compatibility with crepes).
        y_true : npt.NDArray[Any] | None, optional
            True class labels (for compatibility with crepes).

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.
            - If y_true provided: shape (n_samples,) for true class only
            - Otherwise: shape (n_samples, n_classes) for all classes

        """

    @abstractmethod
    def get_name(self) -> str:
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """

    @staticmethod
    def extract_true_class_scores(
        nc_scores: npt.NDArray[np.float64],
        y_true: npt.NDArray[Any],
        classes: npt.NDArray[Any],
    ) -> npt.NDArray[np.float64]:
        """Extract nonconformity scores for the true class from a matrix of scores.

        Parameters
        ----------
        nc_scores : npt.NDArray[np.float64]
            Nonconformity scores of shape (n_samples, n_classes).
        y_true : npt.NDArray[Any]
            True class labels of shape (n_samples,).
        classes : npt.NDArray[Any]
            Unique class labels corresponding to columns in nc_scores.

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores for the true class, shape (n_samples,).
        """
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        class_indexes = np.fromiter(
            (class_to_idx[label] for label in y_true),
            dtype=np.intp,
            count=len(y_true),
        )

        return nc_scores[np.arange(len(y_true)), class_indexes]


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
            Class labels (for compatibility with crepes).
        y_true : npt.NDArray[Any] | None, optional
            True class labels (for compatibility with crepes).

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.
            - If y_true provided: shape (n_samples,) for true class only
            - Otherwise: shape (n_samples, n_classes) for all classes

        """
        nc_scores = hinge(y_prob, None, None)

        if y_true is not None and classes is not None:
            return self.extract_true_class_scores(nc_scores, y_true, classes)
        return nc_scores

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
            Class labels (for compatibility with crepes).
        y_true : npt.NDArray[Any] | None, optional
            True class labels (for compatibility with crepes).

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.
            - If y_true provided: shape (n_samples,) for true class only
            - Otherwise: shape (n_samples, n_classes) for all classes

        """
        nc_scores = margin(y_prob, None, None)

        if y_true is not None and classes is not None:
            return self.extract_true_class_scores(nc_scores, y_true, classes)
        return nc_scores

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
    ) -> npt.NDArray[np.float64]:
        """Compute log non-conformity scores for conformal classifiers.

        Parameters
        ----------
        y_prob : array-like of shape (n_samples, n_classes)
            predicted class probabilities

        Returns
        -------
        scores : ndarray of shape (n_samples, n_classes)
            non-conformity scores for all classes.

        """
        return -np.log(np.maximum(y_prob, 1e-10))

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
            Class labels (for compatibility with crepes).
        y_true : npt.NDArray[Any] | None, optional
            True class labels (for compatibility with crepes).

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.
            - If y_true provided: shape (n_samples,) for true class only
            - Otherwise: shape (n_samples, n_classes) for all classes

        """
        nc_scores = self.log_nc(y_prob)

        if y_true is not None and classes is not None:
            return self.extract_true_class_scores(nc_scores, y_true, classes)
        return nc_scores

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
    For binary SVM with decision function output d:
    Assuming true positive class: conformity = d - 1
    Assuming true negative class: conformity = -d - 1

    """

    def __call__(
        self,
        y_score: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute SVM margin-based conformity scores for binary classification.

        Note: This function expects decision function values (distances to hyperplane)
        rather than probabilities. For sklearn SVC, use decision_function() output.

        Parameters
        ----------
        y_score : npt.NDArray[np.float64]
            Decision function values (signed distances to hyperplane) from SVM.
            For binary classification, shape should be (n_samples,) or (n_samples, 1).
        classes : npt.NDArray[Any] | None, optional
            Class labels (for compatibility with crepes).
        y_true : npt.NDArray[Any] | None, optional
            True class labels (for compatibility with crepes).

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores.
            - If y_true provided: shape (n_samples,) for true class only
            - Otherwise: shape (n_samples, 2) with scores for both classes
        """
        decision_values = y_score.flatten()

        # Always compute conformity scores for both classes (negative, positive)
        # Column 0: assuming negative class, Column 1: assuming positive class
        conformity_matrix = np.column_stack(
            (
                -decision_values - 1.0,  # negative class margin distance
                decision_values - 1.0,  # positive class margin distance
            ),
        )
        nc_scores = -conformity_matrix

        if y_true is not None and classes is not None:
            return self.extract_true_class_scores(nc_scores, y_true, classes)
        return nc_scores

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
