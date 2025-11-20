"""Conformal prediction utils."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from crepes.extras import hinge, margin


def log(
    x_prob: npt.NDArray[Any],
    classes: npt.NDArray[Any] | None = None,
    y: npt.NDArray[Any] | pd.Series | None = None,
) -> npt.NDArray[Any]:
    """Compute log non-conformity scores for conformal classifiers.

    Parameters
    ----------
    x_prob : array-like of shape (n_samples, n_classes)
        predicted class probabilities
    classes : array-like of shape (n_classes,), default=None
        class names
    y : array-like of shape (n_samples,), default=None
        correct target values

    Returns
    -------
    scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
        non-conformity scores. The shape is (n_samples, n_classes)
        if classes and y are None.

    Examples
    --------
    Assuming that ``x_prob`` is an array with predicted probabilities and
    ``classes`` and ``y`` are vectors with the class names (in order) and
    correct class labels, respectively, the non-conformity scores are generated
    by:

    .. code-block:: python

       from crepes.extras import log

    alphas = log(x_prob, classes, y)

    The above results in that ``alphas`` is assigned a vector of the same length
    as ``x_prob`` with a non-conformity score for each object, here
    defined as negative log of the predicted probability for the correct class label.
    These scores can be used when fitting a :class:`.ConformalClassifier` or calibrating a
    :class:`.WrapClassifier`. Non-conformity scores for test objects, for which
    ``y`` is not known, can be obtained from the corresponding predicted
    probabilities (``x_prob_test``) by:

    .. code-block:: python

    alphas_test = log(x_prob_test)

    The above results in that ``alphas_test`` is assigned an array of the same
    shape as ``x_prob_test`` with non-conformity scores for each class in the
    columns for each test object.

    """
    if y is not None:
        if isinstance(y, pd.Series):
            y = np.asarray(y.values)
        class_indexes = np.array(
            [np.argwhere(classes == y[i])[0][0] for i in range(len(y))], dtype=np.intp
        )
        result = -np.log(np.maximum(x_prob[np.arange(len(y)), class_indexes], 1e-10))
    else:
        result = -np.log(np.maximum(x_prob, 1e-10))
    return result


class NonconformityFunctor(ABC):
    """Abstract base class for nonconformity functions.

    Simple wrapper for nonconformity functions with sklearn compatibility.
    Supports pickle serialization and parameter management.
    """

    @abstractmethod
    def __call__(
        self,
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute nonconformity scores.

        Parameters
        ----------
        x_prob : npt.NDArray[Any]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[Any]
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
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute hinge nonconformity scores.

        Parameters
        ----------
        x_prob : npt.NDArray[Any]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[Any]
            Nonconformity scores.

        """
        return hinge(x_prob, classes, y)

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
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute margin nonconformity scores.

        Parameters
        ----------
        x_prob : npt.NDArray[Any]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[Any]
            Nonconformity scores.

        """
        return margin(x_prob, classes, y)

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

    def __call__(
        self,
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute log nonconformity scores.

        Parameters
        ----------
        x_prob : npt.NDArray[Any]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[Any]
            Nonconformity scores.

        """
        return log(x_prob, classes, y)

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

    Based on Pereira et al. (2020) "Conformal measure for SVM".
    Uses the distance to the margin boundary of the predicted class.

    The conformity score is computed as the distance between the instance
    and the margin boundary of the class under consideration. For a test
    instance, the signed distance to the separating hyperplane is d_h = y/||w||,
    and the distance to the margin boundary is d_mb = |d_h - 1/||w||.

    Reference:
        T. Pereira, et al. "Optimizing blood-brain barrier crossing with a
        confidence predictor". Journal of Biomedical Informatics 101 (2020) 103350.
    """

    def __call__(
        self,
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute SVM margin-based conformity scores.

        Note: This function expects decision function values (distances to hyperplane)
        rather than probabilities. For sklearn SVC, use decision_function() output.

        Parameters
        ----------
        x_prob : npt.NDArray[Any]
            Decision function values (signed distances to hyperplane) from SVM.
            For binary classification, shape should be (n_samples,).
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y : npt.NDArray[Any] | None, optional
            True labels for calibration. If provided, returns conformity scores
            for the true class. If None, returns scores for all classes.

        Returns
        -------
        npt.NDArray[Any]
            Conformity scores (higher = more conforming).
            - If y is provided: (n_samples,) with conformity for true class
            - If y is None: (n_samples, n_classes) with conformity for each class

        """
        # For binary classification with decision_function output
        if x_prob.ndim == 1 or (x_prob.ndim == 2 and x_prob.shape[1] == 1):
            decision_values = x_prob.flatten()

            if y is not None:
                # Convert class labels to {-1, +1} if needed
                if isinstance(y, pd.Series):
                    y = y.values
                if classes is not None:
                    y_mapped = np.where(y == classes[1], 1, -1)
                else:
                    y_mapped = np.where(y == 1, 1, -1)
                # Vectorized conformity computation
                pos_mask = y_mapped == 1
                conformity = np.empty_like(decision_values, dtype=np.float64)
                conformity[pos_mask] = decision_values[pos_mask] - 1.0
                conformity[~pos_mask] = -decision_values[~pos_mask] - 1.0
                return -conformity  # Nonconformity
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
        # Each column is the decision value for that class
        if y is not None:
            if isinstance(y, pd.Series):
                y = np.asarray(y.values)
            class_indexes = np.array(
                [np.argwhere(classes == y[i])[0][0] for i in range(len(y))],
                dtype=np.intp,
            )
            # Get decision values for true class
            # Conformity = decision_value - 1 (distance to margin)
            conformity = (
                x_prob[np.arange(len(y)), class_indexes].astype(np.float64) - 1.0
            )
            return -conformity  # Nonconformity
        # Return conformity for all classes
        # Ensure floating dtype to satisfy type checking (mypy) and downstream expectations
        conformity_matrix = x_prob.astype(np.float64) - 1.0
        return -conformity_matrix  # Nonconformity

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
