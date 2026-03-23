"""Conformal prediction wrappers for classification and regression using crepes."""

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from crepes.extras import hinge, margin
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from typing_extensions import override


class NonconformityFunctor(ABC):
    """Abstract base class for nonconformity functions.

    Simple wrapper for nonconformity functions with sklearn compatibility.
    Supports pickle serialization and parameter management.

    All nonconformity functions return scores for all classes.
    Use extract_true_class_scores() to get scores for the true class only.
    """

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
        if y_true is not None and classes is not None:
            return self.extract_true_class_scores(y_prob, y_true, classes)
        return self.calculate_nonconformity(y_prob, None, None)

    @abstractmethod
    def get_name(self) -> str:
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """

    @abstractmethod
    def calculate_nonconformity(
        self,
        y_score: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute nonconformity scores.

        Parameters
        ----------
        y_score : npt.NDArray[np.float64]
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

    def extract_true_class_scores(
        self,
        y_score: npt.NDArray[np.float64],
        y_true: npt.NDArray[Any],
        classes: npt.NDArray[Any],
    ) -> npt.NDArray[np.float64]:
        """Extract nonconformity for the true class using a nonconformity function.

        This method calls the nc function to get all class scores, then extracts
        true class scores.

        Parameters
        ----------
        y_score : npt.NDArray[np.float64]
            Probability predictions or raw predictions.
        y_true : npt.NDArray[Any]
            True class labels of shape (n_samples,).
        classes : npt.NDArray[Any]
            Unique class labels corresponding to columns in scores.

        Returns
        -------
        npt.NDArray[np.float64]
            Nonconformity scores for the true class, shape (n_samples,).

        """
        nc_scores = self.calculate_nonconformity(y_score, classes=classes, y_true=None)
        class_indices = np.searchsorted(classes, y_true)
        return nc_scores[np.arange(len(y_true)), class_indices]


class HingeNonconformity(NonconformityFunctor):
    """Hinge loss nonconformity function for classification."""

    def get_name(self) -> str:  # noqa: PLR6301
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return "hinge"

    @override
    def calculate_nonconformity(
        self,
        y_score: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute hinge nonconformity scores.

        Parameters
        ----------
        y_score : npt.NDArray[np.float64]
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
        return hinge(y_score, classes, y_true)


class MarginNonconformity(NonconformityFunctor):
    """Margin nonconformity function for classification."""

    @override
    def get_name(self) -> str:
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return "margin"

    @override
    def calculate_nonconformity(
        self,
        y_score: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute margin nonconformity scores.

        Parameters
        ----------
        y_score : npt.NDArray[np.float64]
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
            - Otherwise: shapspecificatione (n_samples, n_classes) for all classes

        """
        return margin(y_score, classes, y_true)


class LogNonconformity(NonconformityFunctor):
    """Logarithmic nonconformity function for classification."""

    @override
    def get_name(self) -> str:
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return "log"

    @override
    def calculate_nonconformity(
        self,
        y_score: npt.NDArray[np.float64],
        classes: npt.NDArray[Any] | None = None,
        y_true: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute log nonconformity scores.

        Parameters
        ----------
        y_score : npt.NDArray[np.float64]
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
        return -np.log(np.maximum(y_score, 1e-10))


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

    def get_name(self) -> str:  # noqa: PLR6301
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return "svm_margin"

    @override
    def calculate_nonconformity(
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
        return np.column_stack(
            (
                1.0 + decision_values,  # negative class margin distance
                1.0 - decision_values,  # positive class margin distance
            ),
        )


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
    TypeError
        If the nonconformity is not a string, NonconformityFunctor, or None.

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

    raise TypeError(
        f"Invalid nonconformity specification: {type(nonconformity)}. "
        "Expected str, NonconformityFunctor, or None.",
    )


def _fit_antitonic_regressors(
    p_values_calib: npt.NDArray[np.float64],
    *,
    epsilon: float = 1e-10,
) -> list[IsotonicRegression]:
    """Fit per-class antitonic (decreasing) isotonic regressors.

    Parameters
    ----------
    p_values_calib : npt.NDArray[np.float64]
        Conformal p-values for calibration samples, shape (n_calib, n_classes).
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).

    Returns
    -------
    list[IsotonicRegression]
        One fitted regressor per class.

    """
    n_classes = p_values_calib.shape[1]
    regressors: list[IsotonicRegression] = []
    for y_class in range(n_classes):
        p_y_calib = p_values_calib[:, y_class]
        targets = 1.0 / (p_y_calib + epsilon)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(p_y_calib, targets)
        regressors.append(iso)
    return regressors


def _apply_antitonic_regressors(
    p_values_test: npt.NDArray[np.float64],
    regressors: list[IsotonicRegression],
    *,
    epsilon: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Apply fitted antitonic isotonic regressors to p-values.

    Parameters
    ----------
    p_values_test : npt.NDArray[np.float64]
        Conformal p-values for test samples, shape (n_test, n_classes).
    regressors : list[IsotonicRegression]
        Fitted per-class isotonic regressors.
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).

    Returns
    -------
    npt.NDArray[np.float64]
        Calibrated probabilities of shape (n_test, n_classes), normalized to sum to 1.

    Raises
    ------
    ValueError
        If number of regressors does not match number of classes.

    """
    n_classes = p_values_test.shape[1]
    if len(regressors) != n_classes:
        raise ValueError(
            "Number of regressors must match number of classes. "
            f"Got {len(regressors)} and {n_classes}",
        )

    calibrated_probs_unnorm = np.zeros_like(p_values_test)
    for y_class in range(n_classes):
        p_y_test = p_values_test[:, y_class]
        iso = regressors[y_class]

        g_test = iso.predict(p_y_test)
        g_1 = iso.predict([1.0])[0]
        calibrated_probs_unnorm[:, y_class] = g_1 / (g_test + epsilon)

    prob_sum = calibrated_probs_unnorm.sum(axis=1, keepdims=True)
    return calibrated_probs_unnorm / (prob_sum + epsilon)


class BaseConformalPredictor(BaseEstimator, ABC):  # pylint: disable=too-many-instance-attributes
    """Base class for conformal predictors providing common functionality."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        estimator: BaseEstimator,
        nonconformity: str | NonconformityFunctor | None = None,
        *,
        n_folds: int | None = None,
        mondrian: bool | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize BaseConformalPredictor.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to wrap.
        nonconformity : str | NonconformityFunctor | None, optional
            Nonconformity function to use.
        n_folds : int | None, optional
            Number of folds used for cross-conformal prediction.
        mondrian : bool | None, optional
            Whether to use Mondrian (class or group conditional) conformal prediction.
        random_state : int | None, optional
            Random seed used for generating cross-validation splits.
        **kwargs : Any
            Additional keyword arguments for configuration.

        """
        self.estimator = estimator
        self.nonconformity = nonconformity
        self.kwargs = kwargs
        self.mondrian = bool(mondrian) if mondrian is not None else False
        self.n_folds = n_folds
        check_random_state(random_state)  # for sklearn clone() compatibility
        self.random_state = random_state
        self.models_: list[BaseEstimator] = []
        self.cv_splits_: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] = []

    @property
    def nonconformity(self) -> str | NonconformityFunctor | None:
        """Get the nonconformity parameter value.

        This property is needed for sklearn's get_params/set_params compatibility.
        """
        if self.nonconformity_func is None:
            return None
        if isinstance(self.nonconformity_func, NonconformityFunctor):
            return self.nonconformity_func.get_name()
        return self.nonconformity_func

    @nonconformity.setter
    def nonconformity(self, value: str | NonconformityFunctor | None) -> None:
        """Set the nonconformity function.

        Parameters
        ----------
        value : str | NonconformityFunctor | None
            The nonconformity function to set.

        """
        self.nonconformity_func = create_nonconformity_function(value)

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict using wrapped or cross-conformal models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Returns
        -------
        npt.NDArray[Any]
            Predictions.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        """
        wrapper = getattr(self, "_crepes_wrapper", None)
        if wrapper is not None:
            return wrapper.predict(x)

        models = getattr(self, "models_", None)
        if not models:
            raise ValueError("Must fit before predicting")

        predictions = np.array([model.predict(x) for model in models])
        return np.mean(predictions, axis=0)

    @abstractmethod
    def evaluate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate the conformal predictor. Must be implemented by subclasses.

        Parameters
        ----------
        x: np.ndarray
            Features to evaluate.
        y: np.ndarray
            True labels/targets.
        **kwargs : Any
            Additional parameters for evaluation.

        Returns
        -------
        dict[str, Any]
            Dictionary of evaluation metrics.

        """

    @staticmethod
    def _validate_confidence_level(confidence_level: float) -> float:
        """Validate confidence level parameter.

        Parameters
        ----------
        confidence_level : float
            Confidence level to validate.

        Returns
        -------
        float
            Validated confidence level.

        Raises
        ------
        ValueError
            If confidence level is not between 0 (exclusive) and 1 (inclusive).

        """
        if not isinstance(confidence_level, (int, float)):
            raise ValueError(
                f"confidence_level must be a number, got "
                f"{type(confidence_level).__name__}",
            )

        if not 0 < confidence_level <= 1:
            raise ValueError(
                f"confidence_level must be between 0 (exclusive) and 1 (inclusive), "
                f"got {confidence_level}",
            )

        if confidence_level < 0.5:
            warnings.warn(
                f"Confidence level {confidence_level} is less than 0.5 (50%). "
                "This represents weak confidence and may produce unreliable prediction "
                "sets/intervals.",
                UserWarning,
                stacklevel=3,
            )

        return confidence_level

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            Whether to return parameters of sub-estimators (default: True).

        Returns
        -------
        dict[str, Any]
            Parameter dictionary.

        """
        params = {
            "estimator": self.estimator,
            "nonconformity": self.nonconformity,
        }

        # Get other parameters from parent class (exclude handled keys)
        parent_params = {
            k: v
            for k, v in super().get_params(deep=deep).items()
            if k not in {"estimator", "kwargs", "nonconformity_func"}
        }
        params.update(parent_params)

        params.update(self.kwargs)

        if deep and hasattr(self.estimator, "get_params"):
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f"estimator__{k}": v for k, v in estimator_params.items()})

        return params

    def set_params(self, **params: Any) -> "BaseConformalPredictor":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : Any
            Parameters to set.

        Returns
        -------
        BaseConformalPredictor
            Self.

        """
        # Make a copy to avoid modifying the input dictionary
        params_copy = params.copy()

        # Convert nonconformity parameter to nonconformity_func attribute
        if "nonconformity" in params_copy:
            params_copy["nonconformity_func"] = create_nonconformity_function(
                params_copy.pop("nonconformity"),  # Remove original key
            )
        super().set_params(**params_copy)
        return self
