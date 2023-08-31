"""Module for elements that are applied in the pipeline after the prediction."""

from __future__ import annotations

import abc
from typing import Any, Union, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from numpy import typing as npt
from sklearn.base import BaseEstimator, TransformerMixin

from molpipeline.pipeline_elements.none_handling import NoneFiller
from molpipeline.utils.molpipeline_types import AnyPredictor, AnyTransformer


class PostPredictionTransformation(BaseEstimator, TransformerMixin, abc.ABC):
    """Abstract base class for post prediction steps.

    Post prediction steps are used to transform the output after the prediction step.
    E.g. dummy values are added for entries removed during preprocessing.
    """

    @abc.abstractmethod
    def transform(self, Xt: Any) -> Any:
        """Transform data.

        Parameters
        ----------
        Xt: npt.NDArray[Any]
            Input data.

        Returns
        -------
        npt.NDArray[Any]
            Transformed data.
        """


class PostPredictionWrapper(PostPredictionTransformation):
    """Wrapper for post prediction transformations.

    This class is used to wrap a PipelineElement in a PostPredictionTransformation.
    """

    def __init__(
        self, estimator: Union[AnyPredictor, AnyTransformer, NoneFiller]
    ) -> None:
        """Initialize PostPredictionWrapper.

        Parameters
        ----------
        estimator: PipelineElement
            PipelineElement to be wrapped.
        """
        self.estimator = estimator

    def fit(self, X: npt.NDArray[Any], y: Optional[npt.NDArray[Any]] = None) -> Self:
        """Fit PostPredictionWrapper.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Input data.
        y: Optional[npt.NDArray[Any]]
            Target data.

        Returns
        -------
        Self
            Fitted PostPredictionWrapper.
        """
        if isinstance(self.estimator, NoneFiller):
            self.estimator.fit(X)
        else:
            self.estimator.fit(X, y)
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Transform data.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Input data.

        Returns
        -------
        npt.NDArray[Any]
            Transformed data.
        """
        if hasattr(self.estimator, "predict"):
            return self.estimator.predict(X)
        if hasattr(self.estimator, "transform"):
            return self.estimator.transform(X)
        raise AttributeError(
            f"Estimator {self.estimator} has neither predict nor transform method."
        )

    def fit_transform(
        self,
        X: npt.NDArray[Any],
        y: Optional[npt.NDArray[Any]] = None,
        **fit_params: Any,
    ) -> npt.NDArray[Any]:
        """Fit and transform data.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Input data.
        y: npt.NDArray[Any]
            Target data.
        **fit_params: Any
            Additional parameters for fitting.

        Returns
        -------
        npt.NDArray[Any]
            Transformed data.
        """
        if hasattr(self.estimator, "fit_predict"):
            return self.estimator.fit_predict(X, y, **fit_params)
        if hasattr(self.estimator, "fit_transform"):
            if isinstance(self.estimator, NoneFiller):
                return self.estimator.fit_transform(X)
            return self.estimator.fit_transform(X, y, **fit_params)
        raise AttributeError(
            f"Estimator {self.estimator} has neither fit_predict nor fit_transform method."
        )

    def inverse_transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Inverse transform data.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Input data.

        Returns
        -------
        npt.NDArray[Any]
            Inverse transformed data.
        """
        if hasattr(self.estimator, "inverse_transform"):
            return self.estimator.inverse_transform(X)
        raise AttributeError(
            f"Estimator {self.estimator} has no inverse_transform method."
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters.

        Parameters
        ----------
        deep: bool
            Whether to recursively get parameters.

        Returns
        -------
        dict[str, Any]
            Parameters.
        """
        return {"estimator": self.estimator}

    def set_params(self, **params: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        **params: Any
            Parameters to set.

        Returns
        -------
        dict[str, Any]
            Parameters.
        """
        self.estimator.set_params(**params)
        return self
