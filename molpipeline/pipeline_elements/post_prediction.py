"""Module for elements that are applied in the pipeline after the prediction."""

from __future__ import annotations

import abc
from typing import Any, Union, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from numpy import typing as npt
from sklearn.base import BaseEstimator, clone, TransformerMixin

from molpipeline.abstract_pipeline_elements.core import ABCPipelineElement
from molpipeline.pipeline_elements.none_handling import NoneFiller
from molpipeline.utils.molpipeline_types import AnyPredictor, AnyTransformer


class PostPredictionTransformation(BaseEstimator, TransformerMixin, abc.ABC):
    """Abstract base class for post prediction steps.

    Post prediction steps are used to transform the output after the prediction step.
    E.g. dummy values are added for entries removed during preprocessing.
    """

    @abc.abstractmethod
    def transform(self, X: Any) -> Any:  # pylint: disable=invalid-name
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


class PostPredictionWrapper(PostPredictionTransformation):
    """Wrapper for post prediction transformations.

    This class is used to wrap a PipelineElement in a PostPredictionTransformation.
    """

    def __init__(
        self,
        wrapped_estimator: Union[AnyPredictor, AnyTransformer, NoneFiller],
        **kwargs: Any,
    ) -> None:
        """Initialize PostPredictionWrapper.

        Parameters
        ----------
        wrapped_estimator: PipelineElement
            PipelineElement to be wrapped.
        kwargs: Any
            Parameter of the wrapped_estimator
        Returns
        -------
        None
        """
        self.wrapped_estimator = wrapped_estimator
        if kwargs:
            if isinstance(self.wrapped_estimator, ABCPipelineElement):
                self.wrapped_estimator.set_params(kwargs)
            else:
                self.wrapped_estimator.set_params(**kwargs)

    def fit(
        self,
        X: npt.NDArray[Any],  # pylint: disable=invalid-name
        y: Optional[npt.NDArray[Any]] = None,  # pylint: disable=invalid-name
    ) -> Self:
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
        if isinstance(self.wrapped_estimator, NoneFiller):
            self.wrapped_estimator.fit(X)
        else:
            self.wrapped_estimator.fit(X, y)
        return self

    def transform(
        self, X: npt.NDArray[Any]  # pylint: disable=invalid-name
    ) -> npt.NDArray[Any]:
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
        if hasattr(self.wrapped_estimator, "predict"):
            return self.wrapped_estimator.predict(X)
        if hasattr(self.wrapped_estimator, "transform"):
            return self.wrapped_estimator.transform(X)
        raise AttributeError(
            f"Estimator {self.wrapped_estimator} has neither predict nor transform method."
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
        if hasattr(self.wrapped_estimator, "fit_predict"):
            return self.wrapped_estimator.fit_predict(X, y, **fit_params)
        if hasattr(self.wrapped_estimator, "fit_transform"):
            if isinstance(self.wrapped_estimator, NoneFiller):
                return self.wrapped_estimator.fit_transform(X)
            return self.wrapped_estimator.fit_transform(X, y, **fit_params)
        raise AttributeError(
            f"Estimator {self.wrapped_estimator} has neither fit_predict nor fit_transform method."
        )

    def inverse_transform(
        self, X: npt.NDArray[Any]  # pylint: disable=invalid-name
    ) -> npt.NDArray[Any]:
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
        if hasattr(self.wrapped_estimator, "inverse_transform"):
            return self.wrapped_estimator.inverse_transform(X)
        raise AttributeError(
            f"Estimator {self.wrapped_estimator} has no inverse_transform method."
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
        if deep:
            param_dict = {
                "wrapped_estimator": clone(self.wrapped_estimator),
            }
        else:
            param_dict = {
                "wrapped_estimator": self.wrapped_estimator,
            }
        param_dict.update(self.wrapped_estimator.get_params(deep=deep))
        return param_dict

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
        param_copy = dict(params)
        wrapped_estimator = param_copy.pop("wrapped_estimator")
        if wrapped_estimator:
            self.wrapped_estimator = wrapped_estimator
        if param_copy:
            if isinstance(self.wrapped_estimator, ABCPipelineElement):
                self.wrapped_estimator.set_params(param_copy)
            else:
                self.wrapped_estimator.set_params(**param_copy)
        return self
