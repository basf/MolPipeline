"""Definition of types used in molpipeline."""
from __future__ import annotations

from numbers import Number
from typing import Any, List, Optional, Protocol, TypeVar

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from molpipeline.abstract_pipeline_elements.core import (
    OptionalMol,
    RDKitMol,
)


__all__ = [
    "AnyNumpyElement",
    "AnyPredictor",
    "AnySklearnEstimator",
    "AnyTransformer",
    "AnyIterable",
    "AnySklearnEstimator",
    "Number",
    "NumberIterable",
    "OptionalMol",
    "RDKitMol",
    "TypeConserverdIterable",
]
# One liner type definitions

AnyNumpyElement = TypeVar("AnyNumpyElement", bound=np.generic)

_T = TypeVar("_T")
# mypy: ignore-errors
AnyIterable = TypeVar("AnyIterable", List[_T], npt.NDArray[_T])

# mypy: ignore-errors
NumberIterable = TypeVar("NumberIterable", List[Number], npt.NDArray[Number])

TypeConserverdIterable = TypeVar("TypeConserverdIterable", List[_T], npt.NDArray[_T])


class AnySklearnEstimator(Protocol):
    """Protocol for sklearn estimators."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: bool
            If True, will return the parameters for this estimator.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.
        """

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of this estimator.

        Parameters
        ----------
        params: Any
            Estimator parameters.

        Returns
        -------
        Self
            Estimator with updated parameters.
        """

    def fit(
        self,
        X: npt.NDArray[Any],  # pylint: disable=invalid-name
        y: Optional[npt.NDArray[Any]],  # pylint: disable=invalid-name
        **fit_params: Any,
    ) -> Self:
        """Fit the model with X.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: Optional[npt.NDArray[Any]]
            Target values.
        fit_params: Any
            Additional parameters for fitting.


        Returns
        -------
        Self
            Fitted estimator.
        """


class AnyPredictor(AnySklearnEstimator, Protocol):
    """Protocol for predictors."""

    def fit_predict(
        self,
        X: npt.NDArray[Any],  # pylint: disable=invalid-name
        y: Optional[npt.NDArray[Any]],  # pylint: disable=invalid-name
        **fit_params: Any,
    ) -> npt.NDArray[Any]:
        """Fit the model with X and return predictions.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: Optional[npt.NDArray[Any]]
            Target values.
        fit_params: Any
            Additional parameters for fitting.

        Returns
        -------
        npt.NDArray[Any]
            Predictions.
        """


class AnyTransformer(AnySklearnEstimator, Protocol):
    """Protocol for transformers."""

    def fit_transform(
        self,
        X: npt.NDArray[Any],  # pylint: disable=invalid-name
        y: Optional[npt.NDArray[Any]],  # pylint: disable=invalid-name
        **fit_params: Any,
    ) -> npt.NDArray[Any]:
        """Fit the model with X and return the transformed array.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: Optional[npt.NDArray[Any]]
            Target values.
        fit_params: Any
            Additional parameters for fitting.


        Returns
        -------
        npt.NDArray[Any]
            Transformed array.
        """

    def transform(
        self,
        X: npt.NDArray[Any],  # pylint: disable=invalid-name
    ) -> npt.NDArray[Any]:
        """Transform and return X according to object protocol.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.

        Returns
        -------
        npt.NDArray[Any]
            Transformed array.
        """
