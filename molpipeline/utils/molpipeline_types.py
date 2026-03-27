"""Definition of types used in molpipeline."""

from collections.abc import Sequence
from numbers import Number
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spmatrix

from molpipeline.abstract_pipeline_elements.core import (
    ABCPipelineElement,
    OptionalMol,
    RDKitMol,
)
from molpipeline.error_handling import FilterReinserter

__all__ = [
    "AnyElement",
    "AnyNumpyElement",
    "AnyPredictor",
    "AnySklearnEstimator",
    "AnyStep",
    "AnyTransformer",
    "Number",
    "OptionalMol",
    "RDKitMol",
    "XVar",
    "YVar",
]
# One-liner type definitions

AnyNumpyElement = TypeVar("AnyNumpyElement", bound=np.generic)

_T = TypeVar("_T")
_NT = TypeVar("_NT", bound=np.generic)
TypeFixedVarSeq = TypeVar("TypeFixedVarSeq", bound=Sequence[_T] | npt.NDArray[_NT])  # type: ignore
AnyVarSeq = TypeVar("AnyVarSeq", bound=Sequence[Any] | npt.NDArray[Any])

SparseMatrix = csc_matrix[Any] | coo_matrix[Any] | csr_matrix[Any]
XType = npt.ArrayLike | npt.NDArray[Any] | spmatrix  # Generic model input features
YType = npt.ArrayLike | npt.NDArray[Any] | None  # Generic model target values
# XVar is for the case the input has the same type bis is modified in other ways
# e.g. row removal or value manipulations. Defines parameter AND return type!
XVar = TypeVar("XVar", bound=npt.ArrayLike | npt.NDArray[Any] | spmatrix)
# Same as XVar but for target values.
YVar = TypeVar("YVar", bound=npt.ArrayLike | npt.NDArray[Any] | None)

# FloatCountRange needs renaming to FloatRange
FloatCountRange: TypeAlias = tuple[float | None, float | None]
# IntCountRange needs renaming to IntRange
IntCountRange: TypeAlias = tuple[int | None, int | None]

# IntOrIntCountRange for Typing of count ranges
# - a single int for an exact value match
# - a range given as a tuple with a lower and upper bound
#   - both limits are optional
IntOrIntCountRange: TypeAlias = int | IntCountRange


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
        ...  # pylint: disable=unnecessary-ellipsis

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
        ...  # pylint: disable=unnecessary-ellipsis

    def fit(
        self,
        X: XType,  # noqa: N803
        y: YType,
        **fit_params: Any,
    ) -> Self:
        """Fit the model with X.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: npt.NDArray[Any] | None
            Target values.
        fit_params: Any
            Additional parameters for fitting.


        Returns
        -------
        Self
            Fitted estimator.

        """
        ...  # pylint: disable=unnecessary-ellipsis


class AnyPredictor(AnySklearnEstimator, Protocol):
    """Protocol for predictors."""

    def fit_predict(
        self,
        X: XType,  # noqa: N803
        y: YType,
        **fit_params: Any,
    ) -> npt.NDArray[Any]:
        """Fit the model with X and return predictions.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: npt.NDArray[Any] | None
            Target values.
        fit_params: Any
            Additional parameters for fitting.

        Returns
        -------
        npt.NDArray[Any]
            Predictions.

        """
        ...  # pylint: disable=unnecessary-ellipsis


class AnyTransformer(AnySklearnEstimator, Protocol):
    """Protocol for transformers."""

    def fit_transform(
        self,
        X: npt.NDArray[Any],  # noqa: N803
        y: npt.NDArray[Any] | None,
        **fit_params: Any,
    ) -> npt.NDArray[Any]:
        """Fit the model with X and return the transformed array.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: npt.NDArray[Any] | None
            Target values.
        fit_params: Any
            Additional parameters for fitting.


        Returns
        -------
        npt.NDArray[Any]
            Transformed array.

        """
        ...  # pylint: disable=unnecessary-ellipsis

    def transform(
        self,
        X: npt.NDArray[Any],  # noqa: N803
        **params: Any,
    ) -> npt.NDArray[Any]:
        """Transform and return X according to object protocol.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        params: Any
            Additional parameters for transforming.

        Returns
        -------
        npt.NDArray[Any]
            Transformed array.

        """
        ...  # pylint: disable=unnecessary-ellipsis


AnyElement = (
    AnyTransformer
    | AnyPredictor
    | ABCPipelineElement
    | Literal["passthrough"]
    | FilterReinserter[Any]
)
AnyStep = tuple[str, AnyElement]
