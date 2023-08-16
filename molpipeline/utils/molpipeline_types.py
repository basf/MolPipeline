"""Definition of types used in molpipeline."""
from __future__ import annotations
from typing import Any, Literal, Optional, Protocol
try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy.typing as npt


from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]

OptionalMol = Optional[RDKitMol]

NoneHandlingOptions = Literal["raise", "record_remove", "fill_dummy"]


class AnySklearnEstimator(Protocol):
    """Protocol for sklearn estimators."""
    def get_params(self, deep: bool) -> dict[str, Any]:
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
        None
        """


class AnyPredictor(AnySklearnEstimator, Protocol):
    """Protocol for predictors."""

    def fit_predict(self, X: npt.NDArray[Any], y: Optional[npt.NDArray[Any]], **fit_params: Any) -> npt.NDArray[Any]:
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

    def fit(self, X: npt.NDArray[Any], y: Optional[npt.NDArray[Any]], **fit_params: Any) -> None:
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
        None
        """


class AnyTransformer(AnySklearnEstimator, Protocol):
    """Protocol for transformers."""

    def fit_transform(
        self, X: npt.NDArray[Any], y: Optional[npt.NDArray[Any]], **fit_params: Any
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

    def fit(self, X: npt.NDArray[Any], y: Optional[npt.NDArray[Any]], **fit_params: Any) -> None:
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
        None
        """
