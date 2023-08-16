"""Definition of types used in molpipeline."""
from __future__ import annotations
from typing import Any, Literal, Optional, Protocol

import numpy.typing as npt


from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]

OptionalMol = Optional[RDKitMol]

NoneHandlingOptions = Literal["raise", "record_remove", "fill_dummy"]


class AnyPredictor(Protocol):
    """Protocol for predictors."""

    def fit_predict(self, X: npt.NDArray[Any], y: npt.NDArray[Any], **fit_params: Any) -> npt.NDArray[Any]:
        """Fit the model with X and return predictions.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: npt.NDArray[Any]
            Target values.
        fit_params: Any
            Additional parameters for fitting.

        Returns
        -------
        npt.NDArray[Any]
            Predictions.
        """

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any], **fit_params: Any) -> None:
        """Fit the model with X.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: npt.NDArray[Any]
            Target values.
        fit_params: Any
            Additional parameters for fitting.


        Returns
        -------
        None
        """


class AnyTransformer(Protocol):
    """Protocol for transformers."""

    def fit_transform(
        self, X: npt.NDArray[Any], y: npt.NDArray[Any], **fit_params: Any
    ) -> npt.NDArray[Any]:
        """Fit the model with X and return the transformed array.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: npt.NDArray[Any]
            Target values.
        fit_params: Any
            Additional parameters for fitting.


        Returns
        -------
        npt.NDArray[Any]
            Transformed array.
        """

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any], **fit_params: Any) -> None:
        """Fit the model with X.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: npt.NDArray[Any]
            Target values.
        fit_params: Any
            Additional parameters for fitting.


        Returns
        -------
        None
        """
