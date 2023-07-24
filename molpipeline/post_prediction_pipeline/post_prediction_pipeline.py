"""Post prediction pipeline for modifying the predictions of a model."""
from __future__ import annotations
from typing import Any, Union

import numpy.typing as npt

from molpipeline.utils.molpipeline_types import AnyPredictor, AnyTransformer


class PostPredictionPipeline:
    """Pipeline for post prediction processing.

    Attributes
    ----------
    _name: str
        Name of the pipeline.
    _step_list: list[tuple[str, Union[AnyPredictor, AnyTransformer]]]
        List of tuples containing the name and the pipeline element.
    """

    name: str
    _step_list: list[tuple[str, Union[AnyPredictor, AnyTransformer]]]

    def __init__(
        self,
        step_list: list[tuple[str, Union[AnyPredictor, AnyTransformer]]],
        name: str = "PostPredictionPipeline",
    ) -> None:
        """Initialize PostPredictionPipeline.

        Parameters
        ----------
        step_list: list[tuple[str, Union[AnyPredictor, AnyTransformer]]]
            List of tuples containing the name and the pipeline element.
        name: str
            Name of the pipeline.

        Returns
        -------
        None
        """
        self._name = name
        self._step_list = step_list

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict labels for X.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Features used to derive the prediction.

        Returns
        -------
        npt.NDArray[Any]
            Predicted values for X.
        """
        X_t = X
        for name, step in self._step_list:
            if hasattr(step, "transform"):
                X_t = step.transform(X_t)
            elif hasattr(step, "predict"):
                X_t = step.predict(X_t)
            else:
                raise ValueError(
                    f"Step {name} has neither transform nor predict method."
                )
        return X_t

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> None:
        """Fit the model with X.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Model input.
        y: npt.NDArray[Any]
            Target values.

        Returns
        -------
        None
        """
        self.fit_predict(X, y)

    def fit_predict(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Fit and predict labels for X.

        Parameters
        ----------
        X: npt.NDArray[Any]
            Features to fit the model with.
        y: npt.NDArray[Any]
            Target values to fit the model with.

        Returns
        -------
        npt.NDArray[Any]
            Predictions of the fitted model for X.
        """
        X_t = X
        for name, step in self._step_list:
            if hasattr(step, "fit_transform"):
                X_t = step.fit_transform(X_t, y)
            elif hasattr(step, "fit_predict"):
                X_t = step.fit_predict(X_t, y)
            else:
                raise ValueError(
                    f"Step {name} has neither fit_transform nor fit_predict method."
                )
        return X_t

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: bool
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params: dict[str, Any]
            Parameter names mapped to their values.
        """
        params: dict[str, Any] = {'name': self._name, 'step_list': self._step_list}
        return params
