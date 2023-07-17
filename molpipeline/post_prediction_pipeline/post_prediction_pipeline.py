from __future__ import annotations
from typing import Any, Protocol, Union

import numpy.typing as npt



class PostPredictionPipeline:
    def __init__(
            self,
            step_list: list[tuple[str, Union[_AnyPredictor, _AnyTransformer]]],
            name: str = "PostPredictionPipeline"
    ) -> None:
        """Initialize PostPredictionPipeline.

        Parameters
        ----------
        name: str
            Name of the pipeline.
        """
        self._name = name
        self._pipeline_elements = step_list

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict labels for X.

        Parameters
        ----------
        X: npt.NDArray[Any]

        Returns
        -------
        npt.NDArray[Any]
        """
        X_t = X
        for name, step in self._pipeline_elements:
            if hasattr(step, "transform"):
                X_t = step.transform(X_t)
            elif hasattr(step, "predict"):
                X_t = step.predict(X_t)
            else:
                raise ValueError(f"Step {name} has neither transform nor predict method.")
        return X_t

    def fit_predict(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Fit and predict labels for X.

        Parameters
        ----------
        X: npt.NDArray[Any]
        y: npt.NDArray[Any]

        Returns
        -------
        npt.NDArray[Any]
        """
        X_t = X
        for name, step in self._pipeline_elements:
            if hasattr(step, "fit_transform"):
                X_t = step.fit_transform(X_t, y)
            elif hasattr(step, "fit_predict"):
                X_t = step.fit_predict(X_t, y)
            else:
                raise ValueError(f"Step {name} has neither fit_transform nor fit_predict method.")
        return X_t
