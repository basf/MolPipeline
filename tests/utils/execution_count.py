"""Functions for counting the number of times a function is executed."""

from __future__ import annotations

from typing import Any

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor

from molpipeline import Pipeline
from molpipeline.abstract_pipeline_elements.core import ABCPipelineElement
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToMorganFP


class CountingTransformerWrapper(BaseEstimator):
    """A transformer that counts the number of transformations."""

    def __init__(self, element: ABCPipelineElement):
        """Initialize the wrapper.

        Parameters
        ----------
        element : ABCPipelineElement
            The element to wrap.
        """
        self.element = element
        self.n_transformations = 0

    def fit(self, X: Any, y: Any) -> Self:  # pylint: disable=invalid-name
        """Fit the data.

        Parameters
        ----------
        X : Any
            The input data.
        y : Any
            The target data.

        Returns
        -------
        Any
            The fitted data.
        """
        self.element.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:  # pylint: disable=invalid-name
        """Transform the data.

        Transform is called during prediction, which is not cached.
        Since the transformer is not cached, the counter is not increased.

        Parameters
        ----------
        X : Any
            The input data.

        Returns
        -------
        Any
            The transformed data.
        """
        return self.element.transform(X)

    def fit_transform(self, X: Any, y: Any) -> Any:  # pylint: disable=invalid-name
        """Fit and transform the data.

        Parameters
        ----------
        X : Any
            The input data.
        y : Any
            The target data.

        Returns
        -------
        Any
            The transformed data.
        """
        self.n_transformations += 1
        return self.element.fit_transform(X, y)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters of the transformer.

        Parameters
        ----------
        deep : bool
            If True, the parameters of the transformer are also returned.

        Returns
        -------
        dict[str, Any]
            The parameters of the transformer.
        """
        params = {
            "element": self.element,
        }
        if deep:
            params.update(self.element.get_params(deep))
        return params

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the transformer.

        Parameters
        ----------
        **params
            The parameters to set.

        Returns
        -------
        Self
            The transformer with the set parameters
        """
        element = params.pop("element", None)
        if element is not None:
            self.element = element
        self.element.set_params(**params)
        return self


def get_exec_counted_rf_regressor(random_state: int) -> Pipeline:
    """Get a morgan + random forest pipeline, which counts the number of transformations.

    Parameters
    ----------
    random_state : int
        The random state to use.

    Returns
    -------
    Pipeline
        A pipeline with a morgan fingerprint, physchem descriptors, and a random forest
    """
    smi2mol = SmilesToMol()

    mol2concat = CountingTransformerWrapper(
        MolToMorganFP(radius=2, n_bits=2048),
    )
    rf = RandomForestRegressor(random_state=random_state, n_jobs=1)
    return Pipeline(
        [
            ("smi2mol", smi2mol),
            ("mol2concat", mol2concat),
            ("rf", rf),
        ],
        n_jobs=1,
    )
