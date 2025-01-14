"""Abstract classes for transforming rdkit molecules to float vectors."""

# pylint: disable=too-many-arguments

from __future__ import annotations

import abc
from typing import Any, Iterable, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToAnyPipelineElement,
)
from molpipeline.utils.molpipeline_types import AnyTransformer, RDKitMol


class MolToDescriptorPipelineElement(MolToAnyPipelineElement):
    """PipelineElement which generates a matrix from descriptor-vectors of each molecule."""

    _standardizer: Optional[AnyTransformer]
    _output_type = "float"
    _feature_names: list[str]

    def __init__(
        self,
        standardizer: Optional[AnyTransformer] = StandardScaler(),
        name: str = "MolToDescriptorPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToDescriptorPipelineElement.

        Parameters
        ----------
        standardizer: Optional[AnyTransformer], default=StandardScaler()
            The output is post_processed according to the standardizer if not None.
        name: str:
            Name of the PipelineElement.
        n_jobs: int:
            Number of jobs to use for parallelization.
        uuid: Optional[str]
            UUID of the PipelineElement.

        Returns
        -------
        None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self._standardizer = standardizer
        if self._standardizer is not None:
            self._requires_fitting = True
        self._mean = None
        self._std = None

    @property
    @abc.abstractmethod
    def n_features(self) -> int:
        """Return the number of features."""

    @property
    def feature_names(self) -> list[str]:
        """Return a copy of the feature names."""
        return self._feature_names[:]

    def assemble_output(
        self,
        value_list: Iterable[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Transform output of all transform_single operations to matrix.

        Parameters
        ----------
        value_list: Iterable[npt.NDArray[np.float64]]
            List of numpy arrays with calculated descriptor values of each molecule.

        Returns
        -------
        npt.NDArray[np.float64]
            Matrix with descriptor values of each molecule.
        """
        return np.vstack(list(value_list))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all parameters defined during object initialization.

        Parameters
        ----------
        deep: bool, default=True
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters relevant to initialize the object with same properties.
        """
        params = super().get_params(deep)
        if deep:
            if self._standardizer is not None:
                params["standardizer"] = clone(self._standardizer)
            else:
                params["standardizer"] = None
        else:
            params["standardizer"] = self._standardizer
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: Any
            Dictionary with parameter names and corresponding values.

        Returns
        -------
        Self
            Object with updated parameters.
        """
        parameter_copy = dict(parameters)
        standardizer = parameter_copy.pop("standardizer", None)
        if standardizer is not None:
            self._standardizer = standardizer
        super().set_params(**parameter_copy)
        return self

    def fit_to_result(self, values: list[npt.NDArray[np.float64]]) -> Self:
        """Fit object to data.

        Parameters
        ----------
        values: list[RDKitMol]
            List of RDKit molecules to which the Pipeline element is fitted.

        Returns
        -------
        Self
            Fitted MolToDescriptorPipelineElement.
        """
        value_matrix = np.vstack(list(values))
        if self._standardizer is not None:
            self._standardizer.fit(value_matrix, None)
        return self

    def _normalize_matrix(
        self, value_matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Normalize matrix with descriptor values.

        Parameters
        ----------
        value_matrix: npt.NDArray[np.float64]
            Matrix with descriptor values of molecules.

        Returns
        -------
        npt.NDArray[np.float64]
            Normalized matrix with descriptor values of molecules.
        """
        if self._standardizer is not None:
            return self._standardizer.transform(value_matrix)
        return value_matrix

    def transform(self, values: list[RDKitMol]) -> npt.NDArray[np.float64]:
        """Transform the list of molecules to sparse matrix.

        Parameters
        ----------
        values: list[RDKitMol]
            List of RDKit molecules for which the descriptor vectors are calculated.

        Returns
        -------
        npt.NDArray[np.float64]
            Matrix with descriptor values of molecules.
        """
        descriptor_matrix: npt.NDArray[np.float64] = super().transform(values)
        return descriptor_matrix

    def finalize_single(
        self, value: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Finalize single value. Here: standardize vector.

        Parameters
        ----------
        value: Any
            Single value to be finalized.

        Returns
        -------
        Any
            Finalized value.
        """
        if self._standardizer is not None:
            standadized_value = self._standardizer.transform(value.reshape(1, -1))
            return standadized_value.reshape(-1)
        return value

    @abc.abstractmethod
    def pretransform_single(
        self, value: RDKitMol
    ) -> Union[npt.NDArray[np.float64], InvalidInstance]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol
            Molecule to be transformed to descriptor vector.

        Returns
        -------
        npt.NDArray[np.float64]
            Vector with descriptor values of molecule.
        """
