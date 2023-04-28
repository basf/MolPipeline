"""Abstract classes for transforming rdkit molecules to float vectors."""
# pylint: disable=too-many-arguments

from __future__ import annotations

import abc
from typing import Any, Iterable, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]

from molpipeline.abstract_pipeline_elements.core import (
    MolToAnyPipelineElement,
    NoneHandlingOptions,
)
from molpipeline.utils.multi_proc import wrap_parallelizable_task


class MolToDescriptorPipelineElement(MolToAnyPipelineElement):
    """PipelineElement which generates a matrix from descriptor-vectors of each molecule."""

    _normalize: bool
    _mean: Optional[npt.NDArray[np.float_]]
    _std: Optional[npt.NDArray[np.float_]]

    def __init__(
        self,
        normalize: bool = True,
        name: str = "MolToDescriptorPipelineElement",
        n_jobs: int = 1,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: float = np.nan,
    ) -> None:
        """Initialize MolToDescriptorPipelineElement.

        Parameters
        ----------
        normalize: bool
        name: str
        n_jobs: int
        """
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )
        self._normalize = normalize
        self._mean = None
        self._std = None

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> Self:
        """Create object from json dict.

        Parameters
        ----------
        json_dict: dict[str, Any]
            Dictionary containing all parameters relevant to initialize the object.

        Returns
        -------
        Self
            Object created from json_dict.
        """
        json_dict_copy = dict(json_dict)  # copy, because the dict is modified
        additional_attributes = json_dict_copy.pop("additional_attributes", None)
        if additional_attributes:
            additional_attributes = {
                "mean": np.array(additional_attributes["mean"]),
                "std": np.array(additional_attributes["std"]),
            }
        json_dict_copy["additional_attributes"] = additional_attributes
        return super().from_json(json_dict_copy)

    @property
    @abc.abstractmethod
    def n_features(self) -> int:
        """Return the number of features."""

    def assemble_output(
        self,
        value_list: Iterable[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Transform output of all transform_single operations to matrix.

        Parameters
        ----------
        value_list: Iterable[npt.NDArray[np.float_]]
            List of numpy arrays with calculated descriptor values of each molecule.

        Returns
        -------
        npt.NDArray[np.float_]
            Matrix with descriptor values of each molecule.
        """
        return np.vstack(list(value_list))

    def get_parameters(self) -> dict[str, Any]:
        """Return all parameters defined during object initialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters relevant to initialize the object with same properties.
        """
        params = super().get_parameters()
        params["normalize"] = self._normalize
        return params

    def set_parameters(self, parameters: dict[str, Any]) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: dict[str, Any]
            Dictionary with parameter names and corresponding values.

        Returns
        -------
        Self
            Object with updated parameters.
        """
        super().set_parameters(parameters)
        if "normalize" in parameters:
            self._normalize = parameters["normalize"]
        return self

    @property
    def additional_attributes(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        attribute_dict = super().additional_attributes
        if self._mean is not None:
            attribute_dict["mean"] = self._mean
        if self._std is not None:
            attribute_dict["std"] = self._std
        return attribute_dict

    def fit(self, value_list: list[RDKitMol]) -> Self:
        """Fit object to data.

        Parameters
        ----------
        value_list: list[RDKitMol]
            List of RDKit molecules to which the Pipeline element is fitted.

        Returns
        -------
        Self
            Fitted MolToDescriptorPipelineElement.
        """
        self.fit_transform(value_list)
        return self

    def fit_transform(self, value_list: list[RDKitMol]) -> npt.NDArray[np.float_]:
        """Fit object to data and return the accordingly transformed data.

        Parameters
        ----------
        value_list: list[RDKitMol]
            List of RDKit molecules to which the Pipeline element is fitted and for which the descriptor vectors
            are calculated subsequently.

        Returns
        -------
        npt.NDArray[np.float_]
            Matrix with descriptor values of molecules.
        """
        array_list = wrap_parallelizable_task(
            self._transform_single, value_list, self.n_jobs
        )
        array_list = super()._catch_nones(array_list)
        value_matrix = self.assemble_output(array_list)
        self._mean = np.nanmean(value_matrix, axis=0)
        _std: npt.NDArray[np.float_] = np.nanstd(value_matrix, axis=0)
        _std[np.where(_std == 0)] = 1.0  # avoid division by zero
        self._std = _std
        normalized_matrix = self._normalize_matrix(value_matrix)
        if self.none_handling == "fill_dummy":
            final_matrix: npt.NDArray[np.float_]
            final_matrix = self.none_collector.fill_with_dummy(normalized_matrix)
            return final_matrix
        return normalized_matrix

    def _normalize_matrix(
        self, value_matrix: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        if self._normalize:
            if self._mean is None or self._std is None:
                raise ValueError("Model appears not to be fitted.")
            scaled_matrix = (value_matrix - self._mean) / self._std
            return scaled_matrix
        return value_matrix

    def transform(self, value_list: list[RDKitMol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix.

        Parameters
        ----------
        value_list: list[RDKitMol]
            List of RDKit molecules for which the descriptor vectors are calculated.

        Returns
        -------
        npt.NDArray[np.float_]
            Matrix with descriptor values of molecules.
        """
        descriptor_matrix: npt.NDArray[np.float_] = super().transform(value_list)
        return descriptor_matrix

    def transform_single(self, value: RDKitMol) -> Optional[npt.NDArray[np.float_]]:
        """Normalize _transform_single if required.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule for which the descriptor vector is calculated.

        Returns
        -------
        Optional[npt.NDArray[np.float_]]
            Vector with descriptor values of molecule, or None if the descriptor could not be calculated.
        """
        feature_vector = self._transform_single(value)
        if feature_vector is None:
            return None
        if self._normalize:
            return self._normalize_matrix(feature_vector)
        return feature_vector

    @abc.abstractmethod
    def _transform_single(self, value: RDKitMol) -> Optional[npt.NDArray[np.float_]]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        npt.NDArray[np.float_]
            Vector with descriptor values of molecule.
        """
