"""Abstract classes for transforming rdkit molecules to float vectors."""
# pylint: disable=too-many-arguments

from __future__ import annotations

import abc
from typing import Any, Iterable, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy
import numpy as np
import numpy.typing as npt
from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]

from molpipeline.abstract_pipeline_elements.core import (
    MolToAnyPipelineElement,
    InvalidInstance,
)


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
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToDescriptorPipelineElement.

        Parameters
        ----------
        normalize: bool
        name: str
        n_jobs: int
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self._normalize = normalize
        if self._normalize:
            self._requires_fitting = True
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
        additional_attributes = json_dict_copy.pop("additional_attributes", {})
        if additional_attributes:
            additional_attributes = {
                "_mean": np.array(additional_attributes["_mean"]),
                "_std": np.array(additional_attributes["_std"]),
            }
        json_dict_copy["additional_attributes"] = additional_attributes
        return super().from_json(json_dict_copy)

    @property
    @abc.abstractmethod
    def n_features(self) -> int:
        """Return the number of features."""

    @property
    def normalize(self) -> bool:
        """Return whether the output is normalized."""
        return self._normalize

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

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all parameters defined during object initialization.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters relevant to initialize the object with same properties.
        """
        params = super().get_params(deep)
        if deep:
            params["normalize"] = copy.copy(self._normalize)
        else:
            params["normalize"] = self._normalize
        return params

    def set_params(self, parameters: dict[str, Any]) -> Self:
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
        parameter_copy = dict(parameters)
        normalize = parameter_copy.pop("normalize", None)
        if normalize is not None:
            self._normalize = parameters["normalize"]
        super().set_params(parameter_copy)
        return self

    @property
    def additional_attributes(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        attribute_dict = super().additional_attributes
        if self._mean is not None:
            attribute_dict["_mean"] = self._mean
        if self._std is not None:
            attribute_dict["_std"] = self._std
        return attribute_dict

    def fit_to_result(self, value_list: list[npt.NDArray[np.float_]]) -> Self:
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
        value_matrix = np.vstack(list(value_list))
        self._mean = np.nanmean(value_matrix, axis=0)
        _std: npt.NDArray[np.float_] = np.nanstd(value_matrix, axis=0)
        _std[np.where(_std == 0)] = 1.0  # avoid division by zero
        self._std = _std
        return self

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

    def finalize_single(self, value: Any) -> Any:
        if self.normalize:
            return self._normalize_matrix(value)
        return value

    @abc.abstractmethod
    def pretransform_single(
        self, value: RDKitMol
    ) -> Union[npt.NDArray[np.float_], InvalidInstance]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        npt.NDArray[np.float_]
            Vector with descriptor values of molecule.
        """
