"""Classes for creating arrays from multiple concatenated descriptors or fingerprints."""
from __future__ import annotations
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
from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    MolToFingerprintPipelineElement,
)
from molpipeline.utils.json_operations import pipeline_element_from_json


class MolToConcatenatedVector(MolToAnyPipelineElement):
    """Creates a concatenated descriptor vectored from multiple MolToAny PipelineElements."""

    _element_list: list[MolToAnyPipelineElement]

    # pylint: disable=R0913
    def __init__(
        self,
        element_list: list[MolToAnyPipelineElement],
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "MolToConcatenatedVector",
        n_jobs: int = 1,
    ) -> None:
        """Initialize MolToConcatenatedVector.

        Parameters
        ----------
        element_list: list[MolToAnyPipelineElement]
            List of Pipeline Elements of which the output is concatenated.
        name: str
            name of pipeline.
        n_jobs: int:
            Number of cores used.
        """
        self._element_list = element_list
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )
        for element in self._element_list:
            element.n_jobs = self.n_jobs

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> Self:
        """Create object from json representation.

        Parameters
        ----------
        json_dict: dict[str, Any]
            Json representation of object.
        Returns
        -------
        Self
            Mol2ConcatenatedVector pipeline element specified by json_dict.
        """
        params = dict(json_dict)  # copy, because the dict is modified
        pipeline_element_json_list = params.pop("element_list")
        pipeline_element_list = [
            pipeline_element_from_json(element)
            for element in pipeline_element_json_list
        ]
        params["element_list"] = pipeline_element_list
        return super().from_json(params)

    @property
    def element_list(self) -> list[MolToAnyPipelineElement]:
        """Get pipeline elements."""
        return self._element_list

    @property
    def none_handling(self) -> NoneHandlingOptions:
        """Get none_handling."""
        return self._none_handling

    @none_handling.setter
    def none_handling(self, none_handling: NoneHandlingOptions) -> None:
        """Set none_handling.

        Parameters
        ----------
        none_handling: NoneHandlingOptions
            How to handle None values.

        Returns
        -------
        None
        """
        self._none_handling = none_handling
        for element in self._element_list:
            element.none_handling = none_handling

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all parameters defining the object.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Parameters defining the object.
        """
        parameters = super().get_params(deep)
        if deep:
            parameters["element_list"] = [
                element.copy() for element in self.element_list
            ]
        else:
            parameters["element_list"] = self.element_list
        return parameters

    def set_params(self, parameters: dict[str, Any]) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: dict[str, Any]
            Parameters to set.
        Returns
        -------
        Self
            Mol2ConcatenatedVector object with updated parameters.
        """
        super().set_params(parameters)
        if "element_list" in parameters:
            self._element_list = parameters["element_list"]
        for element in self._element_list:
            element.n_jobs = self.n_jobs
            element.none_handling = self.none_handling
        return self

    def assemble_output(
        self,
        value_list: Iterable[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Transform output of all transform_single operations to matrix.

        Parameters
        ----------
        value_list: Iterable[npt.NDArray[np.float_]]
            List of molecular descriptors or fingerprints which are concatenated to a single matrix.
        Returns
        -------
        npt.NDArray[np.float_]
            Matrix of shape (n_molecules, n_features) with concatenated features specified during init.
        """
        return np.vstack(list(value_list))

    def to_json(self) -> dict[str, Any]:
        """Return json representation of the object.

        Returns
        -------
        dict[str, Any]
            Json representation of the object.
        """
        json_dict = super().to_json()
        json_dict["element_list"] = [element.to_json() for element in self.element_list]
        return json_dict

    def transform(self, value_list: list[RDKitMol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix.

        Parameters
        ----------
        value_list: list[RDKitMol]
            List of molecules to transform.
        Returns
        -------
        npt.NDArray[np.float_]
            Matrix of shape (n_molecules, n_features) with concatenated features specified during init.
        """
        output: npt.NDArray[np.float_] = super().transform(value_list)
        return output

    def fit(self, value_list: list[RDKitMol]) -> Self:
        """Fit each pipeline element.

        Parameters
        ----------
        value_list: list[RDKitMol]
            List of molecules used to fit the pipeline elements creating the concatenated vector.
        Returns
        -------
        Self
            Fitted pipelineelement.
        """
        for pipeline_element in self._element_list:
            pipeline_element.fit(value_list)
        return self

    def _transform_single(self, value: RDKitMol) -> Optional[npt.NDArray[np.float_]]:
        """Get output of each element and concatenate for output."""
        final_vector = []
        for pipeline_element in self._element_list:
            if isinstance(pipeline_element, MolToFingerprintPipelineElement):
                bit_dict = pipeline_element.transform_single(value)
                vector = np.zeros(pipeline_element.n_bits)
                vector[list(bit_dict.keys())] = np.array(list(bit_dict.values()))
            else:
                vector = pipeline_element.transform_single(value)

            if vector is None:
                break

            final_vector.append(vector)
        else:  # no break
            return np.hstack(final_vector)
        return None
