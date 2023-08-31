"""Classes for creating arrays from multiple concatenated descriptors or fingerprints."""
from __future__ import annotations
from typing import Any, Iterable, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self
import numpy as np
import numpy.typing as npt

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToAnyPipelineElement,
)
from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    MolToFingerprintPipelineElement,
)
from molpipeline.utils.json_operations import pipeline_element_from_json
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToConcatenatedVector(MolToAnyPipelineElement):
    """Creates a concatenated descriptor vectored from multiple MolToAny PipelineElements."""

    _element_list: list[MolToAnyPipelineElement]

    # pylint: disable=R0913
    def __init__(
        self,
        element_list: list[MolToAnyPipelineElement],
        name: str = "MolToConcatenatedVector",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
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
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        for element in self._element_list:
            element.n_jobs = self.n_jobs
        self._requires_fitting = any(
            element._requires_fitting for element in element_list
        )

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
        parameter_copy = dict(parameters)
        element_list = parameter_copy.pop("element_list", None)
        if element_list is not None:
            self._element_list = element_list

        super().set_params(parameter_copy)
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
            Fitted pipeline element.
        """
        for pipeline_element in self._element_list:
            pipeline_element.fit(value_list)
        return self

    def pretransform_single(
        self, value: RDKitMol
    ) -> Union[list[Union[npt.NDArray[np.float_], dict[int, int]]], InvalidInstance]:
        """Get pretransform of each element and concatenate for output.

        Parameters
        ----------
        value: RDKitMol
            Molecule to be transformed.

        Returns
        -------
        Union[list[Union[npt.NDArray[np.float_], dict[int, int]]], InvalidInstance]
            List of pretransformed values of each pipeline element.
            If any element returns None, InvalidInstance is returned.
        """
        final_vector = []
        error_message = ""
        for pipeline_element in self._element_list:
            vector = pipeline_element.pretransform_single(value)
            if vector is None:
                error_message += f"{pipeline_element.name} returned None. "
                break

            final_vector.append(vector)
        else:  # no break
            return final_vector
        return InvalidInstance(self.uuid, error_message)

    def finalize_single(self, value: Any) -> Any:
        """Finalize the output of transform_single.

        Parameters
        ----------
        value: Any
            Output of transform_single.

        Returns
        -------
        Any
            Finalized output.
        """
        final_vector_list = []
        for element, sub_value in zip(self._element_list, value):
            final_value = element.finalize_single(sub_value)
            if isinstance(element, MolToFingerprintPipelineElement):
                vector = np.zeros(element.n_bits)
                vector[list(final_value.keys())] = np.array(list(final_value.values()))
                final_value = vector
            if not isinstance(final_value, np.ndarray):
                final_value = np.array(final_value)
            final_vector_list.append(final_value)
        return np.hstack(final_vector_list)

    def fit_to_result(self, value_list: Any) -> Self:
        """Fit the pipeline element to the result of transform_single.

        Parameters
        ----------
        value_list: Any
            Output of transform_single.

        Returns
        -------
        Self
            Fitted pipeline element.
        """
        for element, value in zip(self._element_list, zip(*value_list)):
            element.fit_to_result(value)
        return self
