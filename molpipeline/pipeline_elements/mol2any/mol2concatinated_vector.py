"""Classes for creating arrays from multiple concatenated descriptors or fingerprints."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToAnyPipelineElement,
)
from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    MolToFingerprintPipelineElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToConcatenatedVector(MolToAnyPipelineElement):
    """Creates a concatenated descriptor vectored from multiple MolToAny PipelineElements."""

    _element_list: list[tuple[str, MolToAnyPipelineElement]]

    # pylint: disable=R0913
    def __init__(
        self,
        element_list: list[tuple[str, MolToAnyPipelineElement]],
        name: str = "MolToConcatenatedVector",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
        **kwargs: Any,
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
        output_types = set()
        for _, element in self._element_list:
            element.n_jobs = self.n_jobs
            output_types.add(element.output_type)
        if len(output_types) == 1:
            self._output_type = output_types.pop()
        else:
            self._output_type = "mixed"
        self._requires_fitting = any(
            element[1]._requires_fitting for element in element_list
        )
        self.set_params(kwargs)

    @property
    def element_list(self) -> list[tuple[str, MolToAnyPipelineElement]]:
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
                (str(name), clone(ele)) for name, ele in self.element_list
            ]
        else:
            parameters["element_list"] = self.element_list
        for name, element in self.element_list:
            for key, value in element.get_params(deep=deep).items():
                parameters[f"{name}__{key}"] = value

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
        step_params: dict[str, dict[str, Any]] = {}
        step_dict = dict(self._element_list)
        to_delete_list = []
        for parm, value in parameters.items():
            if "__" not in parm:
                continue
            param_split = parm.split("__")
            param_header = param_split[0]
            # Check if parameter addresses an element
            if param_header not in step_dict:
                continue
            param_tail = "__".join(param_split[1:])
            if param_header not in step_params:
                step_params[param_header] = {}
            step_params[param_header][param_tail] = value
            to_delete_list.append(parm)
        for to_delete in to_delete_list:
            _ = parameter_copy.pop(to_delete, None)
        for step, params in step_params.items():
            step_dict[step].set_params(params)
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

    def transform(self, values: list[RDKitMol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix.

        Parameters
        ----------
        values: list[RDKitMol]
            List of molecules to transform.

        Returns
        -------
        npt.NDArray[np.float_]
            Matrix of shape (n_molecules, n_features) with concatenated features specified during init.
        """
        output: npt.NDArray[np.float_] = super().transform(values)
        return output

    def fit(
        self,
        values: list[RDKitMol],
        labels: Any = None,  # pylint: disable=unused-argument
    ) -> Self:
        """Fit each pipeline element.

        Parameters
        ----------
        values: list[RDKitMol]
            List of molecules used to fit the pipeline elements creating the concatenated vector.
        labels: Any
            Labels for the molecules. Not used.

        Returns
        -------
        Self
            Fitted pipeline element.
        """
        for pipeline_element in self._element_list:
            pipeline_element[1].fit(values)
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
        for name, pipeline_element in self._element_list:
            vector = pipeline_element.pretransform_single(value)
            if isinstance(vector, InvalidInstance):
                error_message += f"{self.name}__{name} returned an InvalidInstance."
                break

            final_vector.append(vector)
        else:  # no break
            return final_vector
        return InvalidInstance(self.uuid, error_message, self.name)

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
        for (_, element), sub_value in zip(self._element_list, value):
            final_value = element.finalize_single(sub_value)
            if isinstance(element, MolToFingerprintPipelineElement) and isinstance(
                final_value, dict
            ):
                vector = np.zeros(element.n_bits)
                vector[list(final_value.keys())] = np.array(list(final_value.values()))
                final_value = vector
            if not isinstance(final_value, np.ndarray):
                final_value = np.array(final_value)
            final_vector_list.append(final_value)
        return np.hstack(final_vector_list)

    def fit_to_result(self, values: Any) -> Self:
        """Fit the pipeline element to the result of transform_single.

        Parameters
        ----------
        values: Any
            Output of transform_single.

        Returns
        -------
        Self
            Fitted pipeline element.
        """
        for element, value in zip(self._element_list, zip(*values)):
            element[1].fit_to_result(value)
        return self
