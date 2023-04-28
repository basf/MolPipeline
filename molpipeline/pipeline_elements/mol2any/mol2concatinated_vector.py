"""Classes for creating arrays from multiple concatenated descriptors or fingerprints."""
from __future__ import annotations
from typing import Any, Iterable, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self
import numpy as np
import numpy.typing as npt
from rdkit import Chem

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

    _component_list: list[MolToAnyPipelineElement]

    # pylint: disable=R0913
    def __init__(
        self,
        component_list: list[MolToAnyPipelineElement],
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "MolToConcatenatedVector",
        n_jobs: int = 1,
    ) -> None:
        """Initialize MolToConcatenatedVector.

        Parameters
        ----------
        component_list: list[MolToAnyPipelineElement]
            List of Pipeline Elements of which the output is concatenated.
        name: str
            name of pipeline.
        n_jobs: int:
            Number of cores used.
        """
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )
        self._component_list = component_list
        for component in self._component_list:
            component.n_jobs = self.n_jobs

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> Self:
        """Create object from json representation."""
        params = dict(json_dict)  # copy, because the dict is modified
        component_json_list = params.pop("component_list")
        component_list = [
            pipeline_element_from_json(component) for component in component_json_list
        ]
        params["component_list"] = component_list
        return super().from_json(params)

    @property
    def component_list(self) -> list[MolToAnyPipelineElement]:
        """Get component_list."""
        return self._component_list[:]

    @property
    def parameters(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        params = super().parameters
        params.update(
            {
                "component_list": [
                    component.copy() for component in self.component_list
                ],
            }
        )
        return params

    def assemble_output(
        self,
        value_list: Iterable[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Transform output of all transform_single operations to matrix."""
        return np.vstack(list(value_list))

    def to_json(self) -> dict[str, Any]:
        """Return json representation of the object."""
        json_dict = super().to_json()
        json_dict["component_list"] = [
            component.to_json() for component in self.component_list
        ]
        return json_dict

    def transform(self, value_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix."""
        output: npt.NDArray[np.float_] = super().transform(value_list)
        return output

    def fit(self, value_list: list[Chem.Mol]) -> None:
        """Fit each pipeline element."""
        for pipeline_element in self._component_list:
            pipeline_element.fit(value_list)

    def _transform_single(self, value: Chem.Mol) -> Optional[npt.NDArray[np.float_]]:
        """Get output of each element and concatenate for output."""
        final_vector = []
        for pipeline_element in self._component_list:
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
