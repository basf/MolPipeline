"""Classes for creating arrays from multiple concatenated descriptors or fingerprints."""
from __future__ import annotations
from typing import Any, Iterable

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

    @property
    def component_list(self) -> list[MolToAnyPipelineElement]:
        """Get component_list."""
        return self._component_list[:]

    @property
    def params(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        params = super().params
        params.update(
            {
                "component_list": [
                    component.copy() for component in self.component_list
                ],
                "name": self.name,
                "n_jobs": self.n_jobs,
            }
        )
        return params

    def copy(self) -> MolToConcatenatedVector:
        """Create a copy of the object."""
        return MolToConcatenatedVector(**self.params)

    def assemble_output(
        self,
        value_list: Iterable[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Transform output of all transform_single operations to matrix."""
        return np.vstack(list(value_list))

    def transform(self, value_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix."""
        output: npt.NDArray[np.float_] = super().transform(value_list)
        return output

    def fit(self, value_list: list[Chem.Mol]) -> None:
        """Fit each pipeline element."""
        for pipeline_element in self._component_list:
            pipeline_element.fit(value_list)

    def _transform_single(self, value: Chem.Mol) -> npt.NDArray[np.float_]:
        """Get output of each element and concatenate for output."""
        final_vector = []
        for pipeline_element in self._component_list:
            if isinstance(pipeline_element, MolToFingerprintPipelineElement):
                bit_dict = pipeline_element.transform_single(value)
                vector = np.zeros(pipeline_element.n_bits)
                vector[list(bit_dict.keys())] = np.array(list(bit_dict.values()))
            else:
                vector = pipeline_element.transform_single(value)
            final_vector.append(vector)
        return np.hstack(final_vector)
