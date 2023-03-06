from typing import Iterable

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    MolToFingerprintPipelineElement
)


class MolToConcatenatedVector(MolToAnyPipelineElement):
    def __init__(
            self,
            pipeline_element_list: list[MolToAnyPipelineElement],
            name: str = "MolToConcatenatedVector",
            n_jobs: int = 1
    ) -> None:
        """

        Parameters
        ----------
        pipeline_element_list: list[MolToAnyPipelineElement]
            List of Pipeline Elements of which the output is concatenated.
        name: str
            name of pipeline.
        n_jobs: int:
            Number of cores used.
        """
        super().__init__(name=name, n_jobs=n_jobs)
        self._pipeline_element_list = pipeline_element_list

    @staticmethod
    def assemble_output(
        value_list: Iterable[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Transform output of all transform_single operations to matrix."""
        return np.vstack(list(value_list))

    def transform(self, value_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix."""
        return self.assemble_output(super().transform(value_list))

    def fit(self, value_list: list[Chem.Mol]) -> None:
        """Fit each pipeline element."""
        for pipeline_element in self._pipeline_element_list:
            pipeline_element.fit(value_list)

    def _transform_single(self, value: Chem.Mol) -> npt.NDArray[np.float_]:
        """Get output of each element and concatenate for output."""
        final_vector = []
        for pipeline_element in self._pipeline_element_list:
            if isinstance(pipeline_element, MolToFingerprintPipelineElement):
                bit_dict = pipeline_element.transform_single(value)
                vector = np.zeros(pipeline_element.n_bits)
                vector[list(bit_dict.keys())] = np.array(list(bit_dict.values()))
            else:
                vector = pipeline_element.transform_single(value)
            final_vector.append(vector)
        return np.hstack(final_vector)
