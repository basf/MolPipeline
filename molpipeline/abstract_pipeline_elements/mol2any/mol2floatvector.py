"""Abstract classes for transforming rdkit molecules to float vectors."""
from __future__ import annotations

import abc
from typing import Iterable

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement


class MolToDescriptorPipelineElement(MolToAnyPipelineElement):
    """PipelineElement which generates a matrix from descriptor-vectors of each molecule."""

    @staticmethod
    def assemble_output(
        value_list: Iterable[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Transform output of all transform_single operations to matrix."""
        return np.vstack(list(value_list))

    def transform(self, value_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix."""
        return self.assemble_output(super().transform(value_list))

    @abc.abstractmethod
    def transform_single(self, value: Chem.Mol) -> npt.NDArray[np.float_]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        npt.NDArray[np.float_]
            Vector with descriptor values of molecule.
        """
