"""Abstract classes for creating rdkit molecules from string representations."""
from __future__ import annotations
from typing import Any

import abc

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import AnyToMolPipelineElement
from molpipeline.utils.molpipe_types import OptionalMol


class StringToMolPipelineElement(AnyToMolPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""

    _input_type = str
    _output_type = Chem.Mol

    def transform(self, value_list: list[str]) -> list[OptionalMol]:
        """Transform the list of molecules to sparse matrix."""
        return super().transform(value_list)

    @abc.abstractmethod
    def _transform_single(self, value: str) -> Chem.Mol:
        """Transform mol to a string.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        str
        """
