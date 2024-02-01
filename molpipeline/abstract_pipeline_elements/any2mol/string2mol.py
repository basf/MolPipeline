"""Abstract classes for creating rdkit molecules from string representations."""

from __future__ import annotations

import abc

from molpipeline.abstract_pipeline_elements.core import AnyToMolPipelineElement
from molpipeline.utils.molpipeline_types import OptionalMol


class StringToMolPipelineElement(AnyToMolPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""

    _input_type = "str"
    _output_type = "RDKitMol"

    def transform(self, values: list[str]) -> list[OptionalMol]:
        """Transform the list of molecules to sparse matrix.

        Parameters
        ----------
        values: list[str]
            List of string representations of molecules which are transformed to RDKit molecules.

        Returns
        -------
        list[OptionalMol]
            List of RDKit molecules. If a string representation could not be transformed to a molecule, None is returned.
        """
        return super().transform(values)

    @abc.abstractmethod
    def pretransform_single(self, value: str) -> OptionalMol:
        """Transform mol to a string.

        Parameters
        ----------
        value: str
            Representation transformed to a RDKit molecule.

        Returns
        -------
        OptionalMol
            RDKit molecule if representation was valid, else InvalidInstance.
        """
