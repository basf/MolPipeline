"""Class for transforming molecules to SMILES representations."""

from __future__ import annotations

import abc

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement


class MolToStringPipelineElement(MolToAnyPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""

    _output_type = "str"

    def transform(self, values: list[Chem.Mol]) -> list[str]:
        """Transform the list of molecules to sparse matrix.

        Parameters
        ----------
        values: list[Chem.Mol]
            List of RDKit molecules which are transformed to a string representation.

        Returns
        -------
        list[str]
            List of string representations of the molecules.
        """
        string_list: list[str] = super().transform(values)
        return string_list

    @abc.abstractmethod
    def pretransform_single(self, value: Chem.Mol) -> str:
        """Transform mol to a string.

        Parameters
        ----------
        value: Chem.Mol
            Molecule to be transformed to SMILES representation.
        Returns
        -------
        str
            SMILES representation of molecule.
        """
