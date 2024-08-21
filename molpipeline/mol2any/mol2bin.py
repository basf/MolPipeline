"""Converter element for molecules to binary string representation."""

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement


class MolToBinary(MolToAnyPipelineElement):
    """PipelineElement to transform a molecule to a binary."""

    def pretransform_single(self, value: Chem.Mol) -> str:
        """Transform a molecule to a binary string.

        Parameters
        ----------
        value: Chem.Mol
            Molecule to be transformed to binary string representation.

        Returns
        -------
        str
            Binary representation of molecule.
        """
        return value.ToBinary()
