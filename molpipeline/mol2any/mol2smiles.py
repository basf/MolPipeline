"""Classes for transforming rdkit molecules to any type of output."""

from __future__ import annotations

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.mol2any.mol2string import (
    MolToStringPipelineElement as _MolToStringPipelineElement,
)


class MolToSmiles(_MolToStringPipelineElement):
    """PipelineElement to transform a molecule to a SMILES string."""

    def pretransform_single(self, value: Chem.Mol) -> str:
        """Transform a molecule to a SMILES string.

        Parameters
        ----------
        value: Chem.Mol
            Molecule to be transformed to SMILES string.

        Returns
        -------
        str
            SMILES string of molecule.
        """
        return str(Chem.MolToSmiles(value))
