"""Classes ment to transform given input to a RDKit molecule."""

from __future__ import annotations

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    SimpleStringToMolElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class SmilesToMol(SimpleStringToMolElement):
    """Transforms Smiles to RDKit Mol objects."""

    def string_to_mol(self, value: str) -> RDKitMol:
        """Transform Smiles string to molecule.

        Parameters
        ----------
        value: str
            SMILES string.

        Returns
        -------
        RDKitMol
            Rdkit molecule if valid SMILES, else None.
        """
        return Chem.MolFromSmiles(value)
