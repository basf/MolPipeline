"""Classes ment to transform given inchi to a RDKit molecule."""

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    SimpleStringToMolElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class InchiToMol(SimpleStringToMolElement):
    """Transforms Inchi to RDKit Mol objects."""

    def string_to_mol(self, value: str) -> RDKitMol:
        """Transform Inchi string to molecule.

        Parameters
        ----------
        value: str
            Inchi string.

        Returns
        -------
        RDKitMol
            Rdkit molecule if valid Inchi, else None.
        """
        return Chem.MolFromInchi(value)
