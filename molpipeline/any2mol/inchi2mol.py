"""Classes ment to transform given inchi to a RDKit molecule."""

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    SimpleStringToMolElement,
)
from molpipeline.abstract_pipeline_elements.core import InvalidInstance, RDKitMol


class InchiToMol(SimpleStringToMolElement):
    """Transforms Inchi to RDKit Mol objects."""

    def string_to_mol(self, value: str) -> RDKitMol | None:
        """Transform Inchi string to molecule.

        Parameters
        ----------
        value: str
            Inchi string.

        Returns
        -------
        RDKitMol | None
            Rdkit molecule if valid Inchi, else InvalidInstance.

        """
        return Chem.MolFromInchi(value)
