"""Classes ment to transform given inchi to a RDKit molecule."""

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    SimpleStringToMolElement,
)
from molpipeline.abstract_pipeline_elements.core import InvalidInstance, OptionalMol


class InchiToMol(SimpleStringToMolElement):
    """Transforms Inchi to RDKit Mol objects."""

    def string_to_mol(self, value: str) -> OptionalMol:
        """Transform Inchi string to molecule.

        Parameters
        ----------
        value: str
            Inchi string.

        Returns
        -------
        OptionalMol
            Rdkit molecule if valid Inchi, else InvalidInstance.

        """
        mol = Chem.MolFromInchi(value)
        if mol is None:
            return InvalidInstance(
                self.uuid,
                f"Invalid Inchi string: {value}",
                self.name,
            )
        return mol
