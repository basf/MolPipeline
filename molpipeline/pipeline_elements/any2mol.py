"""Classes ment to transform given input to a RDKit molecule."""

from rdkit import Chem

from molpipeline.pipeline_elements.abstract_pipeline_elements import AnyToMolPipelineElement
from molpipeline.utils.molpipe_types import OptionalMol


class SmilesToMolPipelineElement(AnyToMolPipelineElement):
    """Transforms Smiles to RDKit Mol objects."""

    def __init__(self, identifier: str = "smiles", name: str = "smiles2mol") -> None:
        """Initialize SmilesToMolPipelineElement.

        Parameters
        ----------
        identifier: str
            Method of assigning identifiers to molecules. At the moment the Smiles is used.
        name: str
            Name of PipelineElement
        """
        self.identifier = identifier
        super().__init__(name)

    def transform(self, value_list: list[str]) -> list[OptionalMol]:
        """Transform a list of SMILES to a list of molecules."""
        return super().transform(value_list)

    def transform_single(self, value: str) -> OptionalMol:
        """Transform Smiles string to molecule.

        Parameters
        ----------
        value: str
            SMILES string.

        Returns
        -------
        OptionalMol
            Rdkit molecule if valid SMILES, else None.
        """
        mol: Chem.Mol = Chem.MolFromSmiles(value)
        if not mol:
            return None
        if self.identifier == "smiles":
            mol.SetProp("identifier", value)
        return mol


class SDFToMolPipelineElement(AnyToMolPipelineElement):
    """PipelineElement transforming a list of SDF strings to mol_objects."""

    identifier: str
    mol_counter: int

    def __init__(self, identifier: str = "enumerate", name: str = "SDF2Mol") -> None:
        """Initialize SDFToMolPipelineElement.

        Parameters
        ----------
        identifier: str
            Method of assigning identifiers to molecules. At the moment molecules are counted.
        name: str
            Name of PipelineElement
        """
        super().__init__(name)
        self.identifier = identifier
        self.mol_counter = 0

    def finish(self) -> None:
        """Reset the mol counter which assigns identifiers."""
        self.mol_counter = 0

    def transform(self, value_list: list[str]) -> list[OptionalMol]:
        """Transform a list of SDF-strings to a list of rdkit molecules."""
        molecule_list = super().transform(value_list)
        self.finish()
        return molecule_list

    def transform_single(self, value: str) -> OptionalMol:
        """Transform an SDF-strings to a rdkit molecule."""
        mol = Chem.MolFromMolBlock(value)
        if self.identifier == "smiles":
            mol.SetProp("identifier", self.mol_counter)
        self.mol_counter += 1
        return mol
