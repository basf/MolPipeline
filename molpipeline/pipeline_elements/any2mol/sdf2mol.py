"""Class for Transforming SDF-strings to rdkit molecules."""

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    StringToMolPipelineElement as _StringToMolPipelineElement,
)
from molpipeline.utils.molpipe_types import OptionalMol


class SDFToMolPipelineElement(_StringToMolPipelineElement):
    """PipelineElement transforming a list of SDF strings to mol_objects."""

    identifier: str
    mol_counter: int

    def __init__(
        self, identifier: str = "enumerate", name: str = "SDF2Mol", n_jobs: int = 1
    ) -> None:
        """Initialize SDFToMolPipelineElement.

        Parameters
        ----------
        identifier: str
            Method of assigning identifiers to molecules. At the moment molecules are counted.
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores used for processing.
        """
        super().__init__(name=name, n_jobs=n_jobs)
        self.identifier = identifier
        self.mol_counter = 0

    def finish(self) -> None:
        """Reset the mol counter which assigns identifiers."""
        self.mol_counter = 0

    def _transform_single(self, value: str) -> OptionalMol:
        """Transform an SDF-strings to a rdkit molecule."""
        mol = Chem.MolFromMolBlock(value)
        if self.identifier == "smiles":
            mol.SetProp("identifier", self.mol_counter)
        self.mol_counter += 1
        return mol