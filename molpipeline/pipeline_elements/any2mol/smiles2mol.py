"""Classes ment to transform given input to a RDKit molecule."""

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    StringToMolPipelineElement as _StringToMolPipelineElement
)
from molpipeline.utils.molpipe_types import OptionalMol


class SmilesToMolPipelineElement(_StringToMolPipelineElement):
    """Transforms Smiles to RDKit Mol objects."""

    def __init__(self, name: str = "smiles2mol", n_jobs: int = 1) -> None:
        """Initialize SmilesToMolPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        """
        super().__init__(name=name, n_jobs=n_jobs)

    def _transform_single(self, value: str) -> OptionalMol:
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
        mol.SetProp("identifier", value)
        return mol

