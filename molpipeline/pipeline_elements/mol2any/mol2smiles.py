"""Classes for transforming rdkit molecules to any type of output."""
from rdkit import Chem

from molpipeline.abstract_pipeline_elements.mol2any.mol2string import (
    MolToStringPipelineElement as _MolToStringPipelineElement,
)


class MolToSmilesPipelineElement(_MolToStringPipelineElement):
    """PipelineElement to transform a molecule to a SMILES string."""

    def __init__(self, name: str = "Mol2Smiles"):
        """Initialize MolToSmilesPipelineElement.

        Parameters
        ----------
        name: str
            name of PipelineElement
        """
        super().__init__(name)

    def _transform_single(self, value: Chem.Mol) -> str:
        """Transform a molecule to a SMILES string."""
        return str(Chem.MolToSmiles(value))
