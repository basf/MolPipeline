"""Classes for transforming rdkit molecules to any type of output."""

from typing import Any

from rdkit import Chem

from molpipeline.pipeline_elements.abstract_pipeline_elements import (
    MolToAnyPipelineElement as _Mol2AnyPipe,
)


class MolToSmilesPipelineElement(_Mol2AnyPipe):
    """PipelineElement to transform a molecule to a SMILES string."""

    def __init__(self, name: str = "Mol2Smiles"):
        """Initialize MolToSmilesPipelineElement.

        Parameters
        ----------
        name: str
            name of PipelineElement
        """
        super().__init__(name)

    def fit(self, value_list: Any) -> None:
        """Do nothing during fit."""

    def transform(self, value_list: list[Chem.Mol]) -> list[str]:
        """Transform a list molecules to a list of SMILES strings."""
        return [self.transform_single(mol) for mol in value_list]

    def transform_single(self, value: Chem.Mol) -> str:
        """Transform a molecule to a SMILES string."""
        return str(Chem.MolToSmiles(value))
