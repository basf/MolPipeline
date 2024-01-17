"""Converter element for molecules to binary string representation."""
from typing import Optional

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement


class MolToBinaryPipelineElement(MolToAnyPipelineElement):
    """PipelineElement to transform a molecule to a binary."""

    def __init__(
        self,
        name: str = "Mol2Binary",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToSmilesPipelineElement.

        Parameters
        ----------
        name: str
            name of PipelineElement
        n_jobs: int
            number of jobs to use for parallelization
        uuid: Optional[str], optional
            uuid of PipelineElement, by default None

        Returns
        -------
        None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: Chem.Mol) -> str:
        """Transform a molecule to a SMILES string.

        Parameters
        ----------
        value: Chem.Mol
            Molecule to be transformed to binary string representation.

        Returns
        -------
        str
            Binary representation of molecule.
        """
        return value.ToBinary()
