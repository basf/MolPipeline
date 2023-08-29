"""Classes for transforming rdkit molecules to any type of output."""
from __future__ import annotations

from typing import Optional

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.mol2any.mol2string import (
    MolToStringPipelineElement as _MolToStringPipelineElement,
)


class MolToSmilesPipelineElement(_MolToStringPipelineElement):
    """PipelineElement to transform a molecule to a SMILES string."""

    def __init__(
        self,
        name: str = "Mol2Smiles",
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
            Molecule to be transformed to SMILES string.

        Returns
        -------
        str
            SMILES string of molecule.
        """
        return str(Chem.MolToSmiles(value))
