"""Classes for encoding molecules as phys-chem vector."""
# pylint: disable=too-many-arguments

from __future__ import annotations

from typing import Optional
import warnings

try:
    from chemprop.v2 import data as cp_data
    from chemprop.v2.data import MoleculeDatapoint
except ImportError:
    warnings.warn(
        "chemprop not installed. MolToChemprop will not work.",
        ImportWarning,
    )

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import (
    MolToAnyPipelineElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToChemprop(MolToAnyPipelineElement):
    """PipelineElement for creating a graph representation based on chemprop molecule classes."""

    def __init__(
        self,
        name: str = "Mol2Chemprop",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToChemprop.

        Parameters
        ----------
        name: str
            Name of the pipeline element. Defaults to "Mol2Chemprop".
        n_jobs: int
            Number of parallel jobs to use. Defaults to 1.
        none_handling: NoneHandlingOptions
            How to handle None values. Defaults to "raise".
        fill_value: Any
            Value to fill None values with. Defaults to None.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )

    def pretransform_single(self, value: RDKitMol) -> Optional[MoleculeDatapoint]:
        """Transform a single molecule to a ChemProp MoleculeDataPoint.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        Optional[MoleculeDatapoint]
            Molecular representation used as input for ChemProp. None if transformation failed.
        """
        smiles = Chem.MolToSmiles(value)
        if not smiles:
            return None

        return cp_data.MoleculeDatapoint(smiles, None)