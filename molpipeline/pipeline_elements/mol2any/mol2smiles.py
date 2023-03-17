"""Classes for transforming rdkit molecules to any type of output."""
from __future__ import annotations
from typing import Any

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import NoneHandlingOptions
from molpipeline.abstract_pipeline_elements.mol2any.mol2string import (
    MolToStringPipelineElement as _MolToStringPipelineElement,
)


class MolToSmilesPipelineElement(_MolToStringPipelineElement):
    """PipelineElement to transform a molecule to a SMILES string."""

    def __init__(
        self,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "Mol2Smiles",
        n_jobs: int = 1,
    ):
        """Initialize MolToSmilesPipelineElement.

        Parameters
        ----------
        name: str
            name of PipelineElement
        """
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )

    @property
    def params(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        return super().params

    def copy(self) -> MolToSmilesPipelineElement:
        """Create a copy of the object."""
        return MolToSmilesPipelineElement(**self.params)

    def _transform_single(self, value: Chem.Mol) -> str:
        """Transform a molecule to a SMILES string."""
        return str(Chem.MolToSmiles(value))
