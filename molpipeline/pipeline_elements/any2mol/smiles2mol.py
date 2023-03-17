"""Classes ment to transform given input to a RDKit molecule."""
from __future__ import annotations

from typing import Any
from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    StringToMolPipelineElement as _StringToMolPipelineElement,
)
from molpipeline.abstract_pipeline_elements.core import NoneHandlingOptions
from molpipeline.utils.molpipe_types import OptionalMol


class SmilesToMolPipelineElement(_StringToMolPipelineElement):
    """Transforms Smiles to RDKit Mol objects."""

    def __init__(
        self,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "smiles2mol",
        n_jobs: int = 1,
    ) -> None:
        """Initialize SmilesToMolPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        """
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )

    @property
    def params(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        return super().params

    def copy(self) -> SmilesToMolPipelineElement:
        """Create a copy of the object."""
        return SmilesToMolPipelineElement(**self.params)

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
