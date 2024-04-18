"""Classes ment to transform given input to a RDKit molecule."""

from __future__ import annotations

from typing import Optional

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    StringToMolPipelineElement as _StringToMolPipelineElement,
)
from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


class SmilesToMol(_StringToMolPipelineElement):
    """Transforms Smiles to RDKit Mol objects."""

    def __init__(
        self,
        name: str = "smiles2mol",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize SmilesToMol.

        Parameters
        ----------
        name: str, optional (default="smiles2mol")
            Name of PipelineElement.
        n_jobs: int, optional (default=1)
            Number of cores used.
        uuid: str | None, optional (default=None)
            UUID of the pipeline element. If None, a random UUID is generated.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: str) -> OptionalMol:
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
        if value is None:
            return InvalidInstance(
                self.uuid,
                f"Invalid SMILES: {value}",
                self.name,
            )

        if not isinstance(value, str):
            return InvalidInstance(
                self.uuid,
                f"Not a string: {value}",
                self.name,
            )

        mol: RDKitMol = Chem.MolFromSmiles(value)

        if not mol:
            return InvalidInstance(
                self.uuid,
                f"Invalid SMILES: {value}",
                self.name,
            )
        mol.SetProp("identifier", value)
        return mol
