"""Classes ment to transform given input to a RDKit molecule."""

from __future__ import annotations

from typing import Optional

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import (
    AnyToMolPipelineElement,
    InvalidInstance,
)
from molpipeline.utils.molpipeline_types import OptionalMol


class BinaryToMol(AnyToMolPipelineElement):
    """Transforms binary string representation to RDKit Mol objects."""

    def __init__(
        self,
        name: str = "bin2mol",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize BinaryToMol.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: str) -> OptionalMol:
        """Transform binary string to molecule.

        Parameters
        ----------
        value: str
            Binary string.

        Returns
        -------
        OptionalMol
            Rdkit molecule if valid binary representation, else None.
        """
        if value is None:
            return InvalidInstance(
                self.uuid,
                f"Invalid binary string: {value}",
                self.name,
            )

        if not isinstance(value, bytes):
            return InvalidInstance(
                self.uuid,
                f"Not bytes: {value}",
                self.name,
            )

        mol: OptionalMol | None = None
        try:
            mol = Chem.Mol(value)
        except RuntimeError:
            pass

        if not mol:
            return InvalidInstance(
                self.uuid,
                f"Invalid binary string: {value}",
                self.name,
            )
        return mol
