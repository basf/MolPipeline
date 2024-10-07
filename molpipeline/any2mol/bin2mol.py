"""Classes ment to transform given input to a RDKit molecule."""

from __future__ import annotations

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import (
    AnyToMolPipelineElement,
    InvalidInstance,
)
from molpipeline.utils.molpipeline_types import OptionalMol


class BinaryToMol(AnyToMolPipelineElement):
    """Transforms binary string representation to RDKit Mol objects."""

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
