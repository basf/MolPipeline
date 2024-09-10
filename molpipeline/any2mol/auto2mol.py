"""Classes to transform given input to a RDKit molecule."""

from __future__ import annotations

from typing import Any, Optional

from molpipeline.abstract_pipeline_elements.core import (
    AnyToMolPipelineElement,
    InvalidInstance,
)
from molpipeline.any2mol.bin2mol import BinaryToMol
from molpipeline.any2mol.inchi2mol import InchiToMol
from molpipeline.any2mol.sdf2mol import SDFToMol
from molpipeline.any2mol.smiles2mol import SmilesToMol
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


class AutoToMol(AnyToMolPipelineElement):
    """Transforms various inputs to RDKit Mol objects.

    A cascade of if clauses is tried to transformer the given input to a molecule.
    """

    elements: tuple[AnyToMolPipelineElement, ...]

    def __init__(
        self,
        name: str = "auto2mol",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
        elements: tuple[AnyToMolPipelineElement, ...] = (
            SmilesToMol(),
            InchiToMol(),
            BinaryToMol(),
            SDFToMol(),
        ),
    ) -> None:
        """Initialize AutoToMol.

        Parameters
        ----------
        name: str, optional (default="auto2mol")
            Name of PipelineElement
        n_jobs: int, optional (default=1")
            Number of parallel jobs to use.
        uuid: str, optional (default=None)
            Unique identifier of PipelineElement.
        elements: tuple[AnyToMol, ...], optional (default=(SmilesToMol(),
            InchiToMol(), BinaryToMol(), SDFToMol()))
            Elements to try to transform the input to a molecule.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        # pipeline elements for transforming the input to a molecule
        self.elements = elements

    def pretransform_single(self, value: Any) -> OptionalMol:
        """Transform input value to molecule.

        Parameters
        ----------
        value: Any
            Input value.

        Returns
        -------
        OptionalMol
            Rdkit molecule if the input can be transformed, else None.
        """
        if value is None:
            return InvalidInstance(
                self.uuid,
                f"Invalid input molecule: {value}",
                self.name,
            )

        if isinstance(value, RDKitMol):
            return value

        # sequentially try to transform the input to a molecule using predefined elements
        for element in self.elements:
            mol = element.pretransform_single(value)
            if not isinstance(mol, InvalidInstance):
                return mol

        return InvalidInstance(
            self.uuid,
            f"Not readable input molecule: {value}",
            self.name,
        )
