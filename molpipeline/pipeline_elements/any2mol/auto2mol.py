"""Classes to transform given input to a RDKit molecule."""

from __future__ import annotations

from typing import Any, Optional

from molpipeline.abstract_pipeline_elements.core import (
    AnyToMolPipelineElement,
    InvalidInstance,
)
from molpipeline.pipeline_elements.any2mol import (
    SDFToMolPipelineElement,
    SmilesToMolPipelineElement,
)
from molpipeline.pipeline_elements.any2mol.bin2mol import BinaryToMolPipelineElement
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


class AutoToMolPipelineElement(AnyToMolPipelineElement):
    """Transforms various inputs to RDKit Mol objects.

    A cascade of if clauses is tried to transformer the given input to a molecule.
    """

    def __init__(
        self,
        name: str = "auto2mol",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
        elements: tuple[AnyToMolPipelineElement, ...] = (
            SmilesToMolPipelineElement(),
            BinaryToMolPipelineElement(),
            SDFToMolPipelineElement(),
        ),
    ) -> None:
        """Initialize AutoToMolPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        # pipeline elements for transforming the input to a molecule
        self.elements = elements

    def pretransform_single(self, value: Any) -> OptionalMol:
        """Transform input value to molecule.

        Parameters
        ----------
        value: str
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
