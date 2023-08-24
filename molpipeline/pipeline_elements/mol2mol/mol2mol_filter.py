"""Classes to filter molecule lists."""

from __future__ import annotations
from typing import Optional

from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
    InvalidInstance,
)
from molpipeline.utils.molpipeline_types import RDKitMol, OptionalMol


class ElementFilterPipelineElement(_MolToMolPipelineElement):
    """ElementFilterPipelineElement which removes molecules containing chemical elements other than specified."""

    def __init__(
        self,
        allowed_element_numbers: Optional[list[int]] = None,
        name: str = "ElementFilterPipe",
        n_jobs: int = 1,
    ) -> None:
        """Initialize ElementFilterPipelineElement.

        Parameters
        ----------
        allowed_element_numbers: list[int]
            List of atomic numbers of elements to allowed in molecules. Per default allowed elements are:
            H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I.
        none_handling: NoneHandlingOptions, optional (default: "raise")
            How to handle None values in the input data.
        fill_value: Any, optional (default: None)
            Value to fill None values with.
        name: str, optional (default: "ElementFilterPipe")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        """
        super().__init__(name=name, n_jobs=n_jobs)
        if allowed_element_numbers is None:
            allowed_element_numbers = [
                1,
                5,
                6,
                7,
                8,
                9,
                14,
                15,
                16,
                17,
                34,
                35,
                53,
            ]
        self.allowed_element_numbers: set[int] = set(allowed_element_numbers)

    def _transform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove molecule containing chemical elements that are not allowed."""
        unique_elements = set(atom.GetAtomicNum() for atom in value.GetAtoms())
        if not unique_elements.issubset(self.allowed_element_numbers):
            forbidden_elements = self.allowed_element_numbers - unique_elements
            return InvalidInstance(
                self,
                f"Molecule contains following forbidden elements: {forbidden_elements}",
            )
        return value
