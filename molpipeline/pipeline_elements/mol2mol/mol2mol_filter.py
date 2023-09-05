"""Classes to filter molecule lists."""

from __future__ import annotations
from typing import Optional

from rdkit import Chem

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
        name: str = "ElementFilterPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize ElementFilterPipelineElement.

        Parameters
        ----------
        allowed_element_numbers: list[int]
            List of atomic numbers of elements to allowed in molecules. Per default allowed elements are:
            H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I.
        name: str, optional (default: "ElementFilterPipe")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
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

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Invalidate molecule containing chemical elements that are not allowed.

        Parameters
        ----------
        value: RDKitMol
            Molecule to check.

        Returns
        -------
        OptionalMol
            Molecule if it contains only allowed elements, else InvalidInstance.
        """
        unique_elements = set(atom.GetAtomicNum() for atom in value.GetAtoms())
        if not unique_elements.issubset(self.allowed_element_numbers):
            forbidden_elements = self.allowed_element_numbers - unique_elements
            return InvalidInstance(
                self.uuid,
                f"Molecule contains following forbidden elements: {forbidden_elements}",
                self.name,
            )
        return value


class MixtureFilterPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes molecules composed of multiple fragments."""

    def __int__(
        self,
        name: str = "MixtureFilterPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MixtureFilterPipelineElement.

        Parameters
        ----------
        name: str, optional (default: "MixtureFilterPipe")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Invalidate molecule containing multiple fragments.

        Parameters
        ----------
        value: RDKitMol
            Molecule to check.

        Returns
        -------
        OptionalMol
            Molecule if it contains only one fragment, else InvalidInstance.
        """
        fragments = Chem.GetMolFrags(value)
        if len(fragments) > 1:
            return InvalidInstance(
                self.uuid,
                f"Molecule contains multiple fragments: {fragments}",
                self.name,
            )
        return value
