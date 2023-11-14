"""Classes to filter molecule lists."""

from __future__ import annotations

from typing import Any, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
)
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


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
        if not isinstance(allowed_element_numbers, set):
            self.allowed_element_numbers = set(allowed_element_numbers)
        else:
            self.allowed_element_numbers = allowed_element_numbers

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of ElementFilterPipelineElement.

        Parameters
        ----------
        deep: bool, optional (default: True)
            If True, return the parameters of all subobjects that are PipelineElements.

        Returns
        -------
        dict[str, Any]
            Parameters of ElementFilterPipelineElement.
        """
        params = super().get_params(deep=deep)
        if deep:
            params["allowed_element_numbers"] = {
                int(atom) for atom in self.allowed_element_numbers
            }
        else:
            params["allowed_element_numbers"] = self.allowed_element_numbers
        return params

    def set_params(self, parameters: dict[str, Any]) -> Self:
        """Set parameters of ElementFilterPipelineElement.

        Parameters
        ----------
        parameters: dict[str, Any]
            Parameters to set.

        Returns
        -------
        Self
            Self.
        """
        parameter_copy = dict(parameters)
        if "allowed_element_numbers" in parameter_copy:
            self.allowed_element_numbers = parameter_copy.pop("allowed_element_numbers")
        super().set_params(parameter_copy)
        return self

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
            forbidden_elements = unique_elements - self.allowed_element_numbers
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
