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


class ElementFilter(_MolToMolPipelineElement):
    """ElementFilter which removes molecules containing chemical elements other than specified."""

    DEFAULT_ALLOWED_ELEMENT_NUMBERS = [
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

    def __init__(
        self,
        allowed_element_numbers: Optional[list[int]] = None,
        name: str = "ElementFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize ElementFilter.

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
            allowed_element_numbers = self.DEFAULT_ALLOWED_ELEMENT_NUMBERS
        if not isinstance(allowed_element_numbers, set):
            self.allowed_element_numbers = set(allowed_element_numbers)
        else:
            self.allowed_element_numbers = allowed_element_numbers

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of ElementFilter.

        Parameters
        ----------
        deep: bool, optional (default: True)
            If True, return the parameters of all subobjects that are PipelineElements.

        Returns
        -------
        dict[str, Any]
            Parameters of ElementFilter.
        """
        params = super().get_params(deep=deep)
        if deep:
            params["allowed_element_numbers"] = {
                int(atom) for atom in self.allowed_element_numbers
            }
        else:
            params["allowed_element_numbers"] = self.allowed_element_numbers
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters of ElementFilter.

        Parameters
        ----------
        parameters: Any
            Parameters to set.

        Returns
        -------
        Self
            Self.
        """
        parameter_copy = dict(parameters)
        if "allowed_element_numbers" in parameter_copy:
            self.allowed_element_numbers = parameter_copy.pop("allowed_element_numbers")
        super().set_params(**parameter_copy)
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


class MixtureFilter(_MolToMolPipelineElement):
    """MolToMol which removes molecules composed of multiple fragments."""

    def __int__(
        self,
        name: str = "MixtureFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MixtureFilter.

        Parameters
        ----------
        name: str, optional (default: "MixtureFilterPipe")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

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
        fragments = Chem.GetMolFrags(value, asMols=True)
        if len(fragments) > 1:
            smiles_fragments = [Chem.MolToSmiles(fragment) for fragment in fragments]
            return InvalidInstance(
                self.uuid,
                f"Molecule contains multiple fragments: {' '.join(smiles_fragments)}",
                self.name,
            )
        return value


class EmptyMoleculeFilter(_MolToMolPipelineElement):
    """EmptyMoleculeFilter which removes empty molecules."""

    def __init__(
        self,
        name: str = "EmptyMoleculeFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize EmptyMoleculeFilter.

        Parameters
        ----------
        name: str, optional (default: "EmptyMoleculeFilterPipe")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Invalidate empty molecule.

        Parameters
        ----------
        value: RDKitMol
            Molecule to check.

        Returns
        -------
        OptionalMol
            Molecule if it is not empty, else InvalidInstance.
        """
        if value.GetNumAtoms() == 0:
            return InvalidInstance(self.uuid, "Molecule contains no atoms.", self.name)
        return value


class InorganicsFilter(_MolToMolPipelineElement):
    """Filters Molecules which do not contain any organic (i.e. Carbon) atoms."""

    CARBON_INORGANICS = ["O=C=O", "[C-]#[O+]"]  # CO2 and CO are not organic
    CARBON_INORGANICS_MAX_ATOMS = 3

    def __init__(
        self,
        name: str = "InorganicsFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize InorganicsFilter.

        Parameters
        ----------
        name: str, optional (default: "InorganicsFilter")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Invalidate molecules not containing a carbon atom.

        Parameters
        ----------
        value: RDKitMol
            Molecule to check.

        Returns
        -------
        OptionalMol
            Molecule if it contains carbon, else InvalidInstance.
        """
        if not any(atom.GetAtomicNum() == 6 for atom in value.GetAtoms()):
            return InvalidInstance(
                self.uuid, "Molecule contains no organic atoms.", self.name
            )

        # Only check for inorganic molecules if the molecule is small enough
        if value.GetNumAtoms() <= self.CARBON_INORGANICS_MAX_ATOMS:
            smiles = Chem.MolToSmiles(value)
            if smiles in self.CARBON_INORGANICS:
                return InvalidInstance(self.uuid, "Molecule is not organic.", self.name)
        return value
