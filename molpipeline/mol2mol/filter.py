"""Classes to filter molecule lists."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from rdkit import Chem
from rdkit.Chem import Descriptors

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
)
from molpipeline.abstract_pipeline_elements.mol2mol import (
    BaseKeepMatchesFilter as _BaseKeepMatchesFilter,
)
from molpipeline.abstract_pipeline_elements.mol2mol import (
    BasePatternsFilter as _BasePatternsFilter,
)
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol
from molpipeline.utils.value_conversions import (
    FloatCountRange,
    IntCountRange,
    count_value_to_tuple,
)


class ElementFilter(_MolToMolPipelineElement):
    """ElementFilter which removes molecules containing chemical elements other than specified.

    Molecular elements are filtered based on their atomic number.
    The filter can be configured to allow only specific elements and/or a specific number of atoms of each element.
    """

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
        allowed_element_numbers: Optional[
            Union[list[int], dict[int, IntCountRange]]
        ] = None,
        name: str = "ElementFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize ElementFilter.

        Parameters
        ----------
        allowed_element_numbers: Optional[Union[list[int], dict[int, CountRange]]]
            List of atomic numbers of elements to allowed in molecules. Per default allowed elements are:
            H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I.
            Alternatively, a dictionary can be passed with atomic numbers as keys and an int for exact count or a tuple of minimum and maximum
        name: str, optional (default: "ElementFilterPipe")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.allowed_element_numbers = allowed_element_numbers  # type: ignore

    @property
    def allowed_element_numbers(self) -> dict[int, tuple[Optional[int], Optional[int]]]:
        """Get allowed element numbers as dict."""
        return self._allowed_element_numbers

    @allowed_element_numbers.setter
    def allowed_element_numbers(
        self,
        allowed_element_numbers: Optional[Union[list[int], dict[int, IntCountRange]]],
    ) -> None:
        """Set allowed element numbers as dict.

        Parameters
        ----------
        allowed_element_numbers: Optional[Union[list[int], dict[int, CountRange]]
            List of atomic numbers of elements to allowed in molecules.
        """
        self._allowed_element_numbers: dict[int, tuple[Optional[int], Optional[int]]]
        if allowed_element_numbers is None:
            allowed_element_numbers = self.DEFAULT_ALLOWED_ELEMENT_NUMBERS
        if isinstance(allowed_element_numbers, (list, set)):
            self._allowed_element_numbers = {
                atom_number: (0, None) for atom_number in allowed_element_numbers
            }
        else:
            self._allowed_element_numbers = {
                atom_number: count_value_to_tuple(count)
                for atom_number, count in allowed_element_numbers.items()
            }

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
                atom_number: (count_tuple[0], count_tuple[1])
                for atom_number, count_tuple in self.allowed_element_numbers.items()
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
        to_process_value = (
            Chem.AddHs(value) if 1 in self.allowed_element_numbers else value
        )

        elements_list = [atom.GetAtomicNum() for atom in to_process_value.GetAtoms()]
        elements_counter = Counter(elements_list)
        if any(
            element not in self.allowed_element_numbers for element in elements_counter
        ):
            return InvalidInstance(
                self.uuid, "Molecule contains forbidden chemical element.", self.name
            )
        for element, (min_count, max_count) in self.allowed_element_numbers.items():
            count = elements_counter[element]
            if (min_count is not None and count < min_count) or (
                max_count is not None and count > max_count
            ):
                return InvalidInstance(
                    self.uuid,
                    f"Molecule contains forbidden number of element {element}.",
                    self.name,
                )
        return value


class SmartsFilter(_BasePatternsFilter):
    """Filter to keep or remove molecules based on SMARTS patterns.

    Notes
    -----
    There are four possible scenarios:
        - mode = "any" & keep_matches = True: Needs to match at least one filter element.
        - mode = "any" & keep_matches = False: Must not match any filter element.
        - mode = "all" & keep_matches = True: Needs to match all filter elements.
        - mode = "all" & keep_matches = False: Must not match all filter elements.
    """

    def _pattern_to_mol(self, pattern: str) -> RDKitMol:
        """Convert SMARTS pattern to RDKit molecule.

        Parameters
        ----------
        pattern: str
            SMARTS pattern to convert.

        Returns
        -------
        RDKitMol
            RDKit molecule.
        """
        return Chem.MolFromSmarts(pattern)


class SmilesFilter(_BasePatternsFilter):
    """Filter to keep or remove molecules based on SMILES patterns.

    Notes
    -----
    There are four possible scenarios:
        - mode = "any" & keep_matches = True: Needs to match at least one filter element.
        - mode = "any" & keep_matches = False: Must not match any filter element.
        - mode = "all" & keep_matches = True: Needs to match all filter elements.
        - mode = "all" & keep_matches = False: Must not match all filter elements.
    """

    def _pattern_to_mol(self, pattern: str) -> RDKitMol:
        """Convert SMILES pattern to RDKit molecule.

        Parameters
        ----------
        pattern: str
            SMILES pattern to convert.

        Returns
        -------
        RDKitMol
            RDKit molecule.
        """
        return Chem.MolFromSmiles(pattern)


class ComplexFilter(_BaseKeepMatchesFilter):
    """Filter to keep or remove molecules based on multiple filter elements.

    Notes
    -----
    There are four possible scenarios:
        - mode = "any" & keep_matches = True: Needs to match at least one filter element.
        - mode = "any" & keep_matches = False: Must not match any filter element.
        - mode = "all" & keep_matches = True: Needs to match all filter elements.
        - mode = "all" & keep_matches = False: Must not match all filter elements.
    """

    def __init__(
        self,
        filter_elements: tuple[_MolToMolPipelineElement, ...],
        keep_matches: bool = True,
        mode: Literal["any", "all"] = "any",
        name: Optional[str] = None,
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize ComplexFilter.

        Parameters
        ----------
        filter_elements: tuple[_MolToMolPipelineElement, ...]
            tuple of filter elements.
        keep_matches: bool, optional (default: True)
            If True, molecules containing the specified patterns are kept, else removed.
        mode: Literal["any", "all"], optional (default: "any")
            If "any", at least one of the specified patterns must be present in the molecule.
            If "all", all of the specified patterns must be present in the molecule.
        name: Optional[str], optional (default: None)
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(
            keep_matches=keep_matches, mode=mode, name=name, n_jobs=n_jobs, uuid=uuid
        )
        self.filter_elements = {element: (1, None) for element in filter_elements}

    @property
    def filter_elements(
        self,
    ) -> dict[_MolToMolPipelineElement, tuple[int, Optional[int]]]:
        """Get filter elements."""
        return self._filter_elements

    @filter_elements.setter
    def filter_elements(
        self, filter_elements: dict[_MolToMolPipelineElement, tuple[int, Optional[int]]]
    ) -> None:
        """Set filter elements.

        Parameters
        ----------
        filter_elements: dict[_MolToMolPipelineElement, tuple[int, Optional[int]]]
            Filter elements to set.
        """
        self._filter_elements = filter_elements

    def _calculate_single_element_value(
        self, filter_element: Any, value: RDKitMol
    ) -> int:
        """Calculate a single filter match for a molecule.

        Parameters
        ----------
        filter_element: Any
            MolToMol Filter to calculate.
        value: RDKitMol
            Molecule to calculate filter match for.

        Returns
        -------
        int
            Filter match.
        """
        mol = filter_element.pretransform_single(value)
        if isinstance(mol, InvalidInstance):
            return 0
        return 1

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of ComplexFilter.

        Parameters
        ----------
        deep: bool, optional (default: True)
            If True, return the parameters of all subobjects that are PipelineElements.

        Returns
        -------
        dict[str, Any]
            Parameters of ComplexFilter.
        """
        params = super().get_params(deep=deep)
        if deep:
            params["filter_elements"] = {
                element: (count_tuple[0], count_tuple[1])
                for element, count_tuple in self.filter_elements.items()
            }
        else:
            params["filter_elements"] = self.filter_elements
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters of ComplexFilter.

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
        if "filter_elements" in parameter_copy:
            self.filter_elements = parameter_copy.pop("filter_elements")
        super().set_params(**parameter_copy)
        return self


class RDKitDescriptorsFilter(_BaseKeepMatchesFilter):
    """Filter to keep or remove molecules based on RDKit descriptors.

    Notes
    -----
    There are four possible scenarios:
        - mode = "any" & keep_matches = True: Needs to match at least one filter element.
        - mode = "any" & keep_matches = False: Must not match any filter element.
        - mode = "all" & keep_matches = True: Needs to match all filter elements.
        - mode = "all" & keep_matches = False: Must not match all filter elements.
    """

    def __init__(
        self,
        descriptors: dict[str, FloatCountRange],
        keep_matches: bool = True,
        mode: Literal["any", "all"] = "any",
        name: Optional[str] = None,
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize DescriptorsFilter.

        Parameters
        ----------
        descriptors: dict[str, FloatCountRange]
            Dictionary of RDKit descriptors to filter by.
            The value must be a tuple of minimum and maximum. If None, no limit is set.
        keep_matches: bool, optional (default: True)
            If True, molecules containing the specified descriptors are kept, else removed.
        mode: Literal["any", "all"], optional (default: "any")
            If "any", at least one of the specified descriptors must be present in the molecule.
            If "all", all of the specified descriptors must be present in the molecule.
        name: Optional[str], optional (default: None)
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(
            keep_matches=keep_matches, mode=mode, name=name, n_jobs=n_jobs, uuid=uuid
        )
        self.descriptors = descriptors

    @property
    def descriptors(self) -> dict[str, FloatCountRange]:
        """Get allowed descriptors as dict."""
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors: dict[str, FloatCountRange]) -> None:
        """Set allowed descriptors as dict.

        Parameters
        ----------
        descriptors: dict[str, FloatCountRange]
            Dictionary of RDKit descriptors to filter by.
        """
        self._descriptors = descriptors
        if not all(hasattr(Descriptors, descriptor) for descriptor in descriptors):
            raise ValueError(
                "You are trying to use an invalid descriptor. Use RDKit Descriptors module."
            )

    @property
    def filter_elements(self) -> dict[str, FloatCountRange]:
        """Get filter elements."""
        return self.descriptors

    def _calculate_single_element_value(
        self, filter_element: Any, value: RDKitMol
    ) -> float:
        """Calculate a single descriptor value for a molecule.

        Parameters
        ----------
        filter_element: Any
            Descriptor to calculate.
        value: RDKitMol
            Molecule to calculate descriptor for.

        Returns
        -------
        float
            Descriptor value.
        """
        return getattr(Descriptors, filter_element)(value)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of DescriptorFilter.

        Parameters
        ----------
        deep: bool, optional (default: True)
            If True, return the parameters of all subobjects that are PipelineElements.

        Returns
        -------
        dict[str, Any]
            Parameters of DescriptorFilter.
        """
        params = super().get_params(deep=deep)
        if deep:
            params["descriptors"] = {
                descriptor: (count_tuple[0], count_tuple[1])
                for descriptor, count_tuple in self.descriptors.items()
            }
        else:
            params["descriptors"] = self.descriptors
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters of DescriptorFilter.

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
        if "descriptors" in parameter_copy:
            self.descriptors = parameter_copy.pop("descriptors")
        super().set_params(**parameter_copy)
        return self


class MixtureFilter(_MolToMolPipelineElement):
    """MolToMol which removes molecules composed of multiple fragments."""

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
