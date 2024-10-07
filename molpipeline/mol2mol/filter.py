"""Classes to filter molecule lists."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional, Sequence, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from loguru import logger
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
from molpipeline.abstract_pipeline_elements.mol2mol.filter import (
    FilterModeType,
    _within_boundaries,
)
from molpipeline.utils.molpipeline_types import (
    FloatCountRange,
    IntCountRange,
    IntOrIntCountRange,
    OptionalMol,
    RDKitMol,
)
from molpipeline.utils.value_conversions import count_value_to_tuple


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
            Union[list[int], dict[int, IntOrIntCountRange]]
        ] = None,
        add_hydrogens: bool = True,
        name: str = "ElementFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize ElementFilter.

        Parameters
        ----------
        allowed_element_numbers: Optional[Union[list[int], dict[int, IntOrIntCountRange]]]
            List of atomic numbers of elements to allowed in molecules. Per default allowed elements are:
            H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I.
            Alternatively, a dictionary can be passed with atomic numbers as keys and an int for exact count or a tuple of minimum and maximum
        add_hydrogens: bool, optional (default: True)
            If True, in case Hydrogens are in allowed_element_list, add hydrogens to the molecule before filtering.
        name: str, optional (default: "ElementFilterPipe")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.allowed_element_numbers = allowed_element_numbers  # type: ignore
        self.add_hydrogens = add_hydrogens

    @property
    def add_hydrogens(self) -> bool:
        """Get add_hydrogens."""
        return self._add_hydrogens

    @add_hydrogens.setter
    def add_hydrogens(self, add_hydrogens: bool) -> None:
        """Set add_hydrogens.

        Parameters
        ----------
        add_hydrogens: bool
            If True, in case Hydrogens are in allowed_element_list, add hydrogens to the molecule before filtering.
        """
        self._add_hydrogens = add_hydrogens
        if self.add_hydrogens and 1 in self.allowed_element_numbers:
            self.process_hydrogens = True
        else:
            if 1 in self.allowed_element_numbers:
                logger.warning(
                    "Hydrogens are included in allowed_element_numbers, but add_hydrogens is set to False. "
                    "Thus hydrogens are NOT added before filtering. You might receive unexpected results."
                )
            self.process_hydrogens = False

    @property
    def allowed_element_numbers(self) -> dict[int, IntCountRange]:
        """Get allowed element numbers as dict."""
        return self._allowed_element_numbers

    @allowed_element_numbers.setter
    def allowed_element_numbers(
        self,
        allowed_element_numbers: Optional[
            Union[list[int], dict[int, IntOrIntCountRange]]
        ],
    ) -> None:
        """Set allowed element numbers as dict.

        Parameters
        ----------
        allowed_element_numbers: Optional[Union[list[int], dict[int, IntOrIntCountRange]]
            List of atomic numbers of elements to allowed in molecules.
        """
        self._allowed_element_numbers: dict[int, IntCountRange]
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
        params["add_hydrogens"] = self.add_hydrogens
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
        if "add_hydrogens" in parameter_copy:
            self.add_hydrogens = parameter_copy.pop("add_hydrogens")
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
        to_process_value = Chem.AddHs(value) if self.process_hydrogens else value
        elements_list = [atom.GetAtomicNum() for atom in to_process_value.GetAtoms()]
        elements_counter = Counter(elements_list)
        if any(
            element not in self.allowed_element_numbers for element in elements_counter
        ):
            return InvalidInstance(
                self.uuid, "Molecule contains forbidden chemical element.", self.name
            )
        for element, (lower_limit, upper_limit) in self.allowed_element_numbers.items():
            count = elements_counter[element]
            if not _within_boundaries(lower_limit, upper_limit, count):
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

    In contrast to the SMARTSFilter, which also can match SMILES, the SmilesFilter
    sanitizes the molecules and, e.g. checks kekulized bonds for aromaticity and
    then sets it to aromatic while the SmartsFilter detects alternating single and
    double bonds.

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

    Attributes
    ----------
    pipeline_filter_elements: Sequence[tuple[str, _MolToMolPipelineElement]]
        pairs of unique names and MolToMol elements to use as filters.
    [...]

    Notes
    -----
    There are four possible scenarios:
        - mode = "any" & keep_matches = True: Needs to match at least one filter element.
        - mode = "any" & keep_matches = False: Must not match any filter element.
        - mode = "all" & keep_matches = True: Needs to match all filter elements.
        - mode = "all" & keep_matches = False: Must not match all filter elements.
    """

    _filter_elements: Mapping[str, tuple[int, Optional[int]]]

    def __init__(
        self,
        pipeline_filter_elements: Sequence[tuple[str, _MolToMolPipelineElement]],
        keep_matches: bool = True,
        mode: FilterModeType = "any",
        name: str | None = None,
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize ComplexFilter.

        Parameters
        ----------
        pipeline_filter_elements: Sequence[tuple[str, _MolToMolPipelineElement]]
            Filter elements to use.
        keep_matches: bool, optional (default: True)
            If True, keep matches, else remove matches.
        mode: FilterModeType, optional (default: "any")
            Mode to filter by.
        name: str, optional (default: None)
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        self.pipeline_filter_elements = pipeline_filter_elements
        super().__init__(
            filter_elements=pipeline_filter_elements,
            keep_matches=keep_matches,
            mode=mode,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )

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
        params = super().get_params(deep)
        params.pop("filter_elements")
        params["pipeline_filter_elements"] = self.pipeline_filter_elements
        if deep:
            for name, element in self.pipeline_filter_elements:
                deep_items = element.get_params().items()
                params.update(
                    ("pipeline_filter_elements" + "__" + name + "__" + key, val)
                    for key, val in deep_items
                )
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
        if "pipeline_filter_elements" in parameter_copy:
            self.pipeline_filter_elements = parameter_copy.pop(
                "pipeline_filter_elements"
            )
            self.filter_elements = self.pipeline_filter_elements  # type: ignore
        for key in parameters:
            if key.startswith("pipeline_filter_elements__"):
                value = parameter_copy.pop(key)
                element_name, element_key = key.split("__")[1:]
                for name, element in self.pipeline_filter_elements:
                    if name == element_name:
                        element.set_params(**{element_key: value})
        super().set_params(**parameter_copy)
        return self

    @property
    def filter_elements(
        self,
    ) -> Mapping[str, tuple[int, Optional[int]]]:
        """Get filter elements."""
        return self._filter_elements

    @filter_elements.setter
    def filter_elements(
        self,
        filter_elements: Sequence[tuple[str, _MolToMolPipelineElement]],
    ) -> None:
        """Set filter elements.

        Parameters
        ----------
        filter_elements: Sequence[tuple[str, _MolToMolPipelineElement]]
            Filter elements to set.
        """
        self.filter_elements_dict = dict(filter_elements)
        if not len(self.filter_elements_dict) == len(filter_elements):
            raise ValueError("Filter elements names need to be unique.")
        self._filter_elements = {
            element_name: (1, None) for element_name, _element in filter_elements
        }

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
        mol = self.filter_elements_dict[filter_element].pretransform_single(value)
        if isinstance(mol, InvalidInstance):
            return 0
        return 1


class RDKitDescriptorsFilter(_BaseKeepMatchesFilter):
    """Filter to keep or remove molecules based on RDKit descriptors.

    Attributes
    ----------
    filter_elements: dict[str, FloatCountRange]
        Dictionary of RDKit descriptors to filter by.
        The value must be a tuple of minimum and maximum. If None, no limit is set.
    [...]

    Notes
    -----
    There are four possible scenarios:
        - mode = "any" & keep_matches = True: Needs to match at least one filter element.
        - mode = "any" & keep_matches = False: Must not match any filter element.
        - mode = "all" & keep_matches = True: Needs to match all filter elements.
        - mode = "all" & keep_matches = False: Must not match all filter elements.
    """

    @property
    def filter_elements(self) -> dict[str, FloatCountRange]:
        """Get allowed descriptors as dict."""
        return self._filter_elements

    @filter_elements.setter
    def filter_elements(self, descriptors: dict[str, FloatCountRange]) -> None:
        """Set allowed descriptors as dict.

        Parameters
        ----------
        descriptors: dict[str, FloatCountRange]
            Dictionary of RDKit descriptors to filter by.
        """
        if not all(hasattr(Descriptors, descriptor) for descriptor in descriptors):
            raise ValueError(
                "You are trying to use an invalid descriptor. Use RDKit Descriptors module."
            )
        self._filter_elements = descriptors

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
