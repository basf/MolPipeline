"""Implementations for the Morgan fingerprint."""

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.DataStructs import (
    SparseBitVect,
)

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    FPReturnAsOption,
    FPTransformSingleReturnDataType,
    MolToFingerprintPipelineElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol
from molpipeline.utils.substructure_handling import AtomEnvironment


class MolToFragmentFP(MolToFingerprintPipelineElement):
    """Pipeline element for fragment fingerprints from substructure SMARTS patterns."""

    def __init__(
        self,
        substructure_list: list[str],
        counted: bool = False,
        return_as: FPReturnAsOption = "sparse",
        name: str = "MolToFragmentFP",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize the MolToFragmentFP element.

        Parameters
        ----------
        substructure_list : list[str]
            List of SMARTS patterns for substructures.
        counted : bool, optional
            Whether to count occurrences (default: False).
        return_as : FPReturnAsOption, optional
            Output format (default: "sparse").
        name : str, optional
            Name of the element (default: "MolToFragmentFP").
        n_jobs : int, optional
            Number of jobs for parallelization (default: 1).
        uuid : str | None, optional
            Unique identifier (default: None).

        """
        super().__init__(
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self.substructure_list = substructure_list
        self.counted = counted

    @property
    def n_bits(self) -> int:
        """Return the number of bits in the fingerprint."""
        return self._n_bits

    @property
    def substructure_list(self) -> list[str]:
        """Get the list of SMARTS substructures."""
        return self._substructure_list

    @substructure_list.setter
    def substructure_list(self, substructure_list: list[str]) -> None:
        """Set the list of SMARTS substructures and update internal state.

        Parameters
        ----------
        substructure_list : list[str]
            The list of SMARTS substructures.

        Raises
        ------
        ValueError
            If a SMARTS pattern is invalid.

        """
        self._substructure_list = substructure_list
        self._filter = FilterCatalog.FilterCatalog()
        self._n_bits: int = len(self._substructure_list)
        self._substructure_obj_list: list[Chem.Mol] = []
        for i, substructure in enumerate(self._substructure_list):
            # Validating Smarts
            smarts_obj = Chem.MolFromSmarts(substructure)
            if smarts_obj is None:
                raise ValueError(f"Invalid SMARTS pattern: {substructure}")
            self._substructure_obj_list.append(smarts_obj)

            # Adding pattern to the filter catalogue
            pattern = FilterCatalog.SmartsMatcher(f"Pattern {i}", substructure, 1)
            self._filter.AddEntry(FilterCatalog.FilterCatalogEntry(str(i), pattern))

    def pretransform_single(
        self,
        value: RDKitMol,
    ) -> FPTransformSingleReturnDataType:
        """Transform a single molecule to its fragment fingerprint.

        Parameters
        ----------
        value : RDKitMol
            The molecule to transform.

        Returns
        -------
        FPTransformSingleReturnDataType
            The fingerprint representation.

        Raises
        ------
        AssertionError
            If the return_as option is unknown.

        """
        matched_bits = [
            int(match.GetDescription()) for match in self._filter.GetMatches(value)
        ]
        if self.counted:
            bit_counts = {}
            for bit in matched_bits:
                bit_counts[bit] = len(
                    value.GetSubstructMatches(self._substructure_obj_list[bit]),
                )
        else:
            bit_counts = dict.fromkeys(matched_bits, 1)
        if self._return_as == "sparse":
            return bit_counts
        vec = [0 for _ in range(self.n_bits)]
        for bit, count in bit_counts.items():
            vec[bit] = count

        if self._return_as == "dense":
            return np.array(vec, dtype=int)

        if self._return_as == "bitvector":
            bitvector = SparseBitVect(self.n_bits)
            for bit in bit_counts:
                bitvector.SetBit(bit)
            return bitvector

        raise AssertionError(f"Unknown return_as option: {self._return_as}")

    def bit2atom_mapping(self, value: RDKitMol) -> dict[int, list[AtomEnvironment]]:
        """Map fingerprint bits to atom environments in the molecule.

        Parameters
        ----------
        value : RDKitMol
            The molecule to analyze.

        Returns
        -------
        dict[int, list[AtomEnvironment]]
            Mapping from bit index to atom environments.

        """
        present_bits = [
            int(match.GetDescription()) for match in self._filter.GetMatches(value)
        ]
        bit2atom_dict = defaultdict(list)
        for bit in present_bits:
            bit_smarts_obj = self._substructure_obj_list[bit]
            matches = value.GetSubstructMatches(bit_smarts_obj)
            for match in matches:
                atom_env = AtomEnvironment(match)
                bit2atom_dict[bit].append(atom_env)

        # Transforming defaultdict to dict
        return dict(bit2atom_dict)
