"""Abstract classes for transforming rdkit molecules to bit vectors."""

from __future__ import annotations  # for all the python 3.8 users out there.

import abc
from typing import Iterable

from rdkit import Chem
from scipy import sparse
from molpipeline.utils.substructure_handling import CircularAtomEnvironment

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.utils.matrices import sparse_from_index_value_dicts


class MolToFingerprintPipelineElement(MolToAnyPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""

    _n_bits: int
    _output_type = sparse.csr_matrix

    @property
    def n_bits(self) -> int:
        """Get number of bits in (or size of) fingerprint."""
        return self._n_bits

    def assemble_output(
        self, row_dict_iterable: Iterable[dict[int, int]]
    ) -> sparse.csr_matrix:
        """Transform output of all transform_single operations to matrix."""
        return sparse_from_index_value_dicts(row_dict_iterable, self._n_bits)

    def transform(self, value_list: list[Chem.Mol]) -> sparse.csr_matrix:
        """Transform the list of molecules to sparse matrix."""
        return self.assemble_output(super().transform(value_list))

    @abc.abstractmethod
    def _transform_single(self, value: Chem.Mol) -> dict[int, int]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        dict[int, int]
            Dictionary to encode row in matrix. Keys: column index, values: column value
        """


class ABCMorganFingerprintPipelineElement(MolToFingerprintPipelineElement, abc.ABC):
    """Abstract Class for Morgan fingerprints."""

    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        name: str = "AbstractMorgan",
        n_jobs: int = 1,
    ):
        """Initialize abstract class.

        Parameters
        ----------
        radius: int
            Radius of fingerprint.
        use_features: bool
            Whether to represent atoms by element or category (donor, acceptor. etc.)
        name: str
            Name of PipelineElement.
        n_jobs:
            Number of jobs.
        """
        super().__init__(name=name, n_jobs=n_jobs)
        self._use_features = use_features
        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(
                f"Number of bits has to be a positive integer! (Received: {radius})"
            )

    @property
    def radius(self) -> int:
        """Get radius of Morgan fingerprint."""
        return self._radius

    @property
    def use_features(self) -> bool:
        """Get whether to encode atoms by features or not."""
        return self._use_features

    @abc.abstractmethod
    def _explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        """Get central atom and radius of all features in molecule."""
        raise NotImplementedError

    def bit2atom_mapping(
        self, mol_obj: Chem.Mol
    ) -> dict[int, list[CircularAtomEnvironment]]:
        """Obtain set of atoms for all features."""
        bit2atom_dict = self._explain_rdmol(mol_obj)
        result_dict: dict[int, list[CircularAtomEnvironment]] = {}
        # Iterating over all present bits and respective matches
        for bit, matches in bit2atom_dict.items():  # type: int, list[tuple[int, int]]
            result_dict[bit] = []
            for central_atom, radius in matches:  # type: int, int
                env = CircularAtomEnvironment.from_mol(mol_obj, central_atom, radius)
                result_dict[bit].append(env)
        # Transforming default dict to dict
        return result_dict
