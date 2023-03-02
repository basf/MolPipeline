import abc
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from scipy import sparse

from molpipe.utils.molpipe_types import OptionalMol
from molpipe.utils.matrices import sparse_from_index_value_dicts


class AnyPipeElement(abc.ABC):
    @abc.abstractmethod
    def fit(self, input_values: Any) -> None:
        """Fit object to input_values."""

    @abc.abstractmethod
    def transform(self, input_values: Any) -> Any:
        """Transform input_values according to object rules."""

    def fit_transform(self, input_values: Any) -> Any:
        """Apply fit function and subsequently transorm the input"""
        self.fit(input_values)
        return self.transform(input_values)


class Mol2MolPipe(AnyPipeElement):
    def transform(self, mol_list: list[OptionalMol]) -> list[OptionalMol]:
        return [mol for mol in map(self._transform_single, mol_list)]

    def _transform_single(self, mol: OptionalMol) -> OptionalMol:
        """Wrap the transform_mol method to handle Nones."""
        if not mol:
            return None
        else:
            return self.transform_single(mol)

    @abc.abstractmethod
    def transform_single(self, mol: Chem.Mol) -> OptionalMol:
        """Transform the molecule according to child dependent rules."""


class Any2Mol(AnyPipeElement):
    @abc.abstractmethod
    def transform(self, any_input: Any) -> list[OptionalMol]:
        """Transform the list of molecules to an array."""


class Mol2Any(AnyPipeElement):
    @abc.abstractmethod
    def transform(self, mol_list: list[Chem.Mol]) -> Any:
        """Transform the list of molecules to output of any type."""


class Mol2Fingerprint(Mol2Any):
    _n_bits: int

    @property
    def n_bits(self) -> int:
        return self._n_bits

    def collect_singles(self, row_dict_iterable:  Iterable[dict[int, int]]) -> sparse.csr_matrix:
        """Transform output of all transform_single operations to matrix."""
        return sparse_from_index_value_dicts(row_dict_iterable, self._n_bits)

    def transform(self, mol_list: list[Chem.Mol]) -> sparse.csr_matrix:
        """Transform the list of molecules to sparse matrix."""
        return self.collect_singles((self.transform_single(mol) for mol in mol_list))

    @abc.abstractmethod
    def transform_single(self, mol: Chem.Mol) -> dict[int, int]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        mol: Chem.Mol

        Returns
        -------
        dict[int, int]
            Dictionary to encode row in matrix. Keys: column index, values: column value
        """


class Mol2Descriptor(Mol2Any):
    @abc.abstractmethod
    def transform(self, mol_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to numpy array."""
