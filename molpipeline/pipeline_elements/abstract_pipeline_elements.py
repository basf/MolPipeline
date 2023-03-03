"""All abstract classes later pipeline elements inherit from."""

import abc
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from scipy import sparse

from molpipeline.utils.molpipe_types import OptionalMol
from molpipeline.utils.matrices import sparse_from_index_value_dicts


class AnyPipeElement(abc.ABC):
    _input_type: type
    _output_type: type
    name: str

    def __init__(self, name: str = "AnyPipeElement") -> None:
        self.name = name

    @property
    def input_type(self) -> type:
        """Return the input type"""
        return self._input_type

    @property
    def output_type(self) -> type:
        """Return the output type"""
        return self._output_type

    def finish(self) -> None:
        pass

    @abc.abstractmethod
    def fit(self, input_values: Any) -> None:
        """Fit object to input_values."""

    def fit_transform(self, input_values: Any) -> Any:
        """Apply fit function and subsequently transform the input"""
        self.fit(input_values)
        return self.transform(input_values)

    @abc.abstractmethod
    def transform(self, input_values: Any) -> Any:
        """Transform input_values according to object rules."""

    @abc.abstractmethod
    def transform_single(self, mol: Chem.Mol) -> OptionalMol:
        """Transform the molecule according to child dependent rules."""


class Mol2MolPipe(AnyPipeElement):
    _input_type = Chem.Mol
    _output_type = Chem.Mol

    def transform(self, mol_list: list[OptionalMol]) -> list[OptionalMol]:
        return [mol for mol in map(self._transform_single, mol_list)]

    def _transform_single(self, mol: OptionalMol) -> OptionalMol:
        """Wrap the transform_mol method to handle Nones."""
        if not mol:
            return None
        else:
            return self.transform_single(mol)


class Any2MolPipe(AnyPipeElement):
    _output_type = Chem.Mol

    def __init__(self, name: str = "Any2Mol") -> None:
        super(Any2MolPipe, self).__init__(name)

    @abc.abstractmethod
    def transform(self, any_input: Any) -> list[OptionalMol]:
        """Transform the list of molecules to an array."""


class Mol2AnyPipe(AnyPipeElement):
    _input_type = Chem.Mol

    @abc.abstractmethod
    def transform(self, mol_list: list[Chem.Mol]) -> Any:
        """Transform the list of molecules to output of any type."""


class Mol2FingerprintPipe(Mol2AnyPipe):
    _n_bits: int
    _output_type = sparse.csr_matrix

    @property
    def n_bits(self) -> int:
        """Number of bits in (or size of) fingerprint."""
        return self._n_bits

    def collect_singles(self, row_dict_iterable: Iterable[dict[int, int]]) -> sparse.csr_matrix:
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


class Mol2DescriptorPipe(Mol2AnyPipe):
    @abc.abstractmethod
    def transform(self, mol_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to numpy array."""
