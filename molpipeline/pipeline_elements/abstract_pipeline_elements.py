"""All abstract classes later pipeline elements inherit from."""

import abc
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from scipy import sparse

from molpipeline.utils.molpipe_types import OptionalMol
from molpipeline.utils.matrices import sparse_from_index_value_dicts
from molpipeline.utils.multi_proc import check_available_cores, wrap_parallelizable_task


class ABCPipelineElement(abc.ABC):
    """Ancestor of all PipelineElements."""

    _input_type: type
    _output_type: type
    name: str

    def __init__(self, name: str = "ABCPipelineElement", n_jobs: int = 1) -> None:
        """Initialize ABCPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores used for processing.
        """
        self.name = name
        self.n_jobs = n_jobs

    @property
    def input_type(self) -> type:
        """Return the input type."""
        return self._input_type

    @property
    def n_jobs(self) -> int:
        """Get the number of cores."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int) -> None:
        """Set the number of cores."""
        self._n_jobs = check_available_cores(n_jobs)

    @property
    def output_type(self) -> type:
        """Return the output type."""
        return self._output_type

    def fit(self, value_list: Any) -> None:
        """Fit object to input_values. Does often nothing."""

    def fit_transform(self, value_list: Any) -> Any:
        """Apply fit function and subsequently transform the input."""
        self.fit(value_list)
        return self.transform(value_list)

    @abc.abstractmethod
    def transform(self, value_list: Any) -> Any:
        """Transform input_values according to object rules."""
        return wrap_parallelizable_task(self.transform_single, value_list, self.n_jobs)

    @abc.abstractmethod
    def transform_single(self, value: Any) -> Any:
        """Transform the molecule according to child dependent rules."""

    def finish(self) -> None:
        """Inform object that iteration has been finished. Does in most cases nothing.

        Called after all transform singles have been processed. From MolPipeline
        """


class MolToMolPipelineElement(ABCPipelineElement, abc.ABC):
    """Abstract PipelineElement where input and outputs are molecules."""

    _input_type = Chem.Mol
    _output_type = Chem.Mol

    def __init__(self, name: str = "MolToMolPipelineElement", n_jobs: int = 1) -> None:
        """Initialize MolToMolPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs)

    def transform(self, value_list: list[OptionalMol]) -> list[OptionalMol]:
        """Transform list of molecules to list of molecules."""
        return wrap_parallelizable_task(self._transform_single_catch_nones, value_list, self.n_jobs)

    @abc.abstractmethod
    def transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Transform the molecule according to child dependent rules."""

    def _transform_single_catch_nones(self, value: OptionalMol) -> OptionalMol:
        """Wrap the transform_single method to handle Nones."""
        if not value:
            return None
        return self.transform_single(value)


class AnyToMolPipelineElement(ABCPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _output_type = Chem.Mol

    def __init__(self, name: str = "AnyToMolPipelineElement", n_jobs: int = 1) -> None:
        """Initialize AnyToMolPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs)

    def transform(self, value_list: Any) -> list[OptionalMol]:
        """Transform list of molecules to list of molecules."""
        mol_list: list[OptionalMol] = super().transform(value_list)  # Stupid mypy...
        return mol_list

    @abc.abstractmethod
    def transform_single(self, value: Any) -> OptionalMol:
        """Transform the input specified in each child to molecules."""


class MolToAnyPipelineElement(ABCPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _input_type = Chem.Mol

    def __init__(self, name: str = "MolToAnyPipelineElement", n_jobs: int = 1) -> None:
        """Initialize MolToAnyPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs)

    @abc.abstractmethod
    def transform_single(self, value: Chem.Mol) -> Any:
        """Transform the molecules to the input specified in each child."""


class MolToFingerprintPipelineElement(MolToAnyPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""

    _n_bits: int
    _output_type = sparse.csr_matrix

    @property
    def n_bits(self) -> int:
        """Get number of bits in (or size of) fingerprint."""
        return self._n_bits

    def collect_rows(self, row_dict_iterable: Iterable[dict[int, int]]) -> sparse.csr_matrix:
        """Transform output of all transform_single operations to matrix."""
        return sparse_from_index_value_dicts(row_dict_iterable, self._n_bits)

    def transform(self, value_list: list[Chem.Mol]) -> sparse.csr_matrix:
        """Transform the list of molecules to sparse matrix."""
        return self.collect_rows(super().transform(value_list))

    @abc.abstractmethod
    def transform_single(self, value: Chem.Mol) -> dict[int, int]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        dict[int, int]
            Dictionary to encode row in matrix. Keys: column index, values: column value
        """


class MolToDescriptorPipelineElement(MolToAnyPipelineElement):
    """PipelineElement which generates a matrix from descriptor-vectors of each molecule."""

    @staticmethod
    def collect_rows(value_list: list[npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
        """Transform output of all transform_single operations to matrix."""
        return np.vstack(value_list)

    def transform(self, value_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix."""
        return self.collect_rows(super().transform(value_list))

    @abc.abstractmethod
    def transform_single(self, value: Chem.Mol) -> npt.NDArray[np.float_]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        npt.NDArray[np.float_]
            Vector with descriptor values of molecule.
        """
