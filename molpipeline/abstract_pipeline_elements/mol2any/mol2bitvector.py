"""Abstract classes for transforming rdkit molecules to bit vectors."""

from __future__ import annotations  # for all the python 3.8 users out there.

import abc
from typing import Any, Iterable, Literal

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy
from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]
from scipy import sparse
from molpipeline.utils.substructure_handling import CircularAtomEnvironment

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.utils.matrices import sparse_from_index_value_dicts


class MolToFingerprintPipelineElement(MolToAnyPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""

    _n_bits: int
    _output_type = sparse.csr_matrix
    _sparse_output: bool

    def __init__(
        self,
        sparse_output: bool = True,
        none_handling: Literal["raise", "record_remove", "fill_dummy"] = "raise",
        fill_value: Any = None,
        name: str = "MolToFingerprintPipelineElement",
        n_jobs: int = 1,
    ):
        """Initialize abstract class.

        Parameters
        ----------
        sparse_output: bool, optional
            True: return sparse matrix, False: return matrix as dense numpy array.
        none_handling: Literal["raise", "record_remove", "fill_dummy"], optional
            Specifies how to handle None values in the input list.
        name: str
            Name of PipelineElement.
        n_jobs:
            Number of jobs.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            none_handling=none_handling,
            fill_value=fill_value,
        )
        self._sparse_output = sparse_output

    @property
    def n_bits(self) -> int:
        """Get number of bits in (or size of) fingerprint."""
        return self._n_bits

    def assemble_output(
        self, value_list: Iterable[dict[int, int]]
    ) -> sparse.csr_matrix:
        """Transform output of all transform_single operations to matrix.

        Parameters
        ----------
        value_list: Iterable[dict[int, int]]
            Iterable of dicts which encode the rows of the feature matrix. Keys: column index, values: column value.
            Each dict represents one molecule.

        Returns
        -------
        sparse.csr_matrix
            Sparse matrix of Morgan-fingerprint features.
        """
        sparse_matrix = sparse_from_index_value_dicts(value_list, self._n_bits)
        if not self._sparse_output:
            sparse_matrix = sparse_matrix.toarray()
        return sparse_matrix

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get object parameters relevant for copying the class.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter names and values.
        """
        parameters = super().get_params(deep)
        if deep:
            parameters["sparse_output"] = copy.copy(self._sparse_output)
        else:
            parameters["sparse_output"] = self._sparse_output

        return parameters

    def set_params(self, parameters: dict[str, Any]) -> Self:
        """Set object parameters relevant for copying the class.

        Parameters
        ----------
        parameters: dict[str, Any]
            Dictionary of parameter names and values.

        Returns
        -------
        Self
            Copied object with updated parameters.
        """
        parameter_dict_copy = dict(parameters)
        _sparse_output = parameter_dict_copy.pop("_sparse_output", None)
        if _sparse_output is not None:
            self._sparse_output = _sparse_output
        super().set_params(parameter_dict_copy)
        return self

    def transform(self, value_list: list[RDKitMol]) -> sparse.csr_matrix:
        """Transform the list of molecules to sparse matrix of Morgan-fingerprint features.

        Parameters
        ----------
        value_list: list[RDKitMol]
            List of RDKit molecules which are transformed to a sparse matrix.

        Returns
        -------
        sparse.csr_matrix
            Sparse matrix of Morgan-fingerprint features.
        """
        return super().transform(value_list)

    @abc.abstractmethod
    def _transform_single(self, value: RDKitMol) -> dict[int, int]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule which is encoded by the fingerprint.

        Returns
        -------
        dict[int, int]
            Dictionary to encode row in matrix. Keys: column index, values: column value.
        """


class ABCMorganFingerprintPipelineElement(MolToFingerprintPipelineElement, abc.ABC):
    """Abstract Class for Morgan fingerprints."""

    # pylint: disable=R0913
    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        sparse_output: bool = True,
        none_handling: Literal["raise", "record_remove"] = "raise",
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
        sparse_output: bool, optional
            True: return sparse matrix, False: return matrix as dense numpy array.
        none_handling: Literal["raise", "record_remove"], optional
            Specifies how to handle None values in the input list.
        name: str
            Name of PipelineElement.
        n_jobs:
            Number of jobs.
        """
        super().__init__(
            sparse_output=sparse_output,
            name=name,
            n_jobs=n_jobs,
            none_handling=none_handling,
        )
        self._use_features = use_features
        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(
                f"Number of bits has to be a positive integer! (Received: {radius})"
            )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get object parameters relevant for copying the class.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter names and values.
        """
        parameters = super().get_params(deep)
        if deep:
            parameters["radius"] = copy.copy(self.radius)
            parameters["use_features"] = copy.copy(self.use_features)
        else:
            parameters["radius"] = self.radius
            parameters["use_features"] = self.use_features

        # remove fill_value from parameters
        parameters.pop("fill_value", None)
        return parameters

    def set_params(self, parameters: dict[str, Any]) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: dict[str, Any]
            Dictionary of parameter names and values.

        Returns
        -------
        Self
            PipelineElement with updated parameters.
        """
        parameter_copy = dict(parameters)
        radius = parameter_copy.pop("radius", None)
        use_features = parameter_copy.pop("use_features", None)

        # explicitly check for None, since 0 is a valid value
        if radius is not None:
            self._radius = parameters["radius"]
        # explicitly check for None, since False is a valid value
        if use_features is not None:
            self._use_features = parameters["use_features"]
        super().set_params(parameter_copy)
        return self

    @property
    def radius(self) -> int:
        """Get radius of Morgan fingerprint."""
        return self._radius

    @property
    def use_features(self) -> bool:
        """Get whether to encode atoms by features or not."""
        return self._use_features

    @abc.abstractmethod
    def _explain_rdmol(self, mol_obj: RDKitMol) -> dict[int, list[tuple[int, int]]]:
        """Get central atom and radius of all features in molecule."""
        raise NotImplementedError

    def bit2atom_mapping(
        self, mol_obj: RDKitMol
    ) -> dict[int, list[CircularAtomEnvironment]]:
        """Obtain set of atoms for all features.

        Parameters
        ----------
        mol_obj: RDKitMol
            RDKit molecule to be encoded.

        Returns
        -------
        dict[int, list[CircularAtomEnvironment]]
            Dictionary with mapping from bit to encoded AtomEnvironments (which contain atom indices).
        """
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
