"""Abstract classes for transforming rdkit molecules to bit vectors."""

from __future__ import annotations  # for all the python 3.8 users out there.

import abc
import copy
from typing import Any, Iterable, Literal, Optional, overload

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from loguru import logger
from rdkit.DataStructs import ExplicitBitVect
from scipy import sparse

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.utils.matrices import sparse_from_index_value_dicts
from molpipeline.utils.molpipeline_types import RDKitMol
from molpipeline.utils.substructure_handling import CircularAtomEnvironment


class MolToFingerprintPipelineElement(MolToAnyPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""

    _n_bits: int
    _output_type = "binary"
    _output_datatype: Literal["sparse", "dense", "explicit_bit_vect"]

    def __init__(
        self,
        sparse_output: bool | None = None,
        output_datatype: Literal["sparse", "dense", "explicit_bit_vect"] = "sparse",
        name: str = "MolToFingerprintPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ):
        """Initialize abstract class.

        Parameters
        ----------
        sparse_output: bool | None
            DEPRECATED: Will be removed. Use output_type instead.
            True: return sparse matrix, False: return matrix as dense numpy array.
        output_datatype: Literal["sparse", "dense", "explicit_bit_vect"]
            Type of output. When "sparse" the fingerprints will be returned as a scipy.sparse.csr_matrix
            holding a sparse representation of the bit vectors. With "dense" a numpy matrix will be returned.
            With "explicit_bit_vect" the fingerprints will be returned as a list of RDKit's
            rdkit.DataStructs.cDataStructs.ExplicitBitVect.
        name: str
            Name of PipelineElement.
        n_jobs:
            Number of jobs.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        if sparse_output is not None:
            logger.warning(
                "sparse_output is deprecated and will be removed in the future. Use output_type instead."
            )
            if sparse_output:
                output_datatype = "sparse"
            else:
                output_datatype = "dense"
        self._output_datatype = output_datatype

    @property
    def n_bits(self) -> int:
        """Get number of bits in (or size of) fingerprint."""
        return self._n_bits

    @overload
    def assemble_output(
        self, value_list: Iterable[dict[int, int]]
    ) -> sparse.csr_matrix: ...

    @overload
    def assemble_output(
        self, value_list: Iterable[ExplicitBitVect]
    ) -> list[ExplicitBitVect]: ...

    def assemble_output(
        self,
        value_list: (
            Iterable[dict[int, int]]
            | Iterable[npt.NDArray[np.int_]]
            | Iterable[ExplicitBitVect]
        ),
    ) -> sparse.csr_matrix | npt.NDArray[np.int_] | list[ExplicitBitVect]:
        """Transform output of all transform_single operations to matrix.

        Parameters
        ----------
        value_list:  Iterable[dict[int, int]] | Iterable[npt.NDArray[np.int_]] | Iterable[ExplicitBitVect]
            Either Iterable of dicts which encode the rows of the feature matrix.
            Keys: column index, values: column value. Each dict represents one molecule.
            Or an Iterable of RDKit's ExplicitBitVect or an Iterable of numpy arrays.

        Returns
        -------
        sparse.csr_matrix | npt.NDArray[np.int_] | list[ExplicitBitVect]
            Matrix of Morgan-fingerprint features.
        """
        if self._output_datatype == "explicit_bit_vect":
            # return as list of RDkit's ExplicitBitVect or list of numpy arrays
            return list(value_list)  # type: ignore
        if self._output_datatype == "dense":
            # return dense numpy matrix
            return np.vstack(list(value_list))  # type: ignore

        # convert dict representation to csr_matrix
        return sparse_from_index_value_dicts(value_list, self._n_bits)  # type: ignore

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
            parameters["output_datatype"] = copy.copy(self._output_datatype)
        else:
            parameters["output_datatype"] = self._output_datatype

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
        sparse_output = parameter_dict_copy.pop("sparse_output", None)
        output_datatype = parameter_dict_copy.pop("output_datatype", None)
        if sparse_output is not None:
            if sparse_output:
                self._output_datatype = "sparse"
            else:
                self._output_datatype = "dense"
        elif output_datatype is not None:
            self._output_datatype = output_datatype
        super().set_params(parameter_dict_copy)
        return self

    def transform(self, values: list[RDKitMol]) -> sparse.csr_matrix:
        """Transform the list of molecules to sparse matrix of Morgan-fingerprint features.

        Parameters
        ----------
        values: list[RDKitMol]
            List of RDKit molecules which are transformed to a sparse matrix.

        Returns
        -------
        sparse.csr_matrix
            Sparse matrix of Morgan-fingerprint features.
        """
        return super().transform(values)

    @abc.abstractmethod
    def pretransform_single(self, value: RDKitMol) -> dict[int, int] | ExplicitBitVect:
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
        sparse_output: bool | None = None,
        output_datatype: Literal["sparse", "dense", "explicit_bit_vect"] = "sparse",
        name: str = "AbstractMorgan",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
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
        name: str
            Name of PipelineElement.
        n_jobs:
            Number of jobs.
        """
        # pylint: disable=R0801
        super().__init__(
            sparse_output=sparse_output,
            output_datatype=output_datatype,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
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
