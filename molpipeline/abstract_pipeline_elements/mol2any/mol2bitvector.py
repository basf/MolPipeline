"""Abstract classes for transforming rdkit molecules to bit vectors."""

from __future__ import annotations  # for all the python 3.8 users out there.

import abc
import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias, get_args, overload, Mapping

import numpy as np
import numpy.typing as npt
from rdkit.DataStructs import (
    ExplicitBitVect,
    IntSparseIntVect,
    SparseBitVect,
    UIntSparseIntVect,
)
from scipy import sparse

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.utils.matrices import sparse_from_index_value_dicts

if TYPE_CHECKING:
    from rdkit.Chem import rdFingerprintGenerator

    from molpipeline.utils.molpipeline_types import RDKitMol


# possible output types for a fingerprint:
# - "sparse" is a sparse csr_matrix
# - "dense" is a numpy array
# - "rdkit" will return RDKit's native datastructure for the respective fingerprint
#           and it's parameters, for example ExplicitBitVect, IntSparseBitVect,
#           UIntSparseBitVect.
FPReturnAsOption: TypeAlias = Literal[
    "sparse",
    "dense",
    "rdkit",
]

# Return type of pretransform_single method. Each returned value corresponds to a
# fingerprints of a single molecule.
FPTransformSingleReturnDataType: TypeAlias = (
    # sparse dict representation
    dict[int, int]
    # dense numpy array
    | npt.NDArray[np.int_]
    # RDKit's native data structures
    | ExplicitBitVect
    | IntSparseIntVect
    | UIntSparseIntVect
    | SparseBitVect
)

# Input types for assemble_output method. Corresponds to iterable of
# FPTransformSingleReturnDataType.
FPAssembleOutputInputType: TypeAlias = (
    # sparse dict representation
    Iterable[dict[int, int]]
    # dense numpy array
    | Iterable[npt.NDArray[np.int_]]
    # RDKit's native data structures
    | Iterable[ExplicitBitVect]
    | Iterable[UIntSparseIntVect]
    | Iterable[IntSparseIntVect]
    | Iterable[SparseBitVect]
)

# Output types for assemble_output method. Corresponds to fingerprints of multiple
# molecules in a single data structure, e.g. a matrix.
FPAssembleOutputOutputType: TypeAlias = (
    sparse.csr_matrix
    | npt.NDArray[np.int_]
    | list[ExplicitBitVect]
    | list[UIntSparseIntVect]
    | list[IntSparseIntVect]
    | list[SparseBitVect]
)


class MolToFingerprintPipelineElement(MolToAnyPipelineElement, abc.ABC):
    """Abstract PipelineElement which transform molecules to fingerprints."""

    _n_bits: int
    _feature_names: list[str]
    _output_type = "binary"
    _return_as: FPReturnAsOption

    def __init__(
        self,
        return_as: FPReturnAsOption = "sparse",
        name: str = "MolToFingerprintPipelineElement",
        n_jobs: int = 1,
        uuid: str | None = None,
    ):
        """Initialize abstract class.

        Parameters
        ----------
        return_as: FPReturnAsOption
            Type of output. When "sparse" the fingerprints will be returned as a
            scipy.sparse.csr_matrix holding a sparse representation of the bit vectors.
            With "dense" a numpy matrix will be returned. With "rdkit" the fingerprints
            will be returned as a list of one of RDKit's ExplicitBitVect,
            IntSparseBitVect, UIntSparseBitVect, etc. depending on the fingerprint
            and parameters.
        name: str
            Name of PipelineElement.
        n_jobs: int, default=1
            Number of jobs.
        uuid: str | None, optional
            Unique identifier.

        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self._validate_return_as(return_as)
        self._return_as = return_as

    @property
    def n_bits(self) -> int:
        """Get number of bits in (or size of) fingerprint."""
        return self._n_bits

    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return self._feature_names[:]

    @staticmethod
    def _validate_return_as(return_as: str) -> None:
        """Validate return_as parameter.

        Parameters
        ----------
        return_as: str
            Type of output. Has to be one of "sparse", "dense" or "rdkit".

        Raises
        ------
        ValueError
            If return_as is not one of the allowed options.

        """
        if return_as not in get_args(FPReturnAsOption):
            raise ValueError(
                f"return_as has to be one of {get_args(FPReturnAsOption)}! "
                f"(Received: {return_as})",
            )

    @overload
    def assemble_output(  # type: ignore
        self,
        value_list: Iterable[npt.NDArray[np.int_]],
    ) -> npt.NDArray[np.int_]: ...

    @overload
    def assemble_output(
        self,
        value_list: Iterable[dict[int, int]],
    ) -> sparse.csr_matrix: ...

    @overload
    def assemble_output(
        self,
        value_list: Iterable[ExplicitBitVect],
    ) -> list[ExplicitBitVect]: ...

    @overload
    def assemble_output(
        self,
        value_list: Iterable[UIntSparseIntVect],
    ) -> list[UIntSparseIntVect]: ...

    @overload
    def assemble_output(
        self,
        value_list: Iterable[IntSparseIntVect],
    ) -> list[IntSparseIntVect]: ...

    @overload
    def assemble_output(
        self,
        value_list: Iterable[SparseBitVect],
    ) -> list[SparseBitVect]: ...

    def assemble_output(
        self,
        value_list: FPAssembleOutputInputType,
    ) -> FPAssembleOutputOutputType:
        """Transform output of all transform_single operations to matrix.

        Parameters
        ----------
        value_list: FPAssembleOutputInputType
            Iterable of fingerprint representation for a single molecule which are
            transformed to a matrix.

        Returns
        -------
        FPAssembleOutputOutputType
            Matrix of fingerprint features.

        """
        if self._return_as == "rdkit":
            # return as list of RDkit's native data structures, depending on the
            # fingerprint method and whether counted=True or not, this can be
            # ExplicitBitVect, IntSparseBitVect or UIntSparseBitVect, etc.
            return list(value_list)  # type: ignore
        if self._return_as == "dense":
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
            parameters["return_as"] = copy.copy(self._return_as)
        else:
            parameters["return_as"] = self._return_as

        return parameters

    def set_params(self, **parameters: Any) -> Self:
        """Set object parameters relevant for copying the class.

        Parameters
        ----------
        parameters: Any
            Dictionary of parameter names and values.

        Returns
        -------
        Self
            Copied object with updated parameters.

        """
        parameter_dict_copy = dict(parameters)
        return_as = parameter_dict_copy.pop("return_as", None)
        if return_as is not None:
            self._validate_return_as(return_as)
            self._return_as = return_as
        super().set_params(**parameter_dict_copy)
        return self

    def transform(self, values: list[RDKitMol]) -> sparse.csr_matrix:
        """Transform the list of molecules to sparse matrix of fingerprint features.

        Parameters
        ----------
        values: list[RDKitMol]
            List of RDKit molecules which are transformed to a sparse matrix.

        Returns
        -------
        sparse.csr_matrix
            Sparse matrix of fingerprint features.

        """
        return super().transform(values)

    @abc.abstractmethod
    def pretransform_single(
        self,
        value: RDKitMol,
    ) -> FPTransformSingleReturnDataType:
        """Transform mol to fingerprint representation.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule which is encoded by the fingerprint.

        Returns
        -------
        FPTransformSingleReturnDataType
            Fingerprint representation encoding one row in the matrix.

        """


class MolToRDKitGenFPElement(MolToFingerprintPipelineElement, abc.ABC):
    """Abstract class for PipelineElements using the FingeprintGenerator64."""

    @property
    def n_bits(self) -> int:
        """Get number of bits in (or size of) fingerprint."""
        return self._n_bits

    @n_bits.setter
    def n_bits(self, value: int) -> None:
        """Set number of bits in Morgan fingerprint.

        Parameters
        ----------
        value: int
            Number of bits in Morgan fingerprint.

        Raises
        ------
        ValueError
            If value is not a positive integer.

        """
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Number of bits has to be a integer > 0! (Received: {value})",
            )
        self._n_bits = value

    @property
    def output_type(self) -> str:
        """Get output type."""
        if self.counted:
            return "integer"
        return "binary"

    def __init__(
        self,
        n_bits: int = 2048,
        counted: bool = False,
        return_as: FPReturnAsOption = "sparse",
        name: str = "MolToRDKitGenFin",
        n_jobs: int = 1,
        uuid: str | None = None,
    ):
        """Initialize abstract class.

        Parameters
        ----------
        n_bits: int, default=2048
            Number of bits in fingerprint.
        counted: bool, default=False
            Whether to count the bits or not.
        return_as: FPReturnAsOption, default="sparse"
            Type of output.
        name: str, default="MolToRDKitGenFin"
            Name of PipelineElement.
        n_jobs: int, default=1
            Number of jobs.
        uuid: str | None, optional
            Unique identifier.

        """
        super().__init__(
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self.n_bits = n_bits
        self.counted = counted

    @abc.abstractmethod
    def _get_fp_generator(self) -> rdFingerprintGenerator.FingerprintGenerator64:
        """Get fingerprint generator.

        Returns
        -------
        rdFingerprintGenerator.FingerprintGenerator64
            Fingerprint generator.

        """

    def pretransform_single(self, value: RDKitMol) -> FPTransformSingleReturnDataType:
        """Transform a single compound to a fingerprint representation.

        Parameters
        ----------
        value: RDKitMol
            Molecule for which the fingerprint is generated.

        Returns
        -------
        FPTransformSingleReturnDataTypes
            If return_as is "rdkit" return RDKit's data structure.
            If return_as is "dense" return numpy array.
            If return_as is "sparse" return dictionary with feature-position as key
            and count as value.

        """
        fingerprint_generator = self._get_fp_generator()
        if self._return_as == "dense":
            if self.counted:
                return fingerprint_generator.GetCountFingerprintAsNumPy(value)
            return fingerprint_generator.GetFingerprintAsNumPy(value)

        if self.counted:
            fingerprint = fingerprint_generator.GetCountFingerprint(value)
        else:
            fingerprint = fingerprint_generator.GetFingerprint(value)

        if self._return_as == "rdkit":
            return fingerprint

        if self.counted:
            return fingerprint.GetNonzeroElements()

        return dict.fromkeys(fingerprint.GetOnBits(), 1)

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
            parameters["counted"] = bool(self.counted)
        else:
            parameters["counted"] = self.counted

        return parameters

    def set_params(self, **parameters: Any) -> Self:
        """Set object parameters relevant for copying the class.

        Parameters
        ----------
        parameters: Any
            Dictionary of parameter names and values.

        Returns
        -------
        Self
            Copied object with updated parameters.

        """
        parameter_dict_copy = dict(parameters)
        counted = parameter_dict_copy.pop("counted", None)
        if counted is not None:
            self.counted = bool(counted)
        super().set_params(**parameter_dict_copy)
        return self

    @abc.abstractmethod
    def bit2atom_mapping(
        self,
        mol_obj: RDKitMol,
    ) -> Mapping[int, Sequence[AtomEnvironment]]:
        """Obtain set of atoms for all features.

        Parameters
        ----------
        mol_obj: RDKitMol
            RDKit molecule to be encoded.

        Returns
        -------
        Mapping[int, Sequence[AtomEnvironment]]
            Dictionary with mapping from bit to encoded
            AtomEnvironments
            (which contain atom indices).

        """
