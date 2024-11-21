"""Abstract classes for transforming rdkit molecules to bit vectors."""

from __future__ import annotations  # for all the python 3.8 users out there.

import abc
import copy
from typing import Any, Iterable, Literal, Optional, get_args, overload

try:
    from typing import Self, TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self, TypeAlias

import numpy as np
import numpy.typing as npt
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ExplicitBitVect
from scipy import sparse

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.utils.matrices import sparse_from_index_value_dicts
from molpipeline.utils.molpipeline_types import RDKitMol
from molpipeline.utils.substructure_handling import CircularAtomEnvironment

# possible output types for a fingerprint:
# - "sparse" is a sparse csr_matrix
# - "dense" is a numpy array
# - "explicit_bit_vect" is a list of RDKit's ExplicitBitVect
OutputDatatype: TypeAlias = Literal["sparse", "dense", "explicit_bit_vect"]


class MolToFingerprintPipelineElement(MolToAnyPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""

    _n_bits: int
    _feature_names: list[str]
    _output_type = "binary"
    _return_as: OutputDatatype

    def __init__(
        self,
        return_as: OutputDatatype = "sparse",
        name: str = "MolToFingerprintPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ):
        """Initialize abstract class.

        Parameters
        ----------
        return_as: Literal["sparse", "dense", "explicit_bit_vect"]
            Type of output. When "sparse" the fingerprints will be returned as a scipy.sparse.csr_matrix
            holding a sparse representation of the bit vectors. With "dense" a numpy matrix will be returned.
            With "explicit_bit_vect" the fingerprints will be returned as a list of RDKit's
            rdkit.DataStructs.cDataStructs.ExplicitBitVect.
        name: str
            Name of PipelineElement.
        n_jobs:
            Number of jobs.
        uuid: Optional[str]
            Unique identifier.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self._return_as = return_as

    @property
    def n_bits(self) -> int:
        """Get number of bits in (or size of) fingerprint."""
        return self._n_bits

    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return self._feature_names[:]

    @overload
    def assemble_output(  # type: ignore
        self, value_list: Iterable[npt.NDArray[np.int_]]
    ) -> npt.NDArray[np.int_]: ...

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
            Or an Iterable of RDKit's ExplicitBitVect or an Iterable of numpy arrays representing the
            fingerprint list.

        Returns
        -------
        sparse.csr_matrix | npt.NDArray[np.int_] | list[ExplicitBitVect]
            Matrix of Morgan-fingerprint features.
        """
        if self._return_as == "explicit_bit_vect":
            # return as list of RDkit's ExplicitBitVect
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
            if return_as not in get_args(OutputDatatype):
                raise ValueError(
                    f"return_as has to be one of {get_args(OutputDatatype)}! (Received: {return_as})"
                )
            self._return_as = return_as
        super().set_params(**parameter_dict_copy)
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
    def pretransform_single(
        self, value: RDKitMol
    ) -> dict[int, int] | npt.NDArray[np.int_] | ExplicitBitVect:
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


class MolToRDKitGenFPElement(MolToFingerprintPipelineElement, abc.ABC):
    """Abstract class for PipelineElements using the FingeprintGenerator64."""

    def __init__(
        self,
        counted: bool = False,
        return_as: OutputDatatype = "sparse",
        name: str = "MolToRDKitGenFin",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ):
        """Initialize abstract class.

        Parameters
        ----------
        counted: bool
            Whether to count the bits or not.
        return_as: Literal["sparse", "dense", "explicit_bit_vect"]
            Type of output. When "sparse" the fingerprints will be returned as a scipy.sparse.csr_matrix
        name: str
            Name of PipelineElement.
        n_jobs:
            Number of jobs.
        uuid: Optional[str]
            Unique identifier.
        """
        super().__init__(
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self.counted = counted

    @abc.abstractmethod
    def _get_fp_generator(self) -> rdFingerprintGenerator.FingeprintGenerator64:
        """Get fingerprint generator.

        Returns
        -------
        rdFingerprintGenerator.FingeprintGenerator64
            Fingerprint generator.
        """

    def pretransform_single(
        self, value: RDKitMol
    ) -> ExplicitBitVect | npt.NDArray[np.int_] | dict[int, int]:
        """Transform a single compound to a dictionary.

        Keys denote the feature position, values the count. Here always 1.

        Parameters
        ----------
        value: RDKitMol
            Molecule for which the fingerprint is generated.

        Returns
        -------
        ExplicitBitVect | npt.NDArray[np.int_] | dict[int, int]
            If return_as is "explicit_bit_vect" return ExplicitBitVect.
            If return_as is "dense" return numpy array.
            If return_as is "sparse" return dictionary with feature-position as key and count as value.
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

        if self._return_as == "explicit_bit_vect":
            return fingerprint

        if self.counted:
            return fingerprint.GetNonzeroElements()

        return {pos: 1 for pos in fingerprint.GetOnBits()}

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


class ABCMorganFingerprintPipelineElement(MolToRDKitGenFPElement, abc.ABC):
    """Abstract Class for Morgan fingerprints."""

    # pylint: disable=R0913
    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        counted: bool = False,
        return_as: Literal["sparse", "dense", "explicit_bit_vect"] = "sparse",
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
        counted: bool
            Whether to count the bits or not.
        return_as: Literal["sparse", "dense", "explicit_bit_vect"]
            Type of output. When "sparse" the fingerprints will be returned as a scipy.sparse.csr_matrix
            holding a sparse representation of the bit vectors. With "dense" a numpy matrix will be returned.
            With "explicit_bit_vect" the fingerprints will be returned as a list of RDKit's
            rdkit.DataStructs.cDataStructs.ExplicitBitVect.
        name: str
            Name of PipelineElement.
        n_jobs:
            Number of jobs.
        uuid: Optional[str]
            Unique identifier.
        """
        # pylint: disable=R0801
        super().__init__(
            return_as=return_as,
            counted=counted,
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

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: Any
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
            self._radius = radius
        # explicitly check for None, since False is a valid value
        if use_features is not None:
            self._use_features = bool(use_features)
        super().set_params(**parameter_copy)
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
        """Get central atom and radius of all features in molecule.

        Parameters
        ----------
        mol_obj: RDKitMol
            RDKit molecule to be encoded.

        Returns
        -------
        dict[int, list[tuple[int, int]]]
            Dictionary with mapping from bit to atom index and radius.
        """
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
