"""Implementations for the Morgan fingerprint."""

from __future__ import annotations  # for all the python 3.8 users out there.

from typing import Any, Iterable, Literal, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy

import numpy as np
import numpy.typing as npt
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.DataStructs import ExplicitBitVect
from scipy import sparse

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    ABCMorganFingerprintPipelineElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToFoldedMorganFingerprint(ABCMorganFingerprintPipelineElement):
    """Folded Morgan Fingerprint.

    Feature-mapping to vector-positions is arbitrary.

    """

    # pylint: disable=R0913
    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        n_bits: int = 2048,
        sparse_output: bool | None = None,
        output_datatype: Literal["sparse", "dense", "explicit_bit_vect"] = "sparse",
        name: str = "Mol2FoldedMorganFingerprint",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize Mol2FoldedMorganFingerprint.

        Parameters
        ----------
        radius: int
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool
            Instead of atoms, features are encoded in the fingerprint. [2]
        sparse_output: bool | None
            DEPRECATED: Will be removed. Use output_type instead.
            True: return sparse matrix, False: return matrix as dense numpy array.
        output_datatype: Literal["sparse", "dense", "explicit_bit_vect"]
            Type of output. When "sparse" the fingerprints will be returned as a scipy.sparse.csr_matrix
            holding a sparse representation of the bit vectors. With "dense" a numpy matrix will be returned.
            With "explicit_bit_vect" the fingerprints will be returned as a list of RDKit's
            rdkit.DataStructs.cDataStructs.ExplicitBitVect.
        n_bits: int
            Size of fingerprint.
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores to use.

        References
        ----------
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
            [2] https://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        """
        # pylint: disable=R0801
        super().__init__(
            radius=radius,
            use_features=use_features,
            sparse_output=sparse_output,
            output_datatype=output_datatype,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        if isinstance(n_bits, int) and n_bits >= 0:
            self._n_bits = n_bits
        else:
            raise ValueError(
                f"Number of bits has to be a positive integer! (Received: {n_bits})"
            )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all parameters defining the object.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameters.
        """
        parameters = super().get_params(deep)
        if deep:
            parameters["n_bits"] = copy.copy(self._n_bits)
        else:
            parameters["n_bits"] = self._n_bits
        return parameters

    def set_params(self, **parameters: dict[str, Any]) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: dict[str, Any]
            Dictionary of parameter names and values.

        Returns
        -------
        Self
            MolToFoldedMorganFingerprint pipeline element with updated parameters.
        """
        parameter_copy = dict(parameters)
        n_bits = parameter_copy.pop("n_bits", None)
        if n_bits is not None:
            self._n_bits = n_bits  # type: ignore
        super().set_params(**parameter_copy)

        return self

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
        dict[int, int]
            Dictionary with feature-position as key and count as value.
        """
        fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self._n_bits,
        )

        if self._output_datatype == "explicit_bit_vect":
            return fingerprint_generator.GetFingerprint(value)
        if self._output_datatype == "dense":
            return fingerprint_generator.GetFingerprintAsNumPy(value)
        # sparse return type
        return {
            bit_idx: 1
            for bit_idx in fingerprint_generator.GetFingerprint(value).GetOnBits()
        }

    def _explain_rdmol(self, mol_obj: RDKitMol) -> dict[int, list[tuple[int, int]]]:
        """Get central atom and radius of all features in molecule."""
        bit_info: dict[int, list[tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol_obj,
            self.radius,
            useFeatures=self._use_features,
            bitInfo=bit_info,
            nBits=self._n_bits,
        )
        return bit_info


class MolToUnfoldedMorganFingerprint(ABCMorganFingerprintPipelineElement):
    """Transforms smiles-strings or molecular objects into unfolded bit-vectors based on Morgan-fingerprints [1].

    Features are mapped to bits based on the amount of molecules they occur in.

    Long version
    ------------
        Circular fingerprints do not have a unique mapping to a bit-vector, therefore the features are mapped to the
        vector according to the number of molecules they occur in. The most occurring feature is mapped to bit 0, the
        second most feature to bit 1 and so on...

        Weak-point: features not seen in the fit method are not mappable to the bit-vector and therefore cause an error.

    References
    ----------
        [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    """

    _bit_mapping: Optional[dict[int, int]]
    _counted: bool

    # pylint: disable=R0913
    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        counted: bool = False,
        sparse_output: bool = True,
        ignore_unknown: bool = False,
        name: str = "Mol2UnfoldedMorganFingerprint",
        n_jobs: int = 1,
    ) -> None:
        """Initialize Mol2UnfoldedMorganFingerprint.

        Parameters
        ----------
        radius: int
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool
            Instead of atoms, features are encoded in the fingerprint. [2]
        counted: bool
            False: bits are binary: on if present in molecule, off if not present
            True: bits are positive integers and give the occurrence of their respective features in the molecule
        sparse_output: bool
            True: return sparse matrix, False: return matrix as dense numpy array.
        ignore_unknown: bool
            If True, features not seen in the fit method are ignored. If False, they cause an error.
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores to use.

        Returns
        -------
        None

        References
        ----------
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
            [2] https://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        """
        super().__init__(
            radius=radius,
            use_features=use_features,
            sparse_output=sparse_output,
            name=name,
            n_jobs=n_jobs,
        )

        if not isinstance(counted, bool):
            raise TypeError("The argument 'counted' must be a bool!")
        self._counted = counted

        if not isinstance(ignore_unknown, bool):
            raise TypeError("The argument 'ignore_unknown' must be a bool!")
        self.ignore_unknown = ignore_unknown
        self._bit_mapping = None
        self._requires_fitting = True

    @property
    def additional_attributes(self) -> dict[str, Any]:
        """Return additional attributes of the element."""
        additional_attributes = super().additional_attributes
        if self._bit_mapping is not None:
            additional_attributes["bit_mapping"] = {
                int(k): int(v) for k, v in self._bit_mapping.items()
            }
        return additional_attributes

    @property
    def bit_mapping(self) -> dict[int, int]:
        """Return the bit mapping.

        Raises
        ------
        ValueError
            If the fingerprint has not been fitted yet.
        """
        if self._bit_mapping is None:
            raise ValueError(
                "The fingerprint has not been fitted yet. Please call the fit method first."
            )
        return self._bit_mapping

    @bit_mapping.setter
    def bit_mapping(self, bit_mapping: dict[int, int]) -> None:
        """Set the bit mapping.

        Parameters
        ----------
        bit_mapping: dict[int, int]
            Mapping for feature hashes to bit-positions.

        Returns
        -------
        None
        """
        self._bit_mapping = bit_mapping

    @property
    def _n_bits(self) -> int:
        """Return the number of bits."""
        return len(self.bit_mapping)

    @_n_bits.setter
    def _n_bits(self, n_bits: int) -> None:
        """Set the number of bits.

        Parameters
        ----------
        n_bits: int

        Returns
        -------
        None
        """
        raise AttributeError("The number of bits is defined by the bit mapping.")

    @property
    def counted(self) -> bool:
        """Return whether the fingerprint is counted, or not."""
        return self._counted

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get all parameters defining the object.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameters defining the object.
        """
        parameters = super().get_params(deep)
        if deep:
            parameters["counted"] = copy.copy(self._counted)
            parameters["ignore_unknown"] = copy.copy(self.ignore_unknown)
        else:
            parameters["counted"] = self.counted
            parameters["ignore_unknown"] = self.ignore_unknown
        return parameters

    def set_params(self, **parameters: Any) -> Self:
        """Set all parameters defining the object.

        Parameters
        ----------
        parameters: dict[str, Any]
            Dictionary of parameter names and values to be set.

        Returns
        -------
        Self
            MolToUnfoldedMorganFingerprint pipeline element with updated parameters.
        """
        super().set_params(**parameters)
        if "counted" in parameters:
            self._counted = parameters["counted"]
        if "ignore_unknown" in parameters:
            self.ignore_unknown = parameters["ignore_unknown"]
        if "bit_mapping" in parameters:
            self.bit_mapping = parameters["bit_mapping"]
        return self

    def _explain_rdmol(self, mol_obj: RDKitMol) -> dict[int, list[tuple[int, int]]]:
        """Get central atom and radius of all features in molecule."""
        original_bit_info: dict[int, list[tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprint(
            mol_obj,
            self.radius,
            useFeatures=self.use_features,
            bitInfo=original_bit_info,
        )
        bit_info = {self.bit_mapping[k]: v for k, v in original_bit_info.items()}
        return bit_info

    def fit(
        self,
        values: list[RDKitMol],
        labels: Any = None,  # pylint: disable=unused-argument
    ) -> Self:
        """Determine all features and assign each a unique position in the fingerprint-vector.

        Parameters
        ----------
        values: list[RDKitMol]
            List of molecules used for fitting.
        labels: Any
            Labels for the molecules. Not used.

        Returns
        -------
        Self
            The fitted MolToUnfoldedMorganFingerprint pipeline element.
        """
        _ = self._fit(values)
        return self

    def fit_transform(
        self,
        values: list[RDKitMol],
        labels: Any = None,  # pylint: disable=unused-argument
    ) -> sparse.csr_matrix:
        """Create a feature mapping based on input and apply it for transformation.

        Parameters
        ----------
        values: list[RDKitMol]
            List of molecules to fit and transform.
        labels: Any
            Labels for the molecules. Not used.

        Returns
        -------
        sparse.csr_matrix
            Fingerprint-matrix of shape (len(value_list), n_features).
        """
        hash_count_list = self._fit(values)
        mapped_feature_count_dicts = [
            self._map_feature_dict(f_dict) for f_dict in hash_count_list
        ]
        return self.assemble_output(mapped_feature_count_dicts)

    def _create_mapping(self, feature_hash_dict_list: Iterable[dict[int, int]]) -> None:
        """Create a mapping from feature hash to bit position.

        Parameters
        ----------
        feature_hash_dict_list: Iterable[dict[int, int]]
            List of feature hash dicts from molecules presented during fitting.

        Returns
        -------
        None
        """
        unraveled_features = [
            f for f_dict in feature_hash_dict_list for f in f_dict.keys()
        ]
        feature_hash, count = np.unique(unraveled_features, return_counts=True)
        feature_hash_count_dict = dict(zip(feature_hash, count))
        unique_features = set(unraveled_features)
        feature_order = sorted(
            unique_features, key=lambda f: (feature_hash_count_dict[f], f), reverse=True
        )
        self.bit_mapping = dict(
            {feature: idx for idx, feature in enumerate(feature_order)}
        )

    def _fit(self, mol_obj_list: list[RDKitMol]) -> list[dict[int, int]]:
        """Transform and return all molecules to feature hashes, create mapping from obtained hashes.

        Parameters
        ----------
        mol_obj_list: list[RDKitMol]
            List of molecules presented during fitting which are used to create the mapping.

        Returns
        -------
        list[dict[int, int]]
            List of feature hash dicts for input molecules.
        """
        hash_count_list = [
            self.pretransform_single(mol_obj) for mol_obj in mol_obj_list
        ]
        self.fit_to_result(hash_count_list)
        return hash_count_list

    def fit_to_result(self, values: Any) -> Self:
        """Fit to the result of the output of the pretransform step.

        Parameters
        ----------
        values: Any
            Output of the pretransform method.

        Returns
        -------
        Self
            The fitted MolToUnfoldedMorganFingerprint pipeline element.
        """
        self._create_mapping(values)
        return self

    def _map_feature_dict(self, feature_count_dict: dict[int, int]) -> dict[int, int]:
        """Transform a dict of feature hash and occurrence to a dict of bit-pos and occurrence.

        Parameters
        ----------
        feature_count_dict: dict[int, int]
            A dictionary with feature-hash and count.

        Returns
        -------
        dict[int, int]
            A dictionary with bit-position for given feature hashes and respective counts as values.
        """
        mapped_count_dict = {}
        for feature_hash, feature_count in feature_count_dict.items():
            bit_position = self.bit_mapping.get(feature_hash)
            if bit_position is None:
                if self.ignore_unknown:
                    continue
                raise KeyError(
                    f"This feature hash did not occur during training: {feature_hash}"
                )
            mapped_count_dict[bit_position] = feature_count
        return mapped_count_dict

    def pretransform_single(self, value: RDKitMol) -> dict[int, int]:
        """Return a dict, where the key is the feature-hash and the value is the count.

        Parameters
        ----------
        value: RDKitMol
            Molecule for which features are derived.

        Returns
        -------
        dict[int, int]
            A dictionary with feature-hash and count.
        """
        morgan_features = AllChem.GetMorganFingerprint(
            value, self.radius, useFeatures=self.use_features
        )
        morgan_feature_count_dict: dict[int, int] = morgan_features.GetNonzeroElements()
        if not self.counted:
            morgan_feature_count_dict = {
                f_hash: 1 for f_hash in morgan_feature_count_dict
            }
        return morgan_feature_count_dict

    def finalize_single(self, value: Any) -> Any:
        """Finalize single value. Here: map hashes to bit pos.

        Parameters
        ----------
        value: Any
            Single value to be finalized.

        Returns
        -------
        Any
            Finalized value.
        """
        bit_count_dict = self._map_feature_dict(value)
        return bit_count_dict
