"""Implementations for the Morgan fingerprint."""

from __future__ import annotations  # for all the python 3.8 users out there.

from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import sparse

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    ABCMorganFingerprintPipelineElement,
)


class Mol2FoldedMorganFingerprint(ABCMorganFingerprintPipelineElement):
    """Folded Morgan Fingerprint.

    Feature-mapping to vector-positions is arbitrary.

    """

    # pylint: disable=R0913
    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        n_bits: int = 2048,
        name: str = "Mol2FoldedMorganFingerprint",
        n_jobs: int = 1,
    ) -> None:
        """Initialize Mol2FoldedMorganFingerprint.

        Parameters
        ----------
        radius: int
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool
            Instead of atoms, features are encoded in the fingerprint. [2]
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
        super().__init__(
            radius=radius, use_features=use_features, name=name, n_jobs=n_jobs
        )
        if isinstance(n_bits, int) and n_bits >= 0:
            self._n_bits = n_bits
        else:
            raise ValueError(
                f"Number of bits has to be a positive integer! (Received: {n_bits})"
            )

    def _transform_single(self, value: Chem.Mol) -> dict[int, int]:
        """Transform a single compound to a dictionary.

        Keys denote the featreu position, values the count. Here always 1.
        """
        fingerprint_vector = AllChem.GetMorganFingerprintAsBitVect(
            value, self.radius, useFeatures=self._use_features, nBits=self._n_bits
        )
        return {bit: 1 for bit in fingerprint_vector.GetOnBits()}

    def _explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
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


class Mol2UnfoldedMorganFingerprint(ABCMorganFingerprintPipelineElement):
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

    _bit_mapping: dict[int, int]
    _counted: bool

    # pylint: disable=R0913
    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        counted: bool = False,
        ignore_unknown: bool = False,
        name: str = "Mol2UnfoldedMorganFingerprint",
        n_jobs: int = 1,
    ):
        """Initialize Mol2UnfoldedMorganFingerprint.

        Parameters
        ----------
        counted: bool
            False: bits are binary: on if present in molecule, off if not present
            True: bits are positive integers and give the occurrence of their respective features in the molecule
        radius: int
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool
            Instead of atoms, features are encoded in the fingerprint. [2]
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores to use.

        References
        ----------
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
            [2] https://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        """
        super().__init__(
            radius=radius, use_features=use_features, name=name, n_jobs=n_jobs
        )

        if not isinstance(counted, bool):
            raise TypeError("The argument 'counted' must be a bool!")
        self._counted = counted

        if not isinstance(ignore_unknown, bool):
            raise TypeError("The argument 'ignore_unknown' must be a bool!")
        self.ignore_unknown = ignore_unknown

    @property
    def counted(self) -> bool:
        """Return whether the fingerprint is counted, or not."""
        return self._counted

    def _explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        """Get central atom and radius of all features in molecule."""
        original_bit_info: dict[int, list[tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprint(
            mol_obj,
            self.radius,
            useFeatures=self.use_features,
            bitInfo=original_bit_info,
        )
        bit_info = {self._bit_mapping[k]: v for k, v in original_bit_info.items()}
        return bit_info

    def fit(self, value_list: list[Chem.Mol]) -> None:
        """Determine all features and assign each a unique position in the fingerprint-vector."""
        _ = self._fit(value_list)

    def fit_transform(self, value_list: list[Chem.Mol]) -> sparse.csr_matrix:
        """Create a feature mapping based on input and apply it for transformation."""
        hash_count_list = self._fit(value_list)
        mapped_feature_count_dicts = [
            self._map_feature_dict(f_dict) for f_dict in hash_count_list
        ]
        return self.collect_rows(mapped_feature_count_dicts)

    def _transform_single(self, value: Chem.Mol) -> dict[int, int]:
        """Return a dict, where the key is the feature-position and the value is the count."""
        feature_count_dict = self._pretransform_single(value)
        bit_count_dict = self._map_feature_dict(feature_count_dict)
        return bit_count_dict

    def _create_mapping(self, feature_hash_dict_list: Iterable[dict[int, int]]) -> None:
        """Create a mapping from feature hash to bit position."""
        unraveled_features = [
            f for f_dict in feature_hash_dict_list for f in f_dict.keys()
        ]
        feature_hash, count = np.unique(unraveled_features, return_counts=True)
        feature_hash_count_dict = dict(zip(feature_hash, count))
        unique_features = set(unraveled_features)
        feature_order = sorted(
            unique_features, key=lambda f: (feature_hash_count_dict[f], f), reverse=True
        )
        self._bit_mapping = dict(
            {feature: idx for idx, feature in enumerate(feature_order)}
        )
        self._n_bits = len(self._bit_mapping)

    def _fit(self, mol_obj_list: list[Chem.Mol]) -> list[dict[int, int]]:
        hash_count_list = [
            self._pretransform_single(mol_obj) for mol_obj in mol_obj_list
        ]
        self._create_mapping(hash_count_list)
        return hash_count_list

    def _map_feature_dict(self, feature_count_dict: dict[int, int]) -> dict[int, int]:
        """Transform a dict of feature hash and occurrence to a dict of bit-pos and occurence.

        Parameters
        ----------
        feature_count_dict: dict[int, int]

        Returns
        -------
        dict[int, int]
        """
        mapped_count_dict = {}
        for feature_hash, feature_count in feature_count_dict.items():
            bit_position = self._bit_mapping.get(feature_hash)
            if bit_position is None:
                if self.ignore_unknown:
                    continue
                raise KeyError(
                    f"This feature hash did not occur during training: {feature_hash}"
                )
            mapped_count_dict[bit_position] = feature_count
        return mapped_count_dict

    def _pretransform_single(self, mol: Chem.Mol) -> dict[int, int]:
        """Return a dict, where the key is the feature-hash and the value is the count."""
        morgan_features = AllChem.GetMorganFingerprint(
            mol, self.radius, useFeatures=self.use_features
        )
        morgan_feature_count_dict: dict[int, int] = morgan_features.GetNonzeroElements()
        if not self.counted:
            morgan_feature_count_dict = {
                f_hash: 1 for f_hash in morgan_feature_count_dict
            }
        return morgan_feature_count_dict
