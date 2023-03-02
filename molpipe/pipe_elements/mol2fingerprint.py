import abc
from typing import Iterable, Optional

import multiprocessing
import numpy as np
from scipy import sparse

from rdkit import Chem
from rdkit.Chem import AllChem

from molpipe.pipe_elements.abstract_pipe import Mol2Fingerprint
from molpipe.utils.substructure_handling import AtomEnvironment, CircularAtomEnvironment


class _BaseMorganFingerprint(Mol2Fingerprint):
    def __init__(self, radius: int = 2, use_features: bool = False):
        super().__init__()
        self._use_features = use_features
        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {radius})")

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def use_features(self) -> bool:
        return self._use_features

    @abc.abstractmethod
    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        raise NotImplementedError

    def bit2atom_mapping(self, mol_obj: Chem.Mol) -> dict[int, list[CircularAtomEnvironment]]:
        bit2atom_dict = self.explain_rdmol(mol_obj)
        result_dict: dict[int, list[CircularAtomEnvironment]] = dict()
        # Iterating over all present bits and respective matches
        for bit, matches in bit2atom_dict.items():  # type: int, list[tuple[int, int]]
            result_dict[bit] = []
            for central_atom, radius in matches:  # type: int, int
                env = CircularAtomEnvironment.from_mol(mol_obj, central_atom, radius)
                result_dict[bit].append(env)
        # Transforming default dict to dict
        return result_dict


class Mol2FoldedMorganFingerprint(_BaseMorganFingerprint):
    def __init__(self, radius: int = 2, use_features: bool = False, n_bits: int = 2048):
        super().__init__(radius=radius, use_features=use_features)
        if isinstance(n_bits, int) and n_bits >= 0:
            self._n_bits = n_bits
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {n_bits})")

    def fit(self, mol_obj_list: list[Chem.Mol]) -> None:
        pass

    def transform_single(self, mol: Chem.Mol) -> dict[int, int]:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, useFeatures=self._use_features, nBits=self._n_bits
        )
        return {bit: 1 for bit in fp.GetOnBits()}

    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        bi: dict[int, list[tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol_obj, self.radius, useFeatures=self._use_features, bitInfo=bi, nBits=self._n_bits
        )
        return bi


class Mol2UnfoldedMorganFingerprint(_BaseMorganFingerprint):
    """Transforms smiles-strings or molecular objects into unfolded bit-vectors based on Morgan-fingerprints [1].
    Features are mapped to bits based on the amount of molecules they occur in.

    Long version:
        Circular fingerprints do not have a unique mapping to a bit-vector, therefore the features are mapped to the
        vector according to the number of molecules they occur in. The most occurring feature is mapped to bit 0, the
        second most feature to bit 1 and so on...

        Weak-point: features not seen in the fit method are not mappable to the bit-vector and therefore cause an error.

    References:
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    """

    _bit_mapping: Optional[dict[int, int]]
    _counted: bool

    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        counted: bool = False,
        ignore_unknown: bool = False,
    ):
        """Initializes the class
        Parameters
        ----------
        counted: bool
            False: bits are binary: on if present in molecule, off if not present
            True: bits are positive integers and give the occurrence of their respective features in the molecule
        radius: int
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool
            Instead of atoms, features are encoded in the fingerprint. [2]

        References
        ----------
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
            [2] https://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        """
        super().__init__(radius=radius, use_features=use_features)
        self._bit_mapping = None

        if not isinstance(counted, bool):
            raise TypeError("The argument 'counted' must be a bool!")
        self._counted = counted

        if not isinstance(ignore_unknown, bool):
            raise TypeError("The argument 'ignore_unknown' must be a bool!")
        self.ignore_unknown = ignore_unknown

    @property
    def counted(self) -> bool:
        """Returns the bool value for enabling counted fingerprint."""
        return self._counted

    @property
    def bit_mapping(self) -> dict[int, int]:
        if self._bit_mapping is None:
            raise AttributeError("Attribute not set. Please call fit first.")
        return self._bit_mapping.copy()

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        bi: dict[int, list[tuple[int, int]]] = dict()
        _ = AllChem.GetMorganFingerprint(
            mol_obj, self.radius, useFeatures=self.use_features, bitInfo=bi
        )
        bit_info = {self.bit_mapping[k]: v for k, v in bi.items()}
        return bit_info

    def fit(self, mol_obj_list: list[Chem.Mol]) -> None:
        _ = self._fit(mol_obj_list)

    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        hash_count_list = self._fit(mol_obj_list)
        mapped_feature_count_dicts = [self._map_feature_dict(f_dict) for f_dict in hash_count_list]
        return self.collect_singles(mapped_feature_count_dicts)

    def transform_single(self, mol: Chem.Mol) -> dict[int, int]:
        """Return a dict, where the key is the feature-position and the value is the count."""
        feature_count_dict = self._pretransform_single(mol)
        bit_count_dict = self._map_feature_dict(feature_count_dict)
        return bit_count_dict

    def _create_mapping(self, feature_hash_dict_list: Iterable[dict[int, int]]) -> None:
        """Create a mapping from feature hash to bit position."""
        unraveled_features = [f for f_dict in feature_hash_dict_list for f in f_dict.keys()]
        feature_hash, count = np.unique(unraveled_features, return_counts=True)
        feature_hash_count_dict = dict(zip(feature_hash, count))
        unique_features = set(unraveled_features)
        feature_order = sorted(
            unique_features, key=lambda f: (feature_hash_count_dict[f], f), reverse=True
        )
        self._bit_mapping = dict({feature: idx for idx, feature in enumerate(feature_order)})
        self._n_bits = len(self._bit_mapping)

    def _fit(self, mol_obj_list: list[Chem.Mol]) -> list[dict[int, int]]:
        hash_count_list = [self._pretransform_single(mol_obj) for mol_obj in mol_obj_list]
        self._create_mapping(hash_count_list)
        return hash_count_list

    def _map_feature_dict(self, feature_count_dict: dict[int, int]) -> dict[int, int]:
        """ Transform a dict of feature hash and occurrence to a dict of bit-pos and occurence.

        Parameters
        ----------
        feature_count_dict: dict[int, int]

        Returns
        -------
        dict[int, int]
        """
        mapped_count_dict = {}
        for feature_hash, feature_count in feature_count_dict.items():
            bit_position = self.bit_mapping.get(feature_hash)
            if bit_position is None:
                if self.ignore_unknown:
                    continue
                else:
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
            morgan_feature_count_dict = {f_hash: 1 for f_hash in morgan_feature_count_dict}
        return morgan_feature_count_dict

