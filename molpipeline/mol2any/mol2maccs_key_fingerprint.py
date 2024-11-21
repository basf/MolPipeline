"""Implementation of MACCS key fingerprint."""

import numpy as np
from numpy import typing as npt
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import ExplicitBitVect

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    MolToFingerprintPipelineElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToMACCSFP(MolToFingerprintPipelineElement):
    """MACCS key fingerprint.

    The MACCS keys are a set of 166 keys that encode the presence or absence of
    particular substructures in a molecule. The MACCS keys are a subset of the
    PubChem substructure keys.

    """

    _n_bits = 167  # MACCS keys have 166 bits + 1 bit for an all-zero vector (bit 0)
    _feature_names = [f"maccs_{i}" for i in range(_n_bits)]

    def pretransform_single(
        self, value: RDKitMol
    ) -> dict[int, int] | npt.NDArray[np.int_] | ExplicitBitVect:
        """Transform a single molecule to MACCS key fingerprint.

        Parameters
        ----------
        value : RDKitMol
            RDKit molecule.

        Returns
        -------
        dict[int, int] | npt.NDArray[np.int_] | ExplicitBitVect
            MACCS key fingerprint.

        """
        fingerprint = MACCSkeys.GenMACCSKeys(value)
        if self._return_as == "explicit_bit_vect":
            return fingerprint
        if self._return_as == "dense":
            return np.array(fingerprint)
        if self._return_as == "sparse":
            return {idx: 1 for idx in fingerprint.GetOnBits()}
        raise ValueError(f"Unknown return_as value: {self._return_as}")
