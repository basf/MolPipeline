"""Implementation of MACCS key fingerprint."""

from typing import Literal

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

    def __init__(
        self,
        return_as: Literal["sparse", "dense", "explicit_bit_vect"] = "sparse",
        name: str = "MolToMACCS",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToMACCS.

        Parameters
        ----------
        return_as: Literal["sparse", "dense", "explicit_bit_vect"], optional (default="sparse")
            Type of output. When "sparse" the fingerprints will be returned as a
            scipy.sparse.csr_matrix holding a sparse representation of the bit vectors.
            With "dense" a numpy matrix will be returned.
            With "explicit_bit_vect" the fingerprints will be returned as a list of RDKit's
            rdkit.DataStructs.cDataStructs.ExplicitBitVect.
        name: str, optional (default="MolToMACCS")
            Name of PipelineElement
        n_jobs: int, optional (default=1)
            Number of cores to use.
        uuid: str | None, optional (default=None)
            UUID of the PipelineElement.

        """
        super().__init__(return_as=return_as, name=name, n_jobs=n_jobs, uuid=uuid)

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
