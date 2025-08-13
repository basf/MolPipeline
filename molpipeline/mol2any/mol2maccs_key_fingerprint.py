"""Implementation of MACCS key fingerprint."""

import numpy as np
from numpy import typing as npt
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import ExplicitBitVect

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    FPReturnAsOption,
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
        return_as: FPReturnAsOption = "sparse",
        name: str = "MolToMACCSFP",
        n_jobs: int = 1,
        uuid: str | None = None,
    ):
        """Initialize MolToMACCSFP.

        Parameters
        ----------
        return_as : FPReturnAsOption, default="sparse"
            Type of output. When "sparse" the fingerprints will be returned as a
            scipy.sparse.csr_matrix holding a sparse representation of the bit vectors.
            With "dense" a numpy matrix will be returned. With "rdkit" the fingerprints
            will be returned as a list of one of RDKit's ExplicitBitVect,
            IntSparseBitVect, UIntSparseBitVect, etc. depending on the fingerprint
            and parameters.
        name : str, default="MolToMACCSFP"
            Name of PipelineElement.
        n_jobs : int, default=1
            Number of cores to use.
        uuid : str | None, optional
            UUID of the PipelineElement.

        """
        super().__init__(  # pylint: disable=R0801
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self._feature_names = [f"maccs_{i}" for i in range(self._n_bits)]

    def pretransform_single(
        self,
        value: RDKitMol,
    ) -> dict[int, int] | npt.NDArray[np.int_] | ExplicitBitVect:
        """Transform a single molecule to MACCS key fingerprint.

        Parameters
        ----------
        value : RDKitMol
            RDKit molecule.

        Raises
        ------
        ValueError
            If the variable `return_as` is not one of the allowed values.

        Returns
        -------
        dict[int, int] | npt.NDArray[np.int_] | ExplicitBitVect
            MACCS key fingerprint.

        """
        fingerprint = MACCSkeys.GenMACCSKeys(value)  # type: ignore[attr-defined]
        if self._return_as == "rdkit":
            return fingerprint
        if self._return_as == "dense":
            return np.array(fingerprint)
        if self._return_as == "sparse":
            return dict.fromkeys(fingerprint.GetOnBits(), 1)
        raise ValueError(f"Unknown return_as value: {self._return_as}")
