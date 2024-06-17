"""Implementations for the Morgan fingerprint."""

from __future__ import annotations  # for all the python 3.8 users out there.

from typing import Any, Literal, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy

import numpy as np
import numpy.typing as npt
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.DataStructs import ExplicitBitVect

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    ABCMorganFingerprintPipelineElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToMorganFP(ABCMorganFingerprintPipelineElement):
    """Folded Morgan Fingerprint.

    Feature-mapping to vector-positions is arbitrary.

    """

    # pylint: disable=R0913
    def __init__(
        self,
        radius: int = 2,
        use_features: bool = False,
        n_bits: int = 2048,
        counted: bool = False,
        return_as: Literal["sparse", "dense", "explicit_bit_vect"] = "sparse",
        name: str = "MolToMorganFP",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToMorganFP.

        Parameters
        ----------
        radius: int, optional (default=2)
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool, optional (default=False)
            Instead of atoms, features are encoded in the fingerprint. [2]
        n_bits: int, optional (default=2048)
            Size of fingerprint.
        counted: bool, optional (default=False)
            If True, the fingerprint will be counted.
            If False, the fingerprint will be binary.
        return_as: Literal["sparse", "dense", "explicit_bit_vect"]
            Type of output. When "sparse" the fingerprints will be returned as a scipy.sparse.csr_matrix
            holding a sparse representation of the bit vectors. With "dense" a numpy matrix will be returned.
            With "explicit_bit_vect" the fingerprints will be returned as a list of RDKit's
            rdkit.DataStructs.cDataStructs.ExplicitBitVect.
        name: str, optional (default="MolToMorganFP")
            Name of PipelineElement
        n_jobs: int, optional (default=1)
            Number of cores to use.
        uuid: str | None, optional (default=None)
            UUID of the PipelineElement.

        References
        ----------
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
            [2] https://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        """
        # pylint: disable=R0801
        super().__init__(
            radius=radius,
            use_features=use_features,
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self.counted = counted
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
            MolToMorganFP pipeline element with updated parameters.
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
        if self._return_as == "explicit_bit_vect":
            if self.counted:
                return fingerprint_generator.GetCountFingerprint(value)
            return fingerprint_generator.GetFingerprint(value)

        if self.counted:
            fingerprint = fingerprint_generator.GetCountFingerprintAsNumPy(value)
        else:
            fingerprint = fingerprint_generator.GetFingerprintAsNumPy(value)

        if self._return_as == "dense":
            return fingerprint

        return {pos: count for pos, count in enumerate(fingerprint) if count > 0}

    def _explain_rdmol(self, mol_obj: RDKitMol) -> dict[int, list[tuple[int, int]]]:
        """Get central atom and radius of all features in molecule.

        Parameters
        ----------
        mol_obj: RDKitMol
            RDKit molecule object

        Returns
        -------
        dict[int, list[tuple[int, int]]]
            Dictionary with bit position as key and list of tuples with atom index and radius as value.
        """
        bit_info: dict[int, list[tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol_obj,
            self.radius,
            useFeatures=self._use_features,
            bitInfo=bit_info,
            nBits=self._n_bits,
        )
        return bit_info
