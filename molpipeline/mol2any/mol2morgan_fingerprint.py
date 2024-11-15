"""Implementations for the Morgan fingerprint."""

from __future__ import annotations  # for all the python 3.8 users out there.

from typing import Any, Literal, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy

from rdkit.Chem import AllChem, rdFingerprintGenerator

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
            counted=counted,
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        if not isinstance(n_bits, int) or n_bits < 1:
            raise ValueError(
                f"Number of bits has to be a positve integer, which is > 0! (Received: {n_bits})"
            )
        self._n_bits = n_bits
        self._feature_names = [f"morgan_{i}" for i in range(self._n_bits)]

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

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: Any
            Dictionary of parameter names and values.

        Returns
        -------
        Self
            MolToMorganFP pipeline element with updated parameters.
        """
        parameter_copy = dict(parameters)
        n_bits = parameter_copy.pop("n_bits", None)
        if n_bits is not None:
            self._n_bits = n_bits
        super().set_params(**parameter_copy)

        return self

    def _get_fp_generator(self) -> rdFingerprintGenerator.FingerprintGenerator:
        """Get the fingerprint generator.

        Returns
        -------
        rdFingerprintGenerator.FingerprintGenerator
            RDKit fingerprint generator.
        """
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self._n_bits,
        )

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
        fp_generator = self._get_fp_generator()
        additional_output = AllChem.AdditionalOutput()
        additional_output.AllocateBitInfoMap()
        # using the dense fingerprint here, to get indices after folding
        _ = fp_generator.GetFingerprint(mol_obj, additionalOutput=additional_output)
        bit_info = additional_output.GetBitInfoMap()
        return bit_info
