"""Implementations for the Morgan fingerprint."""

from __future__ import annotations  # for all the python 3.8 users out there.

from typing import TYPE_CHECKING, Any, Literal, Self

from rdkit.Chem import rdFingerprintGenerator

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    MolToRDKitGenFPElement,
)
from molpipeline.utils.substructure_handling import CircularAtomEnvironment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from molpipeline.utils.molpipeline_types import RDKitMol


class MolToMorganFP(MolToRDKitGenFPElement):
    """Folded Morgan Fingerprint.

    Feature-mapping to vector-positions is arbitrary.

    """

    _radius: int
    _use_features: bool

    @property
    def radius(self) -> int:
        """Get radius of Morgan fingerprint."""
        return self._radius

    @radius.setter
    def radius(self, value: int) -> None:
        """Set radius of Morgan fingerprint.

        Parameters
        ----------
        value: int
            Radius of Morgan fingerprint.

        Raises
        ------
        ValueError
            If value is not a positive integer.

        """
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Radius has to be a positive integer! (Received: {value})",
            )
        self._radius = value

    @property
    def use_features(self) -> bool:
        """Get whether to encode atoms by features or not."""
        return self._use_features

    @use_features.setter
    def use_features(self, value: bool) -> None:
        """Set whether to encode atoms by features or not.

        Parameters
        ----------
        value: bool
            Whether to encode atoms by features or not.

        Raises
        ------
        ValueError
            If value is not a boolean.

        """
        if not isinstance(value, bool):
            raise ValueError(f"Use features has to be a boolean! (Received: {value})")
        self._use_features = value

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
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToMorganFP.

        Parameters
        ----------
        radius: int, default=2
            radius of the circular fingerprint [1].
            Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool, optional (default=False)
            Instead of atoms, features are encoded in the fingerprint. [2]
        n_bits: int, default=2048
            Size of fingerprint.
        counted: bool, default=False
            If True, the fingerprint will be counted.
            If False, the fingerprint will be binary.
        return_as: Literal["sparse", "dense", "explicit_bit_vect"], default="sparse"
            Type of output. When "sparse" the fingerprints will be returned as a
            scipy.sparse.csr_matrix holding a sparse representation of the bit vectors.
            With "dense" a numpy matrix will be returned.
            With "explicit_bit_vect" the fingerprints will be returned as a list of
            RDKit's rdkit.DataStructs.cDataStructs.ExplicitBitVect.
        name: str, default="MolToMorganFP"
            Name of PipelineElement
        n_jobs: int, default=1
            Number of cores to use.
        uuid: str | None, optional
            UUID of the PipelineElement.

        References
        ----------
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
            [2] https://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints

        """
        # pylint: disable=R0801
        super().__init__(
            n_bits=n_bits,
            counted=counted,
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self.use_features = use_features
        self.radius = radius
        self._feature_names = [f"morgan_{i}" for i in range(self.n_bits)]

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
        parameters["n_bits"] = self.n_bits
        parameters["radius"] = self.radius
        parameters["use_features"] = self.use_features
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
        if "n_bits" in parameter_copy:
            self.n_bits = parameter_copy.pop("n_bits")
        if "radius" in parameter_copy:
            self.radius = parameter_copy.pop("radius")
        if "use_features" in parameter_copy:
            self.use_features = parameter_copy.pop("use_features")
        super().set_params(**parameter_copy)

        return self

    def _get_fp_generator(
        self,
    ) -> rdFingerprintGenerator.FingerprintGenerator64:
        """Get the fingerprint generator.

        Returns
        -------
        rdFingerprintGenerator.FingerprintGenerator
            RDKit fingerprint generator.

        """
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
        )

    def bit2atom_mapping(
        self,
        mol_obj: RDKitMol,
    ) -> Mapping[int, Sequence[CircularAtomEnvironment]]:
        """Get central atom and radius of all features in molecule.

        Parameters
        ----------
        mol_obj: RDKitMol
            RDKit molecule object

        Returns
        -------
        Mapping[int, list[tuple[int, int]]]
            Dictionary with bit position as key and list of tuples with atom index and
            radius as value.

        """
        bit2atom_dict = self._get_bit_info_map(mol_obj)
        result_dict: dict[int, list[CircularAtomEnvironment]] = {}
        # Iterating over all present bits and respective matches
        for bit, matches in bit2atom_dict.items():
            result_dict[bit] = []
            for central_atom, radius in matches:
                env = CircularAtomEnvironment.from_mol(mol_obj, central_atom, radius)
                result_dict[bit].append(env)
        return result_dict
