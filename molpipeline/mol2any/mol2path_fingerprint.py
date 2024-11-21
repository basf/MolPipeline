"""Implementations for the RDKit Path Fingerprint."""

from __future__ import annotations  # for all the python 3.8 users out there.

from typing import Any, Literal, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy

from rdkit.Chem import rdFingerprintGenerator

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    MolToRDKitGenFPElement,
)


class Mol2PathFP(
    MolToRDKitGenFPElement
):  # pylint: disable=too-many-instance-attributes
    """Folded Path Fingerprint.

    Feature-mapping to vector-positions is arbitrary.

    """

    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    def __init__(
        self,
        min_path: int = 1,
        max_path: int = 7,
        use_hs: bool = True,
        branched_paths: bool = True,
        use_bond_order: bool = True,
        count_simulation: bool = False,
        count_bounds: Any = None,
        n_bits: int = 2048,
        num_bits_per_feature: int = 2,
        atom_invariants_generator: Any = None,
        counted: bool = False,
        return_as: Literal["sparse", "dense", "explicit_bit_vect"] = "sparse",
        name: str = "Mol2PathFP",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize Mol2PathFP.

        Parameters
        ----------
        min_path: int, optional (default=1)
            Minimum path length.
        max_path: int, optional (default=7)
            Maximum path length.
        use_hs: bool, optional (default=True)
            Include hydrogens (If explicit hydrogens are present in the molecule).
        branched_paths: bool, optional (default=True)
            Include branched paths.
        use_bond_order: bool, optional (default=True)
            Include bond order in path.
        count_simulation: bool, optional (default=False)
            Count simulation.
        count_bounds: Any, optional (default=None)
            Set the bins for the bond count.
        n_bits: int, optional (default=2048)
            Size of fingerprint.
        num_bits_per_feature: int, optional (default=2)
            Number of bits per feature.
        atom_invariants_generator: Any, optional (default=None)
            Atom invariants generator.
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
        [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetRDKitFPGenerator
        """
        # pylint: disable=R0801
        super().__init__(
            counted=counted,
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        if not isinstance(n_bits, int) or n_bits < 1:
            raise ValueError(
                f"Number of bits has to be a positive integer, which is > 0! (Received: {n_bits})"
            )
        self._n_bits = n_bits
        self._feature_names = [f"path_{i}" for i in range(self._n_bits)]
        self._min_path = min_path
        self._max_path = max_path
        self._use_hs = use_hs
        self._branched_paths = branched_paths
        self._use_bond_order = use_bond_order
        self._count_simulation = count_simulation
        self._count_bounds = count_bounds
        self._num_bits_per_feature = num_bits_per_feature
        self._atom_invariants_generator = atom_invariants_generator

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
            parameters["min_path"] = int(self._min_path)
            parameters["max_path"] = int(self._max_path)
            parameters["use_hs"] = bool(self._use_hs)
            parameters["branched_paths"] = bool(self._branched_paths)
            parameters["use_bond_order"] = bool(self._use_bond_order)
            parameters["count_simulation"] = bool(self._count_simulation)
            parameters["count_bounds"] = copy.copy(self._count_bounds)
            parameters["num_bits_per_feature"] = int(self._num_bits_per_feature)
            parameters["atom_invariants_generator"] = copy.copy(
                self._atom_invariants_generator
            )
            parameters["n_bits"] = int(self._n_bits)
        else:
            parameters["min_path"] = self._min_path
            parameters["max_path"] = self._max_path
            parameters["use_hs"] = self._use_hs
            parameters["branched_paths"] = self._branched_paths
            parameters["use_bond_order"] = self._use_bond_order
            parameters["count_simulation"] = self._count_simulation
            parameters["count_bounds"] = self._count_bounds
            parameters["num_bits_per_feature"] = self._num_bits_per_feature
            parameters["atom_invariants_generator"] = self._atom_invariants_generator
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
        min_path = parameter_copy.pop("min_path", None)
        if min_path is not None:
            self._min_path = min_path
        max_path = parameter_copy.pop("max_path", None)
        if max_path is not None:
            self._max_path = max_path
        use_hs = parameter_copy.pop("use_hs", None)
        if use_hs is not None:
            self._use_hs = use_hs
        branched_paths = parameter_copy.pop("branched_paths", None)
        if branched_paths is not None:
            self._branched_paths = branched_paths
        use_bond_order = parameter_copy.pop("use_bond_order", None)
        if use_bond_order is not None:
            self._use_bond_order = use_bond_order
        count_simulation = parameter_copy.pop("count_simulation", None)
        if count_simulation is not None:
            self._count_simulation = count_simulation
        count_bounds = parameter_copy.pop("count_bounds", None)
        if count_bounds is not None:
            self._count_bounds = count_bounds
        num_bits_per_feature = parameter_copy.pop("num_bits_per_feature", None)
        if num_bits_per_feature is not None:
            self._num_bits_per_feature = num_bits_per_feature
        atom_invariants_generator = parameter_copy.pop(
            "atom_invariants_generator", None
        )
        if atom_invariants_generator is not None:
            self._atom_invariants_generator = atom_invariants_generator
        n_bits = parameter_copy.pop("n_bits", None)  # pylint: disable=duplicate-code
        if n_bits is not None:
            self._n_bits = n_bits
        super().set_params(**parameter_copy)
        return self

    def _get_fp_generator(self) -> rdFingerprintGenerator.GetRDKitFPGenerator:
        """Get the fingerprint generator for the RDKit path fingerprint.

        Returns
        -------
        rdFingerprintGenerator.GetRDKitFPGenerator
            RDKit Path fingerprint generator.
        """
        return rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=self._min_path,
            maxPath=self._max_path,
            fpSize=self._n_bits,
            useHs=self._use_hs,
            branchedPaths=self._branched_paths,
            useBondOrder=self._use_bond_order,
            countSimulation=self._count_simulation,
            countBounds=self._count_bounds,
            numBitsPerFeature=self._num_bits_per_feature,
            atomInvariantsGenerator=self._atom_invariants_generator,
        )
