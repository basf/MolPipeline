"""Implementation for 2D pharmacophore fingerprints."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import numpy.typing as npt
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdMolChemicalFeatures import (
    BuildFeatureFactoryFromString,
)
from rdkit.DataStructs import ConvertToExplicit, ExplicitBitVect, IntSparseIntVect
from rdkit.RDPaths import RDDataDir

from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
    FPReturnAsOption,
    MolToFingerprintPipelineElement,
)

if TYPE_CHECKING:
    from molpipeline.utils.molpipeline_types import RDKitMol

_MIN_POINT_COUNT = 2


class MolToPharmacophore2DFP(  # pylint: disable=too-many-instance-attributes
    MolToFingerprintPipelineElement,
):
    """2D Pharmacophore Fingerprint.

    Computes RDKit's 2D pharmacophore fingerprints based on pharmacophore features
    and their 2D distances. The fingerprint encodes the presence of pharmacophore
    feature pairs at specific distance ranges. Distances are determined on the
    molecular graph, not in 3D space.

    This implementation uses RDKit's Pharm2D module which generates fingerprints
    based on:
    - Feature definitions (e.g., donors, acceptors, aromatic rings)
    - Distance bins for feature pairs
    - Configurable parameters for feature factory and signature factory

    Per default, the 2d pharmacophore fingerprint described by Gobbi et al. is used.

    References
    ----------
    Gobbi, A. & Poppinger, D. Genetic optimization of combinatorial libraries.
        Biotechnology and Bioengineering 61, 47-54 (1998).

    """

    def __init__(
        self,
        *,
        feature_definition: str | None = None,
        min_point_count: int = 2,
        max_point_count: int = 3,
        triangular_pruning: bool = True,
        shortest_paths_only: bool = True,
        include_bond_order: bool = False,
        skip_feats: list[str] | None = None,
        distance_bins: list[tuple[float, float]] | None = None,
        counted: bool = False,
        return_as: FPReturnAsOption = "sparse",
        name: str = "MolToPharmacophore2DFP",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToPharmacophore2DFP.

        Parameters
        ----------
        feature_definition : str | None, optional
            Path or content of a feature definition file (.fdef). If None, uses
            configuration by Gobbi et al.
        min_point_count : int, default=2
            Minimum number of pharmacophore points in a signature.
        max_point_count : int, default=3
            Maximum number of pharmacophore points in a signature.
        triangular_pruning : bool, default=True
            Whether to use triangular pruning for efficiency.
        shortest_paths_only : bool, default=True
            Whether to use only shortest paths between features.
        include_bond_order : bool, default=False
            Whether to include bond order information in the fingerprint.
        skip_feats : list[str], optional
            List of feature types to skip. If None, no features are skipped.
        distance_bins : list[tuple[float, float]], optional
            List of distance bins as (min_distance, max_distance) tuples.
            If None, uses default bins by Gobbi et al.
        counted : bool, default=False
            If True, the fingerprint will be counted (values represent occurrence).
            If False, the fingerprint will be binary (values are 0 or 1).
        return_as : Literal["sparse", "dense", "rdkit"], default="sparse"
            Type of output. When "sparse" the fingerprints will be returned as a
            scipy.sparse.csr_matrix holding a sparse representation of the bit vectors.
            With "dense" a numpy matrix will be returned.
            With "rdkit" the fingerprints will be returned as a list of
            RDKit's internal data structure. ExplicitBitVect if counted is False,
            IntSparseIntVect if counted is True.
        name : str, default="MolToPharmacophore2DFP"
            Name of PipelineElement.
        n_jobs : int, default=1
            Number of cores to use.
        uuid : str, optional
            UUID of the PipelineElement.

        """
        super().__init__(  # pylint: disable=R0801
            return_as=return_as,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )

        self._validate_min_max_point_count(min_point_count, max_point_count)

        self._min_point_count = min_point_count
        self._max_point_count = max_point_count
        self._triangular_pruning = triangular_pruning
        self._shortest_paths_only = shortest_paths_only
        self._include_bond_order = include_bond_order
        self._skip_feats = skip_feats if skip_feats is not None else []
        self._counted = counted
        self._feature_definition = feature_definition or Gobbi_Pharm2D.fdef
        self._distance_bins = distance_bins or Gobbi_Pharm2D.defaultBins

        self._validate_distance_bins(self._distance_bins)

        # Initialize factories and calculate fingerprint size
        self._initialize_factories()

    def _initialize_factories(self) -> None:
        """Initialize the feature factory and signature factory.

        Raises
        ------
        ValueError
            If the signature factory does not produce any bits.

        """
        feature_factory = BuildFeatureFactoryFromString(self._feature_definition)

        # Create signature factory
        self._sig_factory = SigFactory(
            feature_factory,
            minPointCount=self._min_point_count,
            maxPointCount=self._max_point_count,
            trianglePruneBins=self._triangular_pruning,
            shortestPathsOnly=self._shortest_paths_only,
            includeBondOrder=self._include_bond_order,
            skipFeats=self._skip_feats,
            useCounts=self._counted,
        )

        # Set distance bins
        self._sig_factory.SetBins(self._distance_bins)

        # Initialize the signature factory to get the fingerprint size
        self._sig_factory.Init()
        self._n_bits = self._sig_factory.GetSigSize()

        if self._n_bits == 0:
            raise ValueError(
                "The signature factory did not produce any bits. "
                "Check the feature definition and parameters.",
            )

        # Generate feature names
        self._feature_names = [
            f"pharm2d_{self._sig_factory.GetBitDescription(i)}"
            for i in range(self._n_bits)
        ]

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the object for pickling.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the object.

        """
        state = self.__dict__.copy()
        # Remove the SigFactory from the state to avoid pickling issues
        state.pop("_sig_factory", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object after unpickling.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the object.

        """
        self.__dict__.update(state)
        # Reinitialize the signature factory after unpickling
        self._initialize_factories()

    @staticmethod
    def _validate_min_max_point_count(
        min_point_count: int,
        max_point_count: int,
    ) -> None:
        """Validate the minimum and maximum point count.

        Parameters
        ----------
        min_point_count : int
            Minimum point count.
        max_point_count : int
            Maximum point count.

        Raises
        ------
        ValueError
            If min_point_count is less than 2 or greater than max_point_count.
            If max_point_count is less than 2.

        """
        if min_point_count < _MIN_POINT_COUNT:
            raise ValueError(
                f"Minimum point count must be at least {_MIN_POINT_COUNT}.",
            )
        if min_point_count > max_point_count:
            raise ValueError(
                "Minimum point count cannot be greater than maximum point count.",
            )

    @staticmethod
    def _validate_distance_bins(
        distance_bins: list[tuple[float, float]] | None,
    ) -> None:
        """Validate distance bins format and values.

        Parameters
        ----------
        distance_bins : list[tuple[float, float]] | None
            List of distance bins as (min_distance, max_distance) tuples.

        Raises
        ------
        ValueError
            If any distance bin does not contain numeric values,
            if min_distance is not less than max_distance,
            or if min_distance is negative.

        """
        if distance_bins is None:
            return
        for i, (min_dist, max_dist) in enumerate(distance_bins):
            if not isinstance(min_dist, (int, float)) or not isinstance(
                max_dist,
                (int, float),
            ):
                raise ValueError(f"Distance bin {i} must contain numeric values")
            if min_dist >= max_dist:
                raise ValueError(
                    f"Distance bin {i}: min_dist must be less than max_dist",
                )
            if min_dist < 0:
                raise ValueError(f"Distance bin {i}: distances must be non-negative")

    @property
    def min_point_count(self) -> int:
        """Get minimum point count."""
        return self._min_point_count

    @property
    def max_point_count(self) -> int:
        """Get maximum point count."""
        return self._max_point_count

    @property
    def triangular_pruning(self) -> bool:
        """Get triangular pruning setting."""
        return self._triangular_pruning

    @property
    def shortest_paths_only(self) -> bool:
        """Get shortest paths only setting."""
        return self._shortest_paths_only

    @property
    def include_bond_order(self) -> bool:
        """Get include bond order setting."""
        return self._include_bond_order

    @property
    def skip_feats(self) -> list[str]:
        """Get list of features to skip."""
        return self._skip_feats[:]

    @property
    def feature_definition(self) -> str:
        """Get feature factory file path."""
        return self._feature_definition

    @property
    def distance_bins(self) -> list[tuple[float, float]]:
        """Get distance bins."""
        return self._distance_bins[:]

    @property
    def counted(self) -> bool:
        """Get counted setting."""
        return self._counted

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all parameters defining the object.

        Parameters
        ----------
        deep : bool, default=True
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameters.

        """
        parameters = super().get_params(deep)
        if deep:
            parameters.update(
                {
                    "feature_definition": copy.copy(self._feature_definition),
                    "min_point_count": copy.copy(self._min_point_count),
                    "max_point_count": copy.copy(self._max_point_count),
                    "triangular_pruning": copy.copy(self._triangular_pruning),
                    "shortest_paths_only": copy.copy(self._shortest_paths_only),
                    "include_bond_order": copy.copy(self._include_bond_order),
                    "skip_feats": copy.deepcopy(self._skip_feats),
                    "distance_bins": copy.deepcopy(self._distance_bins),
                    "counted": copy.copy(self._counted),
                },
            )
        else:
            parameters.update(
                {
                    "feature_definition": self._feature_definition,
                    "min_point_count": self._min_point_count,
                    "max_point_count": self._max_point_count,
                    "triangular_pruning": self._triangular_pruning,
                    "shortest_paths_only": self._shortest_paths_only,
                    "include_bond_order": self._include_bond_order,
                    "skip_feats": self._skip_feats,
                    "distance_bins": self._distance_bins,
                    "counted": self._counted,
                },
            )
        return parameters

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters : Any
            Dictionary of parameter names and values.

        Returns
        -------
        Self
            MolToPharmacophore2DFP pipeline element with updated parameters.

        """
        parameter_copy = dict(parameters)

        # Set parameters that require reinitialization
        needs_reinit = False

        if "feature_definition" in parameter_copy:
            self._feature_definition = parameter_copy.pop("feature_definition")
            needs_reinit = True

        if "min_point_count" in parameter_copy:
            min_point_count = parameter_copy.pop("min_point_count")
            self._validate_min_max_point_count(min_point_count, self._max_point_count)
            self._min_point_count = min_point_count
            needs_reinit = True

        if "max_point_count" in parameter_copy:
            max_point_count = parameter_copy.pop("max_point_count")
            self._validate_min_max_point_count(self._min_point_count, max_point_count)
            self._max_point_count = max_point_count
            needs_reinit = True

        if "triangular_pruning" in parameter_copy:
            self._triangular_pruning = parameter_copy.pop("triangular_pruning")
            needs_reinit = True

        if "shortest_paths_only" in parameter_copy:
            self._shortest_paths_only = parameter_copy.pop("shortest_paths_only")
            needs_reinit = True

        if "include_bond_order" in parameter_copy:
            self._include_bond_order = parameter_copy.pop("include_bond_order")
            needs_reinit = True

        if "skip_feats" in parameter_copy:
            self._skip_feats = parameter_copy.pop("skip_feats")
            needs_reinit = True

        if "distance_bins" in parameter_copy:
            distance_bins = parameter_copy.pop("distance_bins")
            self._validate_distance_bins(distance_bins)
            self._distance_bins = distance_bins
            needs_reinit = True

        if "counted" in parameter_copy:
            self._counted = parameter_copy.pop("counted")
            needs_reinit = True

        # Reinitialize factories if needed
        if needs_reinit:
            self._initialize_factories()

        # Set remaining parameters via parent class
        super().set_params(**parameter_copy)

        return self

    def pretransform_single(
        self,
        value: RDKitMol,
    ) -> ExplicitBitVect | IntSparseIntVect | npt.NDArray[np.int_] | dict[int, int]:
        """Transform a single compound to a pharmacophore fingerprint.

        Parameters
        ----------
        value : RDKitMol
            Molecule for which the pharmacophore fingerprint is generated.

        Returns
        -------
        ExplicitBitVect | IntSparseIntVect | npt.NDArray[np.int_] | dict[int, int]
            If return_as is "rdkit" return ExplicitBitVect when counted=False, else
            IntSparseIntVect.
            If return_as is "dense" return numpy array.
            If return_as is "sparse" return dictionary with feature-position as key
            and count as value.

        """
        # fp will be of type `SparseBitVect` when counted is False, otherwise
        # `IntSparseIntVect`.
        fp = Gen2DFingerprint(value, self._sig_factory)
        if self._return_as == "dense":
            return np.array(fp.ToList())
        if self._return_as == "rdkit":
            if self.counted:
                return fp  # return IntSparseIntVect
            # Convert SparseBitVect to ExplicitBitVect
            return ConvertToExplicit(fp)

        # return sparse representation as dictionary
        if self.counted:
            return fp.GetNonzeroElements()
        return dict.fromkeys(fp.GetOnBits(), 1)

    @classmethod
    def from_file(cls, path: Path | str, **kwargs: Any) -> Self:
        """Create a MolToPharmacophore2DFP instance from a feature definition file.

        Parameters
        ----------
        path : Path | str
            Path to the feature definition file (.fdef).
        **kwargs : Any
            Additional parameters to the MolToPharmacophore2DFP constructor.

        Returns
        -------
        MolToPharmacophore2DFP
            Instance of MolToPharmacophore2DFP with the specified feature definition.

        Raises
        ------
        TypeError
            If the path is not a Path or str.

        """
        if not isinstance(path, (Path, str)):
            raise TypeError("path must be a Path or str.")
        return cls(feature_definition=Path(path).read_text(encoding="utf-8"), **kwargs)

    @classmethod
    def from_preconfiguration(
        cls,
        config_name: Literal["base", "gobbi"],
        **kwargs: Any,
    ) -> Self:
        """Create a preconfigured MolToPharmacophore2DFP instance.

        Preconfigurations:
        - "gobbi": Uses Gobbi's pharmacophore features as defined in:
           Gobbi, A. & Poppinger, D. Genetic optimization of combinatorial libraries.
           Biotechnology and Bioengineering 61, 47-54 (1998).
        - "base": Base configuration defined in RDKit.
           Warning: While this configuration is directly available in RDKit, it is
           unclear how stable it is and how well it was evaluated.
           See Greg's talk from RDKIT UGM 2012 slide 17
           https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf


        Parameters
        ----------
        config_name : Literal["gobbi"]
            Name of the preconfiguration to use.
        **kwargs : Any
            Additional parameters to the MolToPharmacophore2DFP constructor.

        Returns
        -------
        MolToPharmacophore2DFP
            Preconfigured MolToPharmacophore2DFP instance.

        Raises
        ------
        ValueError
            If the configuration name is unknown.

        """
        if config_name == "gobbi":
            # gobbi pharmacophore features are also implemented in RDKit. We just
            # borrow the definition here from the Gobbi_Pharm2D module.
            return cls(
                feature_definition=Gobbi_Pharm2D.fdef,
                min_point_count=2,
                max_point_count=3,
                distance_bins=Gobbi_Pharm2D.defaultBins,
                **kwargs,
            )
        if config_name == "base":
            # RDKit's "base" configuration
            # Feature definitions are taken from RDKit's BaseFeatures.fdef file.
            # Bins are taken from:
            # https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf
            return cls.from_file(
                Path(RDDataDir) / "BaseFeatures.fdef",
                distance_bins=[
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 100),
                ],
                **kwargs,
            )

        raise ValueError(f"Unknown configuration name: {config_name}")
