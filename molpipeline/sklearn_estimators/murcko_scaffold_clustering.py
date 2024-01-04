"""Murcko scaffold clustering estimator."""
from __future__ import annotations

from numbers import Integral
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from rdkit.Chem import MolFromSmiles
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.error_handling import ErrorFilter, ErrorReplacer
from molpipeline.pipeline_elements.mol2any import MolToSmilesPipelineElement
from molpipeline.pipeline_elements.mol2mol import MurckoScaffoldPipelineElement
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


class _MurckoScaffoldClusteringHelperPipelineElement(MurckoScaffoldPipelineElement):
    """MolToMolPipelineElement which yields the Murcko-scaffold of a Molecule and handles linear molecules."""

    # A string representation for linear molecules
    LINEAR_MOLECULE_DUMMY_SCAFFOLD_MOL: RDKitMol = MolFromSmiles("C")

    def __init__(
        self,
        name: str = "_MurckoScaffoldOrLinearPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MurckoScaffoldPipelineElement.

        Parameters
        ----------
        name: str
            Name of pipeline element.
        n_jobs: int
            Number of jobs to use for parallelization.
        uuid: Optional[str]
            UUID of pipeline element.

        Returns
        -------
        None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Extract Murco-scaffold of molecule.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule object which is transformed.

        Returns
        -------
        OptionalMol
            Murco-scaffold of molecule if possible, else InvalidInstance.
        """
        scaffold_result: RDKitMol = super().pretransform_single(value)
        # if no scaffold could be extracted a Molecule with zero atoms is returned. To avoid this entry
        # being substituted by an InvalidInstance by Molpipeline, we return a dummy non-empty linear molecule.
        if scaffold_result.GetNumAtoms() == 0:
            return (
                _MurckoScaffoldClusteringHelperPipelineElement.LINEAR_MOLECULE_DUMMY_SCAFFOLD_MOL
            )
        return scaffold_result


class MurckoScaffoldClustering(ClusterMixin, BaseEstimator):
    """Murcko scaffold clustering estimator."""

    _parameter_constraints: dict[str, Any] = {
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        *,
        n_jobs: int | None = None,
        linear_molecules_strategy: str = "ignore",
    ) -> None:
        """Initialize Murcko scaffold clustering estimator.

        Parameters
        ----------
        n_jobs : int | None
            Number of jobs to use for parallelization. None means 1.
        linear_molecules_strategy : str
            Strategy for handling linear molecules. Can be "ignore" or "own_cluster". "ignore" will
            ignore linear molecules and they will be replaced with NaN in the resulting clustering.
            "own_cluster" will instead cluster linear molecules in their own cluster and give them
            a valid cluster label.
        """
        self.linear_molecules_strategy: str = linear_molecules_strategy
        self.n_jobs: int = (
            1 if n_jobs is None else n_jobs
        )  # @TODO can we re-use n_jobs handling of sklearn (None will be 1 and -1 using all cores) ?
        self.n_clusters_: int | None = None
        self.labels_: npt.NDArray[np.int32] | None = None

    # pylint: disable=C0103,W0613
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: npt.NDArray[np.str_] | list[str] | list[OptionalMol],
        y: npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Fit Murcko scaffold clustering estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Smiles or molecule list or array.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Self
            Fitted estimator.
        """
        X = self._validate_data(X, ensure_min_samples=2, ensure_2d=False, dtype=None)
        return self._fit(X)

    # pylint: disable=C0103,W0613
    def _fit(
        self,
        X: npt.NDArray[np.str_] | list[str] | list[OptionalMol],
    ) -> Self:
        """Fit Murcko scaffold clustering estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Smiles or molecule list or array.

        Returns
        -------
        Self
            Fitted estimator.
        """
        if self.linear_molecules_strategy == "ignore":
            murcko_scaffold = MurckoScaffoldPipelineElement()
        elif self.linear_molecules_strategy == "own_cluster":
            murcko_scaffold = _MurckoScaffoldClusteringHelperPipelineElement()
        else:
            raise ValueError("Invalid linear_molecules_strategy")

        none_filter = ErrorFilter(
            {murcko_scaffold.uuid},
            filter_everything=True,
            name="ErrorFilter",
            n_jobs=1,
            uuid=None,
        )
        none_filler = ErrorReplacer.from_error_filter(
            none_filter,
            None,
        )

        cluster_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("murcko_scaffold", murcko_scaffold),
                ("mol2smi", MolToSmilesPipelineElement()),
                ("none_filter", none_filter),
                ("reshape2d", FunctionTransformer(func=np.atleast_2d)),
                ("transpose", FunctionTransformer(func=np.transpose)),
                ("scaffold_encoder", OrdinalEncoder()),
                ("reshape1d", FunctionTransformer(func=np.ravel)),
                ("none_filler", none_filler),
            ],
            n_jobs=self.n_jobs,
        )

        self.labels_ = cluster_pipeline.fit_transform(X, None)
        if self.labels_ is None:
            raise AssertionError("self.labels_ can not be None")
        self.n_clusters_ = len(np.unique(self.labels_[~np.isnan(self.labels_)]))
        return self

    # pylint: disable=C0103,W0613
    def fit_predict(
        self,
        X: npt.NDArray[np.str_] | list[str] | list[OptionalMol],
        y: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.int32]:
        """Fit and predict Murcko scaffold clustering estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Smiles or molecule list or array.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        np.ndarray[int]
            Cluster labels.
        """
        # pylint: disable=W0246
        return super().fit_predict(X, y)
