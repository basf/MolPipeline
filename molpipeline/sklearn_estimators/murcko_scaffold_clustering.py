"""Murcko scaffold clustering estimator."""
from __future__ import annotations

from numbers import Integral
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.error_handling import ErrorFilter, ErrorReplacer
from molpipeline.pipeline_elements.mol2any import MolToSmilesPipelineElement
from molpipeline.pipeline_elements.mol2mol import (
    EmptyMoleculeFilterPipelineElement,
    MurckoScaffoldPipelineElement,
)
from molpipeline.utils.molpipeline_types import AnyStep, OptionalMol

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


__all__ = [
    "MurckoScaffoldClustering",
]


class MurckoScaffoldClustering(ClusterMixin, BaseEstimator):
    """Murcko scaffold clustering estimator."""

    linear_molecules_strategy: Literal["ignore", "own_cluster"]
    labels_: npt.NDArray[np.int32]
    n_clusters_: int
    _parameter_constraints: dict[str, Any] = {
        "n_jobs": [Integral, None],
    }
    use_smiles: bool

    def __init__(
        self,
        *,
        use_smiles: bool = False,
        n_jobs: int = 1,
        linear_molecules_strategy: Literal["ignore", "own_cluster"] = "ignore",
    ):
        """Initialize Murcko scaffold clustering estimator.

        Parameters
        ----------
        n_jobs : int, optional (default=1)
            Number of jobs to use for parallelization.
        linear_molecules_strategy : Literal["ignore", "own_cluster"], optional (default="ignore")
            Strategy for handling linear molecules. Can be "ignore" or "own_cluster". "ignore" will
            ignore linear molecules and they will be replaced with NaN in the resulting clustering.
            "own_cluster" will instead cluster linear molecules in their own cluster and give them
            a valid cluster label.
        use_smiles : bool, optional (default=False)
            Whether to use SMILES strings as input instead of RDKit molecules.
        """
        self.n_jobs = n_jobs
        self.linear_molecules_strategy = linear_molecules_strategy
        self.use_smiles = use_smiles

    def _generate_pipeline(self) -> Pipeline:
        """Generate the pipeline for the Murcko scaffold clustering estimator.

        Returns
        -------
        Pipeline
            Pipeline for the Murcko scaffold clustering estimator.
        """
        smi2mol = SmilesToMolPipelineElement()
        empty_mol_filter1 = EmptyMoleculeFilterPipelineElement()
        murcko_scaffold = MurckoScaffoldPipelineElement()
        empty_mol_filter2 = EmptyMoleculeFilterPipelineElement()
        mol2smi = MolToSmilesPipelineElement()

        pipeline_step_list: list[AnyStep] = []
        if self.use_smiles:
            pipeline_step_list.append(("smi2mol", smi2mol))
        else:
            pipeline_step_list = []

        pipeline_step_list.extend(
            [
                ("empty_mol_filter1", empty_mol_filter1),
                ("murcko_scaffold", murcko_scaffold),
                ("empty_mol_filter2", empty_mol_filter2),
                ("mol2smi", mol2smi),
            ]
        )

        if self.linear_molecules_strategy == "ignore":
            error_filter = ErrorFilter(filter_everything=True)
            pipeline_step_list.append(("error_filter", error_filter))

        elif self.linear_molecules_strategy == "own_cluster":
            # Create error filter for all errors except empty_mol_filter2
            # This is needed to give linear molecules (empty scaffold) a valid cluster label
            if self.use_smiles:
                filter_ele_list = [smi2mol, empty_mol_filter1, murcko_scaffold, mol2smi]
            else:
                filter_ele_list = [empty_mol_filter1, murcko_scaffold, mol2smi]
            error_filter = ErrorFilter.from_element_list(filter_ele_list)
            pipeline_step_list.append(("error_filter", error_filter))

            # Create and add separate error replacer for murcko_scaffold
            no_scaffold_filter = ErrorFilter.from_element_list([empty_mol_filter2])
            no_scaffold_replacer = ErrorReplacer.from_error_filter(
                no_scaffold_filter, "linear"
            )

            # Directly add the error filter and replacer to the pipeline
            pipeline_step_list.append(("no_scaffold_filter", no_scaffold_filter))
            pipeline_step_list.append(("no_scaffold_replacer", no_scaffold_replacer))
        else:
            raise ValueError(
                f"Invalid value for linear_molecules_strategy: {self.linear_molecules_strategy}"
            )

        error_replacer = ErrorReplacer.from_error_filter(error_filter, np.nan)

        pipeline_step_list.extend(
            [
                ("reshape2d", FunctionTransformer(func=np.atleast_2d)),
                ("transpose", FunctionTransformer(func=np.transpose)),
                ("scaffold_encoder", OrdinalEncoder()),
                ("reshape1d", FunctionTransformer(func=np.ravel)),
                ("error_replacer", error_replacer),
            ]
        )

        cluster_pipeline = Pipeline(
            pipeline_step_list,
            n_jobs=self.n_jobs,
        )
        return cluster_pipeline

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
        cluster_pipeline = self._generate_pipeline()
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
    ) -> npt.NDArray[np.float64]:
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
