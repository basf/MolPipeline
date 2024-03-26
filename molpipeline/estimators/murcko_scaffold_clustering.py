"""Murcko scaffold clustering estimator."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from molpipeline import ErrorFilter, FilterReinserter, Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToSmiles
from molpipeline.mol2mol import EmptyMoleculeFilter, MakeScaffoldGeneric, MurckoScaffold
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
        make_generic: bool = False,
        n_jobs: int = 1,
        linear_molecules_strategy: Literal["ignore", "own_cluster"] = "ignore",
    ):
        """Initialize Murcko scaffold clustering estimator.

        Parameters
        ----------
        make_generic : bool (default=False)
            Makes a Murcko scaffold generic (i.e. all atom types->C and all bonds->single).
        n_jobs : int, optional (default=1)
            Number of jobs to use for parallelization.
        linear_molecules_strategy : Literal["ignore", "own_cluster"], optional (default="ignore")
            Strategy for handling linear molecules. Can be "ignore" or "own_cluster". "ignore" will
            ignore linear molecules, and they will be replaced with NaN in the resulting clustering.
            "own_cluster" will instead cluster linear molecules in their own cluster and give them
            a valid cluster label.
        """
        self.n_jobs = n_jobs
        self.linear_molecules_strategy = linear_molecules_strategy
        self.make_generic = make_generic

    def _generate_pipeline(self) -> Pipeline:
        """Generate the pipeline for the Murcko scaffold clustering estimator.

        Returns
        -------
        Pipeline
            Pipeline for the Murcko scaffold clustering estimator.
        """
        auto2mol = AutoToMol()
        empty_mol_filter1 = EmptyMoleculeFilter()
        murcko_scaffold = MurckoScaffold()
        empty_mol_filter2 = EmptyMoleculeFilter()
        mol2smi = MolToSmiles()

        pipeline_step_list: list[AnyStep] = [
            ("auto2mol", auto2mol),
            ("empty_mol_filter1", empty_mol_filter1),
            ("murcko_scaffold", murcko_scaffold),
        ]

        scaffold_generic_elem: MakeScaffoldGeneric | None = None
        if self.make_generic:
            scaffold_generic_elem = MakeScaffoldGeneric()
            pipeline_step_list.append(("make_scaffold_generic", scaffold_generic_elem))

        pipeline_step_list.extend(
            [
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
            filter_ele_list = [auto2mol, empty_mol_filter1, murcko_scaffold]
            if scaffold_generic_elem is not None:
                filter_ele_list.append(scaffold_generic_elem)
            filter_ele_list.append(mol2smi)
            error_filter = ErrorFilter.from_element_list(filter_ele_list)
            pipeline_step_list.append(("error_filter", error_filter))

            # Create and add separate error replacer for murcko_scaffold
            no_scaffold_filter = ErrorFilter.from_element_list([empty_mol_filter2])
            no_scaffold_replacer = FilterReinserter.from_error_filter(
                no_scaffold_filter, "linear"
            )

            # Directly add the error filter and replacer to the pipeline
            pipeline_step_list.append(("no_scaffold_filter", no_scaffold_filter))
            pipeline_step_list.append(("no_scaffold_replacer", no_scaffold_replacer))
        else:
            raise ValueError(
                f"Invalid value for linear_molecules_strategy: {self.linear_molecules_strategy}"
            )

        error_replacer = FilterReinserter.from_error_filter(error_filter, np.nan)

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
        **params: Any,
    ) -> Self:
        """Fit Murcko scaffold clustering estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Smiles or molecule list or array.
        y : Ignored
            Not used, present for API consistency by convention.
        **params : Any
            Additional keyword arguments.

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

    def fit_predict(
        self,
        X: (
            npt.NDArray[np.str_] | list[str] | list[OptionalMol]
        ),  # pylint: disable=C0103
        y: npt.NDArray[np.float64] | None = None,
        **params: Any,  # pylint: disable=unused-argument
    ) -> npt.NDArray[np.float64]:
        """Fit and predict Murcko scaffold clustering estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Smiles or molecule list or array.
        y : Ignored
            Not used, present for API consistency by convention.
        **params : Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray[int]
            Cluster labels.
        """
        # pylint: disable=W0246
        return super().fit_predict(X, y, **params)
