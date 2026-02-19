"""Baseline estimators for molecular property prediction."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import (
    MolToConcatenatedVector,
    MolToMorganFP,
    MolToRDKitPhysChem,
)
from molpipeline.utils.molpipeline_types import AnyStep


def _make_physchem_morgan_feature_pipeline_elements() -> list[AnyStep]:
    """Make pipeline elements for concatenated PhysChem and Morgan count features.

    Returns
    -------
    list[AnyStep]
        A list of pipeline elements for generating concatenated PhysChem and Morgan
        count features.

    """
    return [
        ("auto2mol", AutoToMol()),  # reading molecules
        (
            "morgan_physchem",
            MolToConcatenatedVector(
                [
                    (
                        "RDKitPhysChem",
                        MolToRDKitPhysChem(),
                    ),
                    (
                        "MorganFP",
                        MolToMorganFP(
                            n_bits=2048,
                            radius=2,
                            return_as="dense",
                            counted=True,
                        ),
                    ),
                ],
            ),
        ),
    ]


def _make_rf_baseline_pipeline(
    rf_type: type[RandomForestClassifier] | type[RandomForestRegressor],
    n_jobs: int = 1,
    random_state: int | None = None,
    error_handling: bool = False,
) -> Pipeline:
    """Create a Random Forest pipeline with baseline configuration.

    Parameters
    ----------
    rf_type: type[RandomForestClassifier] | type[RandomForestRegressor]
        The type of Random Forest model to use (classifier or regressor).
    n_jobs: int, default=1
        Number of parallel jobs to use.
    random_state: int | None, optional
        Random seed for reproducibility.
    error_handling : bool, default=False
        Whether to include automated error handling in the pipeline.

    Returns
    -------
        Pipeline
            A pipeline with a Random Forest model with baseline configuration.

    """
    featurization_elems = _make_physchem_morgan_feature_pipeline_elements()

    # remove zero-variance features. Usually, doesn't really improve accuracy
    # but makes training faster.
    variance_filter = ("variance_filter", VarianceThreshold(0.0))

    model = (
        "model",
        rf_type(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            max_features="log2",
        ),
    )

    elements: list[AnyStep]
    if error_handling:
        error_filter = ("error_filter_feature", ErrorFilter(filter_everything=True))
        error_reinserter = (
            "error_reinserter",
            PostPredictionWrapper(
                FilterReinserter.from_error_filter(error_filter[1], np.nan),
            ),
        )
        elements = [
            *featurization_elems,
            error_filter,
            variance_filter,
            model,
            error_reinserter,
        ]
    else:
        elements = [*featurization_elems, variance_filter, model]

    return Pipeline(
        elements,
        n_jobs=n_jobs,
    )


def get_rf_classifier_baseline(
    n_jobs: int = 1,
    random_state: int | None = None,
    error_handling: bool = False,
) -> Pipeline:
    """Get a Random Forest classifier with an experience-based baseline configuration.

    The pipeline:
    - Use n_estimators=500 instead of sklearn's default n_estimators=100.
    - Use Morgan count fingerprint instead of commonly used binary fingerprint.
    - Concatenates Morgan count fingerprints with RDKit's PhysChem descriptors, which
      ofter work orthogonally to fingerprints.
    - Based on our experience, minimal hparam tuning of Random Forests's max_features
      can help to improve accuracy a little bit and speed up training significantly.
      In this configuration, we set: max_features="log2".

    Parameters
    ----------
    n_jobs: int, default=1
        Number of parallel jobs to use.
    random_state: int | None, optional
        Random seed for reproducibility.
    error_handling : bool, default=False
        Whether to include automated error handling in the pipeline.

    Returns
    -------
        Pipeline
            A pipeline with a Random Forest Classifier with baseline configuration.

    """
    return _make_rf_baseline_pipeline(
        RandomForestClassifier,
        n_jobs,
        random_state,
        error_handling,
    )


def get_rf_regressor_baseline(
    n_jobs: int = 1,
    random_state: int | None = None,
    error_handling: bool = False,
) -> Pipeline:
    """Get a Random Forest regressor with an experience-based baseline configuration.

    The pipeline:
    - Use n_estimators=500 instead of sklearn's default n_estimators=100.
    - Use Morgan count fingerprint instead of commonly used binary fingerprint.
    - Concatenates Morgan count fingerprints with RDKit's PhysChem descriptors, which
      ofter work orthogonally to fingerprints.
    - Based on our experience, minimal hparam tuning of Random Forests's max_features
      can help to improve accuracy a little bit and speed up training significantly.
      In this configuration, we set: max_features="log2".

    Parameters
    ----------
    n_jobs: int, default=1
        Number of parallel jobs to use.
    random_state: int | None, optional
        Random seed for reproducibility.
    error_handling : bool, default=False
        Whether to include automated error handling in the pipeline.

    Returns
    -------
        Pipeline
            A pipeline with a Random Forest Regressor with baseline configuration.

    """
    return _make_rf_baseline_pipeline(
        RandomForestRegressor,
        n_jobs,
        random_state,
        error_handling,
    )
