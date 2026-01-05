"""Baseline estimators for molecular property prediction."""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import (
    MolToConcatenatedVector,
    MolToMorganFP,
    MolToRDKitPhysChem,
)
from molpipeline.utils.molpipeline_types import AnyStep


def _get_physchem_morgan_feature_pipeline_elements() -> list[AnyStep]:
    """Get pipeline elements for concatenated PhysChem and Morgan count features.

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
                        MolToRDKitPhysChem(
                            standardizer=None,  # we avoid standardization at this point
                        ),
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


def get_rf_classifier_baseline(
    n_jobs: int = 1,
    random_state: int | None = None,
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

    Returns
    -------
        Pipeline
            A pipeline with a Random Forest Classifier with baseline configuration.

    """
    return Pipeline(
        [
            *_get_physchem_morgan_feature_pipeline_elements(),
            # remove zero-variance features. Usually, doesn't really improve accuracy
            # but makes training faster.
            ("variance_filter", VarianceThreshold(0.0)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    max_features="log2",
                ),
            ),
        ],
        n_jobs=n_jobs,
    )


def get_rf_regressor_baseline(
    n_jobs: int = 1,
    random_state: int | None = None,
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

    Returns
    -------
        Pipeline
            A pipeline with a Random Forest Classifier with baseline configuration.

    """
    return Pipeline(
        [
            *_get_physchem_morgan_feature_pipeline_elements(),
            # remove zero-variance features. Usually, doesn't really improve accuracy
            # but makes training faster.
            ("variance_filter", VarianceThreshold(0.0)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    max_features="log2",
                ),
            ),
        ],
        n_jobs=n_jobs,
    )
