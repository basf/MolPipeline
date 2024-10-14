"""This module contains the default models used for testing molpipeline functions and classes."""

from sklearn.ensemble import RandomForestClassifier

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.mol2any import (
    MolToConcatenatedVector,
    MolToMorganFP,
    MolToRDKitPhysChem,
)
from molpipeline.post_prediction import PostPredictionWrapper


def get_morgan_physchem_rf_pipeline() -> Pipeline:
    """Get a pipeline combining Morgan fingerprints and physicochemical properties with a RandomForestClassifier.

    Returns
    -------
    Pipeline
        A pipeline combining Morgan fingerprints and physicochemical properties with a RandomForestClassifier.
    """
    error_filter = ErrorFilter(filter_everything=True)
    pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMol()),
            (
                "mol2fp",
                MolToConcatenatedVector(
                    [
                        ("morgan", MolToMorganFP(n_bits=2048)),
                        ("physchem", MolToRDKitPhysChem()),
                    ]
                ),
            ),
            ("error_filter", error_filter),
            ("rf", RandomForestClassifier()),
            (
                "filter_reinserter",
                PostPredictionWrapper(
                    FilterReinserter.from_error_filter(error_filter, None)
                ),
            ),
        ],
        n_jobs=1,
    )
    return pipeline
