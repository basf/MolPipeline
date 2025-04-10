"""This module contains the default models used for testing molpipeline functions and classes."""

from sklearn.ensemble import RandomForestClassifier

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.mol2any import (
    MolToConcatenatedVector,
    MolToMorganFP,
    MolToRDKitPhysChem,
    MolToSmiles,
)
from molpipeline.mol2mol import (
    EmptyMoleculeFilter,
    FragmentDeduplicator,
    MetalDisconnector,
    MixtureFilter,
    SaltRemover,
    StereoRemover,
    TautomerCanonicalizer,
    Uncharger,
)
from molpipeline.mol2mol.filter import ElementFilter
from molpipeline.post_prediction import PostPredictionWrapper


def get_morgan_physchem_rf_pipeline(n_jobs: int = 1) -> Pipeline:
    """Get a pipeline combining Morgan fingerprints and physicochemical properties with a RandomForestClassifier.

    Parameters
    ----------
    n_jobs: int, default=-1
        Number of parallel jobs to use.

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
            ("rf", RandomForestClassifier(n_jobs=n_jobs)),
            (
                "filter_reinserter",
                PostPredictionWrapper(
                    FilterReinserter.from_error_filter(error_filter, None)
                ),
            ),
        ],
        n_jobs=n_jobs,
    )
    return pipeline


def get_standardization_pipeline(n_jobs: int = 1) -> Pipeline:
    """Get the standardization pipeline.

    Parameters
    ----------
    n_jobs: int, optional (default=-1)
        The number of jobs to use for standardization.
        In case of -1, all available CPUs are used.

    Returns
    -------
    Pipeline
        The standardization pipeline.
    """
    error_filter = ErrorFilter(filter_everything=True)
    # Set up pipeline
    standardization_pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMol()),
            ("metal_disconnector", MetalDisconnector()),
            ("salt_remover", SaltRemover()),
            ("element_filter", ElementFilter()),
            ("uncharge1", Uncharger()),
            ("canonical_tautomer", TautomerCanonicalizer()),
            ("uncharge2", Uncharger()),
            ("stereo_remover", StereoRemover()),
            ("fragment_deduplicator", FragmentDeduplicator()),
            ("mixture_remover", MixtureFilter()),
            ("empty_molecule_remover", EmptyMoleculeFilter()),
            ("mol2smi", MolToSmiles()),
            ("error_filter", error_filter),
            ("error_replacer", FilterReinserter.from_error_filter(error_filter, None)),
        ],
        n_jobs=n_jobs,
    )
    return standardization_pipeline
