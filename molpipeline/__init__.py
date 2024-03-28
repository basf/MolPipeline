"""Initialize the molpipeline package."""

# pylint: disable=no-name-in-module
from rdkit.Chem import PropertyPickleOptions, SetDefaultPickleProperties

from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.pipeline import Pipeline
from molpipeline.post_prediction import PostPredictionWrapper

# Keep all properties when pickling. Otherwise, we will lose properties set on RDKitMol when passed to
# multiprocessing subprocesses.
SetDefaultPickleProperties(PropertyPickleOptions.AllProps)

__all__ = [
    "Pipeline",
    "ErrorFilter",
    "FilterReinserter",
    "PostPredictionWrapper",
]
