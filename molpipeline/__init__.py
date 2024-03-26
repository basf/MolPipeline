"""Initialize the molpipeline package."""

# pylint: disable=no-name-in-module
from rdkit.Chem import PropertyPickleOptions, SetDefaultPickleProperties

from .post_prediction import PostPredictionWrapper

# Keep all properties when pickling. Otherwise, we will lose properties set on RDKitMol when passed to
# multiprocessing subprocesses.
SetDefaultPickleProperties(PropertyPickleOptions.AllProps)

from .error_handling import ErrorFilter, FilterReinserter
from .pipeline import Pipeline

__all__ = [
    "Pipeline",
    "ErrorFilter",
    "FilterReinserter",
    "PostPredictionWrapper",
]
