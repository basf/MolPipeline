"""Module for ensemble models."""

from molpipeline.estimators.ensemble.split_ensemble import (
    SplitEnsembleClassifier,
    SplitEnsembleRegressor,
)

__all__ = ["SplitEnsembleClassifier", "SplitEnsembleRegressor"]
