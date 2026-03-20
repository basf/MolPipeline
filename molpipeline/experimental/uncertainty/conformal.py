"""Backward-compatible conformal prediction module.

This module preserves the legacy import path
`molpipeline.experimental.uncertainty.conformal` by re-exporting symbols from
split modules.
"""

from molpipeline.experimental.uncertainty.conformal_base import (
    BaseConformalPredictor,
    _apply_antitonic_regressors,
    _fit_antitonic_regressors,
)
from molpipeline.experimental.uncertainty.conformal_classifier import (
    ConformalClassifier,
    CrossConformalClassifier,
)
from molpipeline.experimental.uncertainty.conformal_regressor import (
    ConformalRegressor,
    CrossConformalRegressor,
)

__all__ = [
    "BaseConformalPredictor",
    "ConformalClassifier",
    "ConformalRegressor",
    "CrossConformalClassifier",
    "CrossConformalRegressor",
    "_apply_antitonic_regressors",
    "_fit_antitonic_regressors",
]
