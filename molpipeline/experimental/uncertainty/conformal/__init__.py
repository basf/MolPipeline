"""Conformal prediction subpackage for classification and regression.

This package provides conformal prediction wrappers using crepes:
- ConformalClassifier / CrossConformalClassifier
- ConformalRegressor / CrossConformalRegressor
- BaseConformalPredictor
- Nonconformity utilities (via uncertainty.utils)
"""

from molpipeline.experimental.uncertainty.conformal.conformal_base import (
    BaseConformalPredictor,
    _apply_antitonic_regressors,
    _fit_antitonic_regressors,
)
from molpipeline.experimental.uncertainty.conformal.conformal_classsifier import (
    ConformalClassifier,
    CrossConformalClassifier,
)
from molpipeline.experimental.uncertainty.conformal.conformal_regressor import (
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
