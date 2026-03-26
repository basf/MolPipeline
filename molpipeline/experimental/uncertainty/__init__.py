"""Conformal prediction wrappers for classification and regression using crepes.

This module provides a unified interface for conformal prediction with 4 main classes:
- ConformalClassifier: Single-model conformal classification
- CrossConformalClassifier: Cross-validation conformal classification
- ConformalRegressor: Single-model conformal regression
- CrossConformalRegressor: Cross-validation conformal regression

All classes use composition with crepes and provide full sklearn compatibility.
"""

# Import the four main conformal prediction classes via the legacy module path.
from molpipeline.experimental.uncertainty.conformal import (
    ConformalClassifier,
    ConformalRegressor,
    CrossConformalClassifier,
    CrossConformalRegressor,
    _apply_antitonic_regressors,
    _fit_antitonic_regressors,
)

# Import nonconformity functions via the legacy utils module path.
from molpipeline.experimental.uncertainty.utils import (
    HingeNonconformity,
    LogNonconformity,
    MarginNonconformity,
    NonconformityFunctor,
    SVMMarginNonconformity,
    create_nonconformity_function,
)

# Export all the important classes
__all__ = [
    "ConformalClassifier",
    "ConformalRegressor",
    "CrossConformalClassifier",
    "CrossConformalRegressor",
    "HingeNonconformity",
    "LogNonconformity",
    "MarginNonconformity",
    "NonconformityFunctor",
    "SVMMarginNonconformity",
    "_apply_antitonic_regressors",
    "_fit_antitonic_regressors",
    "create_nonconformity_function",
]
