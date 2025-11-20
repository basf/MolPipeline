"""Conformal prediction wrappers for classification and regression using crepes.

This module provides a unified interface for conformal prediction with four main classes:
- ConformalClassifier: Single-model conformal classification
- CrossConformalClassifier: Cross-validation conformal classification
- ConformalRegressor: Single-model conformal regression
- CrossConformalRegressor: Cross-validation conformal regression

All classes use composition with crepes and provide full sklearn compatibility.
"""

# Import the four main conformal prediction classes
from molpipeline.experimental.uncertainty.conformal import (
    ConformalClassifier,
    ConformalRegressor,
    CrossConformalClassifier,
    CrossConformalRegressor,
)

# Import nonconformity functions from utils
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
    "create_nonconformity_function",
]
