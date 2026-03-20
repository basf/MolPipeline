"""Backward-compatible utilities module for conformal prediction.

This module preserves the legacy import path
`molpipeline.experimental.uncertainty.utils` by re-exporting nonconformity
utilities from `conformal_base`.
"""

from molpipeline.experimental.uncertainty.conformal_base import (
    HingeNonconformity,
    LogNonconformity,
    MarginNonconformity,
    NonconformityFunctor,
    SVMMarginNonconformity,
    create_nonconformity_function,
)

__all__ = [
    "NonconformityFunctor",
    "HingeNonconformity",
    "MarginNonconformity",
    "LogNonconformity",
    "SVMMarginNonconformity",
    "create_nonconformity_function",
]
