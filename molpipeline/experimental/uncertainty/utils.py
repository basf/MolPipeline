"""Conformal prediction utils"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, is_classifier, is_regressor


def _bin_targets(y: npt.NDArray[Any], n_bins: int = 10) -> npt.NDArray[np.int_]:
    """Bin continuous targets for stratified splitting in regression.

    Returns
    -------
        Binned targets.

    """
    y = np.asarray(y)
    bins = np.linspace(np.min(y), np.max(y), n_bins + 1)
    y_binned = np.digitize(y, bins) - 1
    y_binned[y_binned == n_bins] = n_bins - 1
    return y_binned


def _detect_estimator_type(
    estimator: BaseEstimator,
) -> Literal["classifier", "regressor"]:
    """Automatically detect whether an estimator is a classifier or regressor.

    Returns
    -------
    Literal["classifier", "regressor"]
        The detected estimator type.

    Raises
    ------
    ValueError
        If type cannot be determined.

    """
    if is_classifier(estimator):
        return "classifier"
    if is_regressor(estimator):
        return "regressor"
    raise ValueError(
        f"Could not determine if {type(estimator).__name__} is a "
        "classifier or regressor. Please specify estimator_type explicitly.",
    )
