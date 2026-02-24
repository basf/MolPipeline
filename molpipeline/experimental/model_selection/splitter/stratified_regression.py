"""Stratified K-Fold splitter for regression tasks using quantile-based binning."""

from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state


def create_continuous_stratified_folds(
    y: npt.NDArray[Any],
    n_splits: int,
    n_groups: int = 10,
    random_state: int | None = None,
) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
    """Create stratified folds for continuous targets using quantile-based binning.

    This method creates stratified cross-validation folds for regression by:
    1. Binning continuous targets into quantile-based groups
    2. Using stratified sampling to ensure balanced target distribution
    3. Returning train/validation index pairs

    Parameters
    ----------
    y : npt.NDArray[Any]
        Continuous target values to stratify.
    n_splits : int
        Number of cross-validation folds.
    n_groups : int, optional
        Number of quantile groups to create for stratification (default: 10).
    random_state : int | None, optional
        Random state for reproducibility.

    Returns
    -------
    list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]
        List of (train_indices, validation_indices) tuples for each fold.

    """
    rng = check_random_state(random_state)

    n_effective_groups = min(n_groups, len(np.unique(y)))
    rng_noise = np.random.default_rng(random_state)
    y_mod = np.asarray(y) + rng_noise.random(len(y)) * 1e-9
    # Use pandas qcut for quantile binning
    y_binned = pd.qcut(y_mod, n_effective_groups, labels=False, duplicates="drop")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=rng,
    )

    fold_assignments = np.zeros(len(y), dtype=int)
    for fold_idx, (_, val_indices) in enumerate(splitter.split(y, y_binned)):
        fold_assignments[val_indices] = fold_idx

    cv_splits = []
    for fold_idx in range(n_splits):
        val_indices = np.where(fold_assignments == fold_idx)[0]
        train_indices = np.where(fold_assignments != fold_idx)[0]
        cv_splits.append((train_indices, val_indices))

    return cv_splits
