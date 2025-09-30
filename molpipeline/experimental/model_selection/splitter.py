"""Splitter for creating train/test sets."""

from collections.abc import Generator
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import BaseShuffleSplit, StratifiedKFold
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import check_array, shuffle
from sklearn.utils.validation import _num_samples, check_random_state

SplitModeOption = Literal["groups", "samples"]


class GroupShuffleSplit(BaseShuffleSplit):
    """Creates a shuffle split while considering groups.

    This is a modified version of sklearn's GroupShuffleSplit which can control with
    the `split_mode` parameter whether `train_size` and `test_size` refer to the number
    of groups (sklearn's implementation) or the number of samples.

    Note that this functionality is also an open PR in sklearn:
    https://github.com/scikit-learn/scikit-learn/pull/29683
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: float | None = None,
        train_size: float | None = None,
        split_mode: SplitModeOption = "groups",
        random_state: int | RandomState | None = None,
    ) -> None:
        """Create a new GroupShuffleSplit.

        Parameters
        ----------
        n_splits: int, default=5
            Number of re-shuffling & splitting iterations.
        test_size: float | None, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
        train_size: float | None, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the train split.
            If int, represents the absolute number of train samples.
            If None, the value is set to the complement of the test size.
        split_mode: SplitSizeOption, default='groups'
            Determines whether `train_size` and `test_size` refer to the number of
            groups or the number of samples.
        random_state: int | RandomState | None, optional
            Controls the randomness of the training and testing indices produced.
            Pass an int for reproducible output across multiple function calls.

        Raises
        ------
        ValueError
            If `split_mode` is not 'groups' or 'samples'.

        """
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.2
        if split_mode not in get_args(SplitModeOption):
            raise ValueError(
                "Bad parameter 'split_mode'. Allowed are 'groups' and 'samples'."
            )
        self.split_mode = split_mode

    def _iter_indices_split_mode_samples(
        self,
        X: Any,  # pylint: disable=invalid-name
        groups: npt.ArrayLike,
    ) -> Generator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]], None, None]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X: Any
            The input data to split.
        groups: npt.ArrayLike
            Group labels for the samples used while splitting the dataset into
            train/test set.
            This parameter is required and should not be None.

        Yields
        ------
        train: npt.NDArray[np.int_]
            The training set indices for that split.
        test: npt.NDArray[np.int_]
            The testing set indices for that split.
        """
        # pylint: disable=too-many-locals
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )
        rng = check_random_state(self.random_state)

        unique_groups, group_indices, group_counts = np.unique(
            groups,
            return_inverse=True,
            return_counts=True,
        )
        unique_groups_indices = np.arange(len(unique_groups))

        for _ in range(self.n_splits):
            # pre-compute random assignments to train or test set for each group
            random_bucket_assignments = rng.randint(0, 2, size=len(unique_groups))

            # randomize the group order for assignment to train/test
            group_counts_shuffled, unique_groups_indices_shuffled = shuffle(
                group_counts, unique_groups_indices, random_state=rng
            )

            # track train and test sets in arrays of length 2
            samples_sizes_target = np.array([n_train, n_test], dtype=np.int_)
            bucket_sizes = np.zeros(2, dtype=np.int_)
            bucket_elements: list[list[int]] = [[], []]

            # assign groups to buckets randomly but consider their size:
            #    1. if the group fits in the randomly assigned bucket, assign it there.
            #    2. if it fits into the other bucket, assign it there.
            #    3. if it does not fit in both buckets, assign it to the bucket which
            #       will be closer to its target size.
            for i, group_index in enumerate(unique_groups_indices_shuffled):
                group_size = group_counts_shuffled[i]
                assigned_bucket = random_bucket_assignments[i]

                first_bucket_size = bucket_sizes[assigned_bucket] + group_size
                second_bucket_size = bucket_sizes[1 - assigned_bucket] + group_size

                # first, try to assign the group randomly to a bucket
                bucket_selection = assigned_bucket
                if first_bucket_size <= samples_sizes_target[assigned_bucket]:
                    bucket_selection = assigned_bucket
                elif second_bucket_size <= samples_sizes_target[1 - assigned_bucket]:
                    bucket_selection = 1 - assigned_bucket
                else:
                    # the group does not fit in any bucket. It is assigned to the bucket
                    # which will be closer to its target sample sizes.
                    first_diff = (
                        first_bucket_size - samples_sizes_target[assigned_bucket]
                    )
                    second_diff = (
                        second_bucket_size - samples_sizes_target[1 - assigned_bucket]
                    )
                    if second_diff < first_diff:
                        bucket_selection = 1 - assigned_bucket

                bucket_elements[bucket_selection].append(group_index)
                bucket_sizes[bucket_selection] += group_size

            # map group indices back to sample indices
            train = np.flatnonzero(np.isin(group_indices, bucket_elements[0]))
            test = np.flatnonzero(np.isin(group_indices, bucket_elements[1]))

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test

    def _iter_indices(
        self,
        X: Any,
        y: Any = None,
        groups: npt.ArrayLike | None = None,
    ) -> Generator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]], None, None]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X: Any
            The input data to split.
        y: Any, optional
            The target variable for supervised learning problems. Default is None.
        groups: npt.ArrayLike | None
            Group labels for the samples used while splitting the dataset into
            train/test set.
            This parameter is required and should not be None.

        Yields
        ------
        train: npt.NDArray[np.int_]
            The training set indices for that split.
        test: npt.NDArray[np.int_]
            The testing set indices for that split.

        Raises
        ------
        ValueError
            If the 'groups' parameter is None.
        AssertionError
            If the 'split_mode' parameter is not 'groups' or 'samples'.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)

        if self.split_mode == "groups":
            if groups is None:
                # assert for mypy
                raise AssertionError("The 'groups' parameter should not be None.")
            classes, group_indices = np.unique(groups, return_inverse=True)
            for group_train, group_test in super()._iter_indices(X=classes):
                # these are the indices of classes in the partition
                # invert them into data indices

                train = np.flatnonzero(np.isin(group_indices, group_train))
                test = np.flatnonzero(np.isin(group_indices, group_test))

                yield train, test
        elif self.split_mode == "samples":
            if groups is None:
                # assert for mypy
                raise AssertionError("The 'groups' parameter should not be None.")
            yield from self._iter_indices_split_mode_samples(X, groups)

        else:
            raise AssertionError("Unknown parameter for 'split_mode'.")


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
