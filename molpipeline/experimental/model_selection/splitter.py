"""Splitter for creating train/test sets."""

from typing import Any, Generator, Literal, get_args

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
from sklearn.model_selection import (
    BaseShuffleSplit,
)
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import check_array, shuffle
from sklearn.utils.validation import _num_samples, check_random_state

SplitSizeOption = Literal["groups", "samples"]


class GroupShuffleSplit(BaseShuffleSplit):
    """Creates a shuffle split while considering groups.

    This is a modified version of sklearn's GroupShuffleSplit which can control with
    the `split_size` parameter whether `train_size` and `test_size` refer to the number
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
        split_size: SplitSizeOption = "groups",
        random_state: int | RandomState | None = None
    ) -> None:
        """Create a new GroupShuffleSplit.

             Parameters
        ----------
        n_splits: int, default=5
            Number of re-shuffling & splitting iterations.
        test_size: float, int, or None, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
        train_size: float, int, or None, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
            If int, represents the absolute number of train samples.
            If None, the value is set to the complement of the test size.
        split_size: SplitSizeOption, default='groups'
            Determines whether `train_size` and `test_size` refer to the number of groups or the number of samples.
        random_state: int, RandomState instance or None, default=None
            Controls the randomness of the training and testing indices produced.
            Pass an int for reproducible output across multiple function calls.
        """
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.2
        if split_size not in get_args(SplitSizeOption):
            raise ValueError(
                "Bad parameter 'split_size'. Allowed are 'groups' and 'samples'."
            )
        self.split_size = split_size

    def _iter_indices_split_size_samples(
        self, X: Any, groups: npt.ArrayLike  # pylint: disable=invalid-name
    ) -> Generator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]], None, None]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X: Any
            The input data to split.
        y: Any, optional
            The target variable for supervised learning problems. Default is None.
        groups: npt.ArrayLike | None
            Group labels for the samples used while splitting the dataset into train/test set.
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

        classes, group_indices, group_counts = np.unique(
            groups,
            return_inverse=True,
            return_counts=True,
        )
        class_indices = np.arange(len(classes))

        for _ in range(self.n_splits):

            # pre-compute random assignments to train or test set for each group
            random_bucket_assignments = rng.randint(0, 2, size=len(classes))

            # randomize the group order for assignment to train/test
            group_counts_shuffled, class_indices_shuffled = shuffle(
                group_counts, class_indices, random_state=rng
            )

            # track train and test sets in arrays of length 2
            samples_sizes = np.array([n_train, n_test], dtype=np.int_)
            bucket_sizes = np.zeros(2, dtype=np.int_)
            bucket_elements: list[list[int]] = [[], []]

            for class_index, group_size, bucket_index in zip(
                class_indices_shuffled, group_counts_shuffled, random_bucket_assignments
            ):
                first_bucket_size = bucket_sizes[bucket_index] + group_size
                second_bucket_size = bucket_sizes[1 - bucket_index] + group_size

                # first, try to assign the group randomly to a bucket
                bucket_selection = bucket_index
                if first_bucket_size <= samples_sizes[bucket_index]:
                    bucket_selection = bucket_index
                elif second_bucket_size <= samples_sizes[1 - bucket_index]:
                    bucket_selection = 1 - bucket_index
                else:
                    # the group does not fit in any bucket. It is assigned to the bucket
                    # which will be closer to its target sample sizes.
                    first_diff = first_bucket_size - samples_sizes[bucket_index]
                    second_diff = second_bucket_size - samples_sizes[1 - bucket_index]
                    if second_diff < first_diff:
                        bucket_selection = 1 - bucket_index

                bucket_elements[bucket_selection].append(class_index)
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
            Group labels for the samples used while splitting the dataset into train/test set.
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
            If the 'split_size' parameter is not 'groups' or 'samples'.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)

        if self.split_size == "groups":
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
        elif self.split_size == "samples":
            if groups is None:
                # assert for mypy
                raise AssertionError("The 'groups' parameter should not be None.")
            yield from self._iter_indices_split_size_samples(X, groups)

        else:
            raise AssertionError("Unknown parameter for 'split_size'.")
