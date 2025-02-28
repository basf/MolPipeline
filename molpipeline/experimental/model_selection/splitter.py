from sklearn.model_selection import (
    BaseShuffleSplit,
)

import numpy as np
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import check_array, shuffle
from sklearn.utils.validation import _num_samples, check_random_state


class GroupShuffleSplit(BaseShuffleSplit):
    def __init__(
        self,
        n_splits=5,
        *,
        test_size=None,
        train_size=None,
        group_by="number",
        random_state=None
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.2
        if group_by not in ["size", "number"]:
            raise ValueError(
                "Bad parameter 'group_by'. Allowed are 'size' and 'number'."
            )
        self._group_by = group_by

    # def _iter_indices_by_group_size2(self, X, groups):
    #     n_samples = _num_samples(X)
    #     n_train, n_test = _validate_shuffle_split(
    #         n_samples,
    #         self.test_size,
    #         self.train_size,
    #         default_test_size=self._default_test_size,
    #     )
    #     rng = check_random_state(self.random_state)
    #
    #     classes, group_indices, group_counts = np.unique(
    #         groups,
    #         return_inverse=True,
    #         return_counts=True,
    #     )
    #     class_indices = np.arange(len(classes))
    #
    #     for i in range(self.n_splits):
    #
    #         # pre-compute random assignments to train or test set for each group
    #         random_bucket_assignments = rng.randint(0, 2, size=len(classes))
    #
    #         # randomize the group order for assignment to train/test
    #         group_counts_shuffled, class_indices_shuffled = shuffle(
    #             group_counts, class_indices, random_state=rng
    #         )
    #
    #         # track train and test sets in arrays of length 2
    #         samples_sizes = np.array([n_train, n_test], dtype=np.int_)
    #         bucket_sizes = np.zeros(2, dtype=np.int_)
    #         bucket_elements = [[], []]
    #
    #         for class_index, group_size, bucket_index in zip(
    #             class_indices_shuffled, group_counts_shuffled, random_bucket_assignments
    #         ):
    #             first_bucket_size = bucket_sizes[bucket_index] + group_size
    #             second_bucket_size = bucket_sizes[1 - bucket_index] + group_size
    #
    #             # first, try to assign the group randomly to a bucket
    #             if first_bucket_size <= samples_sizes[bucket_index]:
    #                 bucket_elements[bucket_index].append(class_index)
    #                 bucket_sizes[bucket_index] += group_size
    #             elif second_bucket_size <= samples_sizes[1 - bucket_index]:
    #                 bucket_elements[1 - bucket_index].append(class_index)
    #                 bucket_sizes[1 - bucket_index] += group_size
    #             else:
    #                 # both buckets are full
    #                 # assign the group to the bucket with the small difference to the target split sizes
    #                 first_diff = first_bucket_size - samples_sizes[bucket_index]
    #                 second_diff = second_bucket_size - samples_sizes[1 - bucket_index]
    #                 if first_diff < second_diff:
    #                     bucket_elements[bucket_index].append(class_index)
    #                     bucket_sizes[bucket_index] += group_size
    #                 else:
    #                     bucket_elements[1 - bucket_index].append(class_index)
    #                     bucket_sizes[1 - bucket_index] += group_size
    #
    #         train = np.flatnonzero(np.isin(group_indices, bucket_elements[0]))
    #         test = np.flatnonzero(np.isin(group_indices, bucket_elements[1]))
    #
    #         train = rng.permutation(train)
    #         test = rng.permutation(test)
    #
    #         yield train, test

    def _iter_indices_by_group_size(self, X, groups):
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

        for i in range(self.n_splits):

            # pre-compute random assignments to train or test set for each group
            random_bucket_assignments = rng.randint(0, 2, size=len(classes))

            # randomize the group order for assignment to train/test
            group_counts_shuffled, class_indices_shuffled = shuffle(
                group_counts, class_indices, random_state=rng
            )

            # track train and test sets in arrays of length 2
            samples_sizes = np.array([n_train, n_test], dtype=np.int_)
            bucket_sizes = np.zeros(2, dtype=np.int_)
            bucket_elements = [[], []]

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

    # def _iter_indices_by_group_size2(self, X, groups):
    #     # validation checks taken from super._iter_indices
    #     n_samples = _num_samples(X)
    #     n_train, n_test = _validate_shuffle_split(
    #         n_samples,
    #         self.test_size,
    #         self.train_size,
    #         default_test_size=self._default_test_size,
    #     )
    #     rng = check_random_state(self.random_state)
    #
    #     classes, group_indices, group_counts = np.unique(
    #         groups,
    #         return_inverse=True,
    #         return_counts=True,
    #     )
    #     n_classes = len(classes)
    #
    #     for i in range(self.n_splits):
    #         # random partition
    #         permutation = rng.permutation(n_classes)
    #
    #         # fill the test set first
    #         test_sample_size = 0
    #         idx = 0
    #         for class_index in permutation:
    #             idx += 1
    #             test_sample_size += group_counts[class_index]
    #             if test_sample_size >= n_test:
    #                 break
    #
    #         ind_test = permutation[:idx]
    #         ind_train = permutation[idx : (n_test + n_train)]
    #
    #         train = np.flatnonzero(np.isin(group_indices, ind_train))
    #         test = np.flatnonzero(np.isin(group_indices, ind_test))
    #
    #         yield train, test

    def _iter_indices(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)

        if self._group_by == "number":
            classes, group_indices = np.unique(groups, return_inverse=True)
            for group_train, group_test in super()._iter_indices(X=classes):
                # these are the indices of classes in the partition
                # invert them into data indices

                train = np.flatnonzero(np.isin(group_indices, group_train))
                test = np.flatnonzero(np.isin(group_indices, group_test))

                yield train, test
        elif self._group_by == "size":
            yield from self._iter_indices_by_group_size(X, groups)

        else:
            raise AssertionError("Unknown parameter for 'group_by'.")

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)
