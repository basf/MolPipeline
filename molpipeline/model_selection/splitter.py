from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupShuffleSplit,
    BaseShuffleSplit,
)

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import check_array

X = np.ones(shape=(8, 2))
y = np.ones(shape=(8, 1))
groups = np.array([1, 1, 2, 2, 2, 3, 3, 3])

gss = GroupShuffleSplit(n_splits=2, train_size=0.7, random_state=42)
gss.get_n_splits()

print(gss)
GroupShuffleSplit(n_splits=2, random_state=42, test_size=None, train_size=0.7)
train, test = next(gss.split(X, y, groups))


class SampleBasedGroupShuffleSplit(BaseShuffleSplit):
    def __init__(
        self,
        n_splits=5,
        *,
        test_size=None,
        train_size=None,
        group_by: str = "number",
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
            pass
        else:
            raise AssertionError("Unknown parameter for '_group_by'.")

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)
