"""The implementation of pairwise difference learners."""

import abc
from itertools import combinations, product
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels

from molpipeline.utils.molpipeline_types import AnyPredictor

ModelVar = TypeVar("ModelVar", bound=AnyPredictor | BaseEstimator)

_N_PROBA_CLASSES = 2


def dual_vector_combinations(
    vector_1: npt.ArrayLike,
    vector_2: npt.ArrayLike,
    mode: Literal["combine", "diff", "combine_and_diff"] = "combine",
) -> npt.NDArray[Any]:
    """Combine two vectors and return the dual combined vector.

    Parameters
    ----------
    vector_1 : npt.ArrayLike
        Vector to form combinations of.
    vector_2 : npt.ArrayLike
        Vector to form combinations of.
    mode : Literal["combine", "diff", "combine_and_diff"], default="combine"
        Mode of combination. Options are:
        - "combine": concatenate the two vectors (default)
        - "diff": calculate the difference between the two vectors
        - "combine_and_diff": concatenate the two vectors and their difference

    Returns
    -------
    npt.NDArray[Any]
        Combined vector of the two input vectors.

    Raises
    ------
    ValueError
        If mode is not one of 'combine', 'diff', 'combine_and_diff'.

    """
    if mode == "combine":
        return np.vstack([np.hstack(comb) for comb in product(vector_1, vector_2)])
    if mode == "diff":
        return np.vstack([a1 - a2 for a1, a2 in product(vector_1, vector_2)])
    if mode == "combine_and_diff":
        return np.vstack(
            [np.hstack((a1, a2, a1 - a2)) for a1, a2 in product(vector_1, vector_2)],
        )
    raise ValueError(
        f"Invalid mode: {mode}. Valid options are 'combine', 'diff', "
        f"'combine_and_diff'.",
    )


def single_vector_combinations(
    vector: npt.ArrayLike,
    mode: Literal["combine", "diff", "combine_and_diff"] = "combine",
) -> npt.NDArray[Any]:
    """Combine all rows of the vector and return the combined vector.

    Using dual_vector_combinations(X, X) does not result in the same result as
    single_vector_combinations(X) as the former includes combinations of the same row
    with itself.

    Parameters
    ----------
    vector : npt.ArrayLike
        Vector to form combinations of.
    mode : Literal["combine", "diff", "combine_and_diff"], default="combine"
        Mode of combination. Options are:
        - "combine": concatenate the two vectors (default)
        - "diff": calculate the difference between the two vectors
        - "combine_and_diff": concatenate the two vectors and their difference

    Returns
    -------
    npt.NDArray[Any]
        Combined vector of all pairwise row combinations.

    Raises
    ------
    ValueError
        If mode is not one of 'combine', 'diff', 'combine_and_diff'.

    """
    if mode == "combine":
        return np.vstack([np.hstack(comb) for comb in combinations(vector, r=2)])
    if mode == "diff":
        return np.vstack([a1 - a2 for a1, a2 in combinations(vector, r=2)])
    if mode == "combine_and_diff":
        return np.vstack(
            [np.hstack((a1, a2, a1 - a2)) for a1, a2 in combinations(vector, r=2)],
        )
    raise ValueError(
        f"Invalid mode: {mode}. Valid options are 'combine', 'diff', "
        f"'combine_and_diff'.",
    )


class PairwiseDifferenceLearner(BaseEstimator, abc.ABC, Generic[ModelVar]):
    """Base class for pairwise difference learners."""

    def __init__(
        self,
        estimator: ModelVar,
        mode: Literal["combine", "diff", "combine_and_diff"] = "combine",
    ) -> None:
        """Initialize the pairwise difference learner.

        Parameters
        ----------
        estimator : ModelVar
            The underlying estimator used for pairwise difference learning.
        mode : Literal["combine", "diff", "combine_and_diff"]
            Mode of pairwise feature combination.

        """
        self.estimator = estimator
        self.mode = mode

    def fit(
        self,
        X: npt.ArrayLike,  # noqa: N803
        y: npt.ArrayLike,
    ) -> "PairwiseDifferenceLearner[ModelVar]":
        """Fit the model to the data.

        Parameters
        ----------
        X : npt.ArrayLike
            Feature matrix of shape (n_samples, n_features).
        y : npt.ArrayLike
            Target values of shape (n_samples,).

        Returns
        -------
        PairwiseDifferenceLearner
            Fitted estimator.

        """
        self.fit_x = X
        self.fit_y = y
        x_combined = single_vector_combinations(X, mode=self.mode)
        y_diff = single_vector_combinations(y, mode="diff")

        self.estimator.fit(x_combined, y_diff)
        return self

    @abc.abstractmethod
    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:  # noqa: N803
        """Predict target values for the samples in X.

        Parameters
        ----------
        X : npt.ArrayLike
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        npt.NDArray[Any]
            Predicted target values of shape (n_samples,).

        """


class PairwiseDifferenceRegressor(RegressorMixin, PairwiseDifferenceLearner[ModelVar]):
    """Pairwise difference regressor."""

    def predict(
        self,
        X: npt.ArrayLike,  # noqa: N803
        return_std: bool = False,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Predict the target values for the given input.

        Parameters
        ----------
        X : npt.ArrayLike
            Feature matrix of shape (n_samples, n_features).
        return_std : bool
            If True, also return the standard deviation of predictions.

        Returns
        -------
        npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]
            Predicted mean values, and optionally their standard deviations.

        """
        mean_preds = []
        std_preds = []
        for x_ in X:
            x_combined = dual_vector_combinations(
                x_.reshape(1, -1),
                self.fit_x,
                mode=self.mode,
            )
            y_diff_pred = self.estimator.predict(x_combined)
            y_pred = self.fit_y + y_diff_pred
            mean_preds.append(np.mean(y_pred))
            std_preds.append(np.std(y_pred))

        mean_preds_arr = np.array(mean_preds)
        std_preds_arr = np.array(std_preds)

        if return_std:
            return mean_preds_arr, std_preds_arr
        return mean_preds_arr


class PairwiseDifferenceClassifier(
    ClassifierMixin,
    PairwiseDifferenceLearner[ModelVar],
):
    """Pairwise difference classifier.

    Trains an underlying estimator on pairwise-combined features with the sign
    of the label difference as target (+1 if y_i > y_j, -1 if y_i < y_j).
    At prediction time, votes from all training-pair comparisons are aggregated
    to select the most likely class.

    The underlying ``estimator`` must support ``predict`` (and optionally
    ``predict_proba`` for ``predict_proba`` to work on this classifier).
    """

    estimators: list[ModelVar]

    def __init__(
        self,
        estimator: ModelVar,
        mode: Literal["combine", "diff", "combine_and_diff"] = "combine",
    ) -> None:
        """Initialize the pairwise difference classifier.

        Parameters
        ----------
        estimator : ModelVar
            The underlying estimator used for pairwise difference learning.
        mode : Literal["combine", "diff", "combine_and_diff"]
            Mode of pairwise feature combination.

        """
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.estimators_ = []

        super().__init__(estimator=estimator, mode=mode)

    def fit(
        self,
        X: npt.ArrayLike,  # noqa: N803
        y: npt.ArrayLike,
    ) -> "PairwiseDifferenceClassifier":
        """Fit the classifier to the data.

        Parameters
        ----------
        X : npt.ArrayLike
            Feature matrix of shape (n_samples, n_features).
        y : npt.ArrayLike
            Class labels of shape (n_samples,).

        Returns
        -------
        PairwiseDifferenceClassifier
            Fitted estimator.

        """
        x_mat = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = unique_labels(y)
        self.fit_x = x_mat

        fit_y = self.ohe.fit_transform(y.reshape(-1, 1))
        self.fit_y = fit_y.toarray() if hasattr(fit_y, "toarray") else np.asarray(fit_y)

        # Abs to only check if class differ or are identical
        y_combined = np.abs(single_vector_combinations(self.fit_y, mode="diff"))

        x_combined = single_vector_combinations(x_mat, mode=self.mode)
        # No model for cls 0, as it will be the class not predicted by the other models.
        for i in range(1, y_combined.shape[1]):
            target_i = y_combined[:, i]
            class_i_model = clone(self.estimator)
            class_i_model.fit(x_combined, target_i)
            self.estimators_.append(class_i_model)

        return self

    def predict_proba(
        self,
        X: npt.ArrayLike,  # noqa: N803
    ) -> npt.NDArray[np.float64]:
        """Predict class probabilities for the samples in X.

        For each test sample, pairwise comparisons against all training samples
        are made. The predicted sign indicates whether the test sample's label
        is greater (+1) or smaller (-1) than each training sample's label. A
        vote tally over the training labels determines the final prediction.

        Parameters
        ----------
        X : npt.ArrayLike
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted class probabilities of shape (n_samples, n_classes).

        Raises
        ------
        AssertionError
            If the underlying estimator does not predict binary probabilities or
            if an invalid probability distribution is detected.

        """
        x_mat = np.asarray(X)
        predictions = []
        for x_ in x_mat:
            x_ = dual_vector_combinations(x_.reshape(1, -1), self.fit_x, mode=self.mode)
            proba_list = []
            for i, estimator in enumerate(self.estimators_, 1):
                if hasattr(estimator, "predict_proba"):
                    proba_diff = estimator.predict_proba(x_)
                    if not proba_diff.shape[1] == _N_PROBA_CLASSES:
                        raise AssertionError(
                            f"Expected binary classification for pairwise difference "
                            f"model, but got {proba_diff.shape[1]} classes in "
                            f"predict_proba output.",
                        )
                    # For each training sample j:
                    # - if j is class i: prob(test=i) = prob(same)  = proba_diff[j, 0]
                    # - if j is not i:   prob(test=i) = prob(diff)  = proba_diff[j, 1]
                    fit_y_col = self.fit_y[:, i]
                    proba_class = np.mean(
                        np.where(fit_y_col == 1, proba_diff[:, 0], proba_diff[:, 1]),
                    )
                else:  # If no predict_proba available, estimate proba via avg.
                    # Delta is binary: Has same class or not
                    y_delta_pred = estimator.predict(x_)
                    proba_class = np.array(self.fit_y[:, i])  # Copy ref
                    # Flip label if delta is predicted: 1 - 1 = 0; 1 - 0 = 1
                    proba_class[y_delta_pred > 0] = 1 - proba_class[y_delta_pred > 0]
                    proba_class = np.mean(proba_class)
                proba_list.append(proba_class)
            proba_complenent = 1 - np.sum(proba_list)
            if not 0 < proba_complenent < 1:
                raise AssertionError("Wired proba detected!")
            proba_list.insert(0, proba_complenent)
            predictions.append(proba_list)
        return np.array(predictions, dtype=np.float64)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:  # noqa: N803
        """Predict class labels for the samples in X.

        Parameters
        ----------
        X : npt.ArrayLike
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        npt.NDArray[Any]
            Predicted class labels of shape (n_samples,).

        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.ohe.categories_[0][class_indices]
