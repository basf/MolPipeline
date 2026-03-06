"""Scorer that ignores a given value in the prediction array."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn import metrics
from sklearn.metrics._scorer import _BaseScorer


class _IgnoreValueScoreFunc:  # pylint: disable=too-few-public-methods
    """Callable score function that filters out a given ignore value before scoring.

    This class wraps a score function so that it can be pickled.
    """

    def __init__(self, score_func: Callable[..., float], ignore_value: Any) -> None:
        """Initialize _IgnoreValueScoreFunc.

        Parameters
        ----------
        score_func : Callable[..., float]
            The underlying score function.
        ignore_value : Any
            The value to ignore in the prediction array.

        """
        self.score_func = score_func
        self.ignore_value = ignore_value
        # Forward the name of the wrapped score function because sklearn GridSearchCV's
        # multimetric scoring calls __name__ on the score function for its __repr__
        self.__name__ = getattr(score_func, "__name__", type(score_func).__name__)

    def __call__(
        self,
        y_true: npt.NDArray[np.float64 | np.int_],
        y_pred: npt.NDArray[np.float64 | np.int_],
        **kwargs: Any,
    ) -> float:
        """Compute the score for the given prediction arrays.

        Parameters
        ----------
        y_true : npt.NDArray[np.float64 | np.int_]
            The true values.
        y_pred : npt.NDArray[np.float64 | np.int_]
            The predicted values.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            The score for the given prediction arrays.

        """
        retained_y_true: npt.NDArray[np.bool_]
        retained_y_pred: npt.NDArray[np.bool_]
        if pd.notna(self.ignore_value):
            retained_y_true = ~np.equal(y_true, self.ignore_value)
            retained_y_pred = ~np.equal(y_pred, self.ignore_value)
        else:
            retained_y_true = pd.notna(y_true)
            retained_y_pred = pd.notna(y_pred)

        all_retained = retained_y_pred & retained_y_true

        if not np.all(all_retained):
            logger.warning(
                f"Warning, prediction array contains NaN values, "
                f"removing {sum(~all_retained)} elements",
            )
        y_true_ = np.copy(np.array(y_true)[all_retained])
        y_pred_ = np.array(np.array(y_pred)[all_retained].tolist())
        kwargs_ = dict(kwargs)
        if "sample_weight" in kwargs_ and kwargs_["sample_weight"] is not None:
            kwargs_["sample_weight"] = kwargs_["sample_weight"][all_retained]
        return self.score_func(y_true_, y_pred_, **kwargs_)


def ignored_value_scorer(
    scorer: str | _BaseScorer,
    ignore_value: Any = None,
) -> _BaseScorer:
    """Create a scorer that ignores a given value in the prediction array.

    This is relevant for pipline models which replace errors with a given value.
    The wrapped scorer will ignore that value and return the corresponding score.

    Parameters
    ----------
    scorer : str or _BaseScorer
        The scorer to wrap.
    ignore_value : Any, optional
        The value to ignore in the prediction array.
        Default: None

    Returns
    -------
    _BaseScorer
        The scorer that ignores the given value.

    """
    if isinstance(scorer, str):
        scorer = metrics.get_scorer(scorer)

    score_func = scorer._score_func  # noqa: SLF001
    response_method = scorer._response_method  # noqa: SLF001
    scorer_kwargs = scorer._kwargs  # noqa: SLF001
    if scorer._sign < 0:  # noqa: SLF001
        scorer_kwargs["greater_is_better"] = False

    newscore = _IgnoreValueScoreFunc(score_func, ignore_value)

    return metrics.make_scorer(
        newscore,
        response_method=response_method,
        **scorer_kwargs,
    )
