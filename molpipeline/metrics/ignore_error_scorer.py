"""Scorer that ignores a given value in the prediction array."""
from __future__ import annotations
from typing import Any, Sequence

from loguru import logger
import numpy as np
from sklearn.metrics._scorer import (
    _ProbaScorer,
    _ThresholdScorer,
    _BaseScorer,
)  # pylint: disable=protected-access
from sklearn import metrics


def ignored_value_scorer(
    scorer: str | _BaseScorer, ignore_value: Any = None
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

    needs_proba = isinstance(scorer, _ProbaScorer)
    needs_threshold = isinstance(scorer, _ThresholdScorer)
    score_func = scorer._score_funcn  # pylint: disable=protected-access

    def newscore(
            y_true: Sequence[float | int],
            y_pred: Sequence[float | int],
            **kwargs: Any,
    ) -> float:
        """Compute the score for the given prediction arrays.

        Parameters
        ----------
        y_true : Iterable[float | int]
            The true values.
        y_pred : Iterable[float | int]
            The predicted values.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            The score for the given prediction arrays.
        """
        retained_y_true = ~np.equal(y_true, ignore_value)
        retained_y_pred = ~np.equal(y_pred, ignore_value)
        all_retained = retained_y_pred & retained_y_true

        if not np.all(all_retained):
            logger.warning(
                f"Warning, prediction array contains NaN values, removing {sum(~all_retained)} elements"
            )

        y_true_ = np.array(y_true)[all_retained]
        y_pred_ = np.array(y_pred)[all_retained]
        if np.any(np.isnan(y_true_)):
            raise ValueError("y_true still contains NaN values")
        if np.any(np.isnan(y_pred_)):
            raise ValueError("y_pred still contains NaN values")
        _kwargs = dict(kwargs)
        if "sample_weight" in _kwargs and _kwargs["sample_weight"] is not None:
            _kwargs["sample_weight"] = _kwargs["sample_weight"][all_retained]
        return score_func(y_true_, y_pred_, **_kwargs)

    return metrics.make_scorer(
        newscore, needs_proba=needs_proba, needs_threshold=needs_threshold
    )
