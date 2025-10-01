"""Wrapper for Chemprop loss functions."""

from typing import Any

import torch
from chemprop.nn.metrics import MSE as _MSE
from chemprop.nn.metrics import SID as _SID
from chemprop.nn.metrics import BCELoss as _BCELoss
from chemprop.nn.metrics import BinaryAUROC as _BinaryAUROC
from chemprop.nn.metrics import ChempropMetric as _ChempropMetric
from chemprop.nn.metrics import CrossEntropyLoss as _CrossEntropyLoss
from chemprop.nn.metrics import DirichletLoss as _DirichletLoss
from chemprop.nn.metrics import EvidentialLoss as _EvidentialLoss
from chemprop.nn.metrics import MulticlassMCCMetric as _MulticlassMCCMetric
from chemprop.nn.metrics import MVELoss as _MVELoss
from numpy.typing import ArrayLike


class LossFunctionParamMixin:
    """Mixin for loss functions to get and set parameters."""

    _original_task_weights: ArrayLike

    def __init__(self: _ChempropMetric, task_weights: ArrayLike | float = 1.0) -> None:
        """Initialize the loss function.

        Parameters
        ----------
        task_weights : ArrayLike | float, optional
            The weights for each task.

        """
        super().__init__(task_weights=task_weights)  # type: ignore
        self._original_task_weights = task_weights

    # pylint: disable=unused-argument
    def get_params(self: _ChempropMetric, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Get the parameters of the loss function.

        Parameters
        ----------
        deep : bool, optional
            Not used, only present to match the sklearn API.

        Returns
        -------
        dict[str, Any]
            The parameters of the loss function.

        """
        return {"task_weights": self._original_task_weights}

    def set_params(self: _ChempropMetric, **params: Any) -> _ChempropMetric:
        """Set the parameters of the loss function.

        Parameters
        ----------
        **params : Any
            The parameters to set.

        Returns
        -------
        Self
            The loss function with the new parameters.

        """
        task_weights = params.pop("task_weights", None)
        if task_weights is not None:
            self._original_task_weights = task_weights
            state_dict = self.state_dict()
            state_dict["task_weights"] = torch.as_tensor(
                task_weights,
                dtype=torch.float,
            ).view(1, -1)
            self.load_state_dict(state_dict)
        return self


class BCELoss(LossFunctionParamMixin, _BCELoss):
    """Binary cross-entropy loss function."""


class DirichletLoss(LossFunctionParamMixin, _DirichletLoss):
    """Dirichlet loss function."""


class MulticlassMCCMetric(LossFunctionParamMixin, _MulticlassMCCMetric):
    """Multiclass Matthews correlation coefficient metric."""


class CrossEntropyLoss(LossFunctionParamMixin, _CrossEntropyLoss):
    """Cross-entropy loss function."""


class EvidentialLoss(LossFunctionParamMixin, _EvidentialLoss):
    """Evidential loss function."""


class MSE(LossFunctionParamMixin, _MSE):
    """Mean squared error loss function."""


class MVELoss(LossFunctionParamMixin, _MVELoss):
    """Mean value entropy loss function."""


class SID(LossFunctionParamMixin, _SID):
    """SID score function."""


class BinaryAUROC(LossFunctionParamMixin, _BinaryAUROC):  # pylint: disable=too-many-ancestors
    """Binary area under the receiver operating characteristic curve metric."""
