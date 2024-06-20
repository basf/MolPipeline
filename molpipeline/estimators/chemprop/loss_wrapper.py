"""Wrapper for Chemprop loss functions."""

from typing import Any

import torch
from chemprop.nn.loss import BCELoss as _BCELoss
from chemprop.nn.loss import BinaryDirichletLoss as _BinaryDirichletLoss
from chemprop.nn.loss import CrossEntropyLoss as _CrossEntropyLoss
from chemprop.nn.loss import EvidentialLoss as _EvidentialLoss
from chemprop.nn.loss import LossFunction as _LossFunction
from chemprop.nn.loss import MSELoss as _MSELoss
from chemprop.nn.loss import MulticlassDirichletLoss as _MulticlassDirichletLoss
from chemprop.nn.loss import MVELoss as _MVELoss
from chemprop.nn.loss import SIDLoss as _SIDLoss
from numpy.typing import ArrayLike


class LossFunctionParamMixin:
    """Mixin for loss functions to get and set parameters."""

    _original_task_weights: ArrayLike

    def __init__(self: _LossFunction, task_weights: ArrayLike) -> None:
        """Initialize the loss function.

        Parameters
        ----------
        task_weights : ArrayLike
            The weights for each task.

        """
        super().__init__(task_weights=task_weights)  # type: ignore
        self._original_task_weights = task_weights

    # pylint: disable=unused-argument
    def get_params(self: _LossFunction, deep: bool = True) -> dict[str, Any]:
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

    def set_params(self: _LossFunction, **params: Any) -> _LossFunction:
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
                task_weights, dtype=torch.float
            ).view(1, -1)
            self.load_state_dict(state_dict)
        return self


class BCELoss(LossFunctionParamMixin, _BCELoss):
    """Binary cross-entropy loss function."""


class BinaryDirichletLoss(LossFunctionParamMixin, _BinaryDirichletLoss):
    """Binary Dirichlet loss function."""


class CrossEntropyLoss(LossFunctionParamMixin, _CrossEntropyLoss):
    """Cross-entropy loss function."""


class EvidentialLoss(LossFunctionParamMixin, _EvidentialLoss):
    """Evidential loss function."""


class MSELoss(LossFunctionParamMixin, _MSELoss):
    """Mean squared error loss function."""


class MulticlassDirichletLoss(LossFunctionParamMixin, _MulticlassDirichletLoss):
    """Multiclass Dirichlet loss function."""


class MVELoss(LossFunctionParamMixin, _MVELoss):
    """Mean value entropy loss function."""


class SIDLoss(LossFunctionParamMixin, _SIDLoss):
    """SID loss function."""
