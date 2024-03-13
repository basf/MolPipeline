"""Wrapper classes for the chemprop components to make them compatible with scikit-learn."""

from typing import Any, Iterable, Self

import torch
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.models.model import MPNN as _MPNN
from chemprop.nn.agg import Aggregation
from chemprop.nn.agg import MeanAggregation as _MeanAggregation
from chemprop.nn.agg import SumAggregation as _SumAggregation
from chemprop.nn.ffn import MLP
from chemprop.nn.loss import LossFunction
from chemprop.nn.message_passing import BondMessagePassing as _BondMessagePassing
from chemprop.nn.message_passing import MessagePassing
from chemprop.nn.metrics import BCELoss, Metric
from chemprop.nn.predictors import BinaryClassificationFFN as _BinaryClassificationFFN
from chemprop.nn.predictors import Predictor
from chemprop.nn.predictors import RegressionFFN as _RegressionFFN
from chemprop.nn.utils import Activation, get_activation_function
from sklearn.base import BaseEstimator
from torch import Tensor, nn


# pylint: disable=too-many-ancestors, too-many-instance-attributes
class BondMessagePassing(_BondMessagePassing, BaseEstimator):
    """A wrapper for the BondMessagePassing class."""

    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 3,
        dropout_rate: float = 0.0,
        activation: str | Activation = Activation.RELU,
        undirected: bool = False,
        d_vd: int | None = None,
    ):
        """Initialize the BondMessagePassing class.

        Parameters
        ----------
        d_v : int, optional (default=DEFAULT_ATOM_FDIM)
            The input vertices feature dimension, by default DEFAULT_ATOM_FDIM
        d_e : int, optional (default=DEFAULT_BOND_FDIM)
            The input edges feature dimension, by default DEFAULT_BOND_FDIM
        d_h : int, optional (default=DEFAULT_HIDDEN_DIM)
            The hidden layer dimension, by default DEFAULT_HIDDEN_DIM
        bias : bool, optional (default=False)
            Whether to use bias in the weight matrices.
        depth : int, optional (default=3)
            Number of message passing layers.
        dropout_rate : float, optional (default=0)
            Dropout rate.
        activation : str or Activation, optional (default=Activation.RELU)
            Activation function.
        undirected : bool, optional (default=False)
            Whether to use undirected edges.
        d_vd : int or None, optional (default=None)
            Dimension of additional vertex descriptors that will be concatenated to the hidden features before readout
        """
        super().__init__(
            d_v,
            d_e,
            d_h,
            bias,
            depth,
            dropout_rate,
            activation,
            undirected,
            d_vd,
        )
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.d_vd = d_vd
        self.bias = bias
        self.activation = activation
        self.dropout_rate = dropout_rate

    def reinitialize_network(self) -> Self:
        """Reinitialize the network with the current parameters."""
        self.W_i, self.W_h, self.W_o, self.W_d = self.setup(
            self.d_v, self.d_e, self.d_h, self.d_vd, self.bias
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = get_activation_function(self.activation)
        return self

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the class and update the network."""
        super().set_params(**params)
        self.reinitialize_network()
        return self


# pylint: disable=too-many-ancestors, too-many-instance-attributes
class BinaryClassificationFFN(_BinaryClassificationFFN, BaseEstimator):
    """A wrapper for the BinaryClassificationFFN class."""

    n_targets: int = 1
    _default_criterion: LossFunction = BCELoss()

    def __init__(
        self,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        criterion: LossFunction | None = None,
    ):
        """Initialize the BinaryClassificationFFN class.

        Parameters
        ----------
        n_tasks : int, optional (default=1)
            Number of tasks.
        input_dim : int, optional (default=DEFAULT_HIDDEN_DIM)
            Input dimension.
        hidden_dim : int, optional (default=300)
            Hidden dimension.
        n_layers : int, optional (default=1)
            Number of layers.
        dropout : float, optional (default=0)
            Dropout rate.
        activation : str, optional (default="relu")
            Activation function.
        criterion : LossFunction or None, optional (default=None)
            Loss function. None defaults to BCELoss.
        """
        super().__init__(
            n_tasks=n_tasks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            criterion=criterion,
        )
        self.n_tasks = n_tasks
        self._input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation

    @property
    def input_dim(self) -> int:
        """Get the dimension of input."""
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value: int) -> None:
        """Set the dimension of input."""
        self._input_dim = value

    @property
    def n_tasks(self) -> int:
        """Get the number of tasks."""
        return self._n_tasks

    @n_tasks.setter
    def n_tasks(self, value: int) -> None:
        """Set the number of tasks."""
        self._n_tasks = value

    def reinitialize_fnn(self) -> Self:
        """Reinitialize the feedforward network."""
        self.ffn = MLP(
            self.input_dim,
            self.n_tasks * self.n_targets,
            self.hidden_dim,
            self.n_layers,
            self.dropout,
            self.activation,
        )
        return self

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the class and reinitialize the feedforward network."""
        super().set_params(**params)
        self.reinitialize_fnn()
        return self


class MPNN(_MPNN, BaseEstimator):
    """A wrapper for the MPNN class.

    The MPNN is the main model class in chemprop. It consists of a message passing network, an aggregation function,
    and a feedforward network for prediction.
    """

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = True,
        metric_list: Iterable[Metric] | None = None,
        task_weight: Tensor | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metric_list,
            task_weight,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
        )
        self.metric_list = metric_list
        self.batch_norm = batch_norm
        self.task_weight = task_weight
        self.metrics = metric_list

    def reinitialize_network(self) -> Self:
        """Reinitialize the network with the current parameters."""
        self.bn = (
            nn.BatchNorm1d(self.message_passing.output_dim)
            if self.batch_norm
            else nn.Identity()
        )
        if self.metric_list is None:
            # pylint: disable=protected-access
            self.metrics = [self.predictor._default_metric, self.criterion]
        else:
            self.metrics = list(self.metric_list) + [self.criterion]
        if self.task_weight is None:
            w_t = torch.ones(self.n_tasks)
        else:
            w_t = torch.tensor(self.task_weight)
        self.w_t = nn.Parameter(w_t.unsqueeze(0), False)
        return self

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the class and update the network."""
        super().set_params(**params)
        self.reinitialize_network()
        return self


# pylint: disable=too-many-ancestors
class MeanAggregation(_MeanAggregation, BaseEstimator):
    """Aggregate the graph-level representation by averaging the node representations."""

    def __init__(self, dim: int = 0):
        """Initialize the MeanAggregation class.

        Parameters
        ----------
        dim : int, optional (default=0)
            The dimension to aggregate over. See torch_scater.scatter for more details.
        """
        super().__init__(dim)


# pylint: disable=too-many-ancestors
class SumAggregation(_SumAggregation, BaseEstimator):
    """Aggregate the graph-level representation by summing the node representations."""

    def __init__(self, dim: int = 0):
        """Initialize the SumAggregation class.

        Parameters
        ----------
        dim : int, optional (default=0)
            The dimension to aggregate over. See torch_scater.scatter for more details.
        """
        super().__init__(dim)
