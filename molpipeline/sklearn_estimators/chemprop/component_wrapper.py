import abc
from typing import Any, Iterable, Self

import torch
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.exceptions import InvalidShapeError
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


class BondMessagePassing(_BondMessagePassing, BaseEstimator):

    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 3,
        dropout_rate: float = 0,
        activation: str | Activation = Activation.RELU,
        undirected: bool = False,
        d_vd: int | None = None,
    ):
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

    def update_network(self) -> Self:
        self.W_i, self.W_h, self.W_o, self.W_d = self.setup(self.d_v, self.d_e, self.d_h, self.d_vd, self.bias)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = get_activation_function(self.activation)

    def set_params(self, **params: Any) -> Self:
        super().set_params(**params)
        self.update_network()
        return self

    def get_params(self, deep=True) -> dict[str, Any]:
        params = super().get_params(deep)
        return params


class BinaryClassificationFFN(_BinaryClassificationFFN, BaseEstimator):
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
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value: int) -> None:
        self._input_dim = value

    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    @n_tasks.setter
    def n_tasks(self, value: int) -> None:
        self._n_tasks = value

    def reinitialize_fnn(self) -> Self:
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
        super().set_params(**params)
        self.reinitialize_fnn()
        return self


class MPNN(_MPNN, BaseEstimator):
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
        self.batch_norm = batch_norm
        self.task_weight = task_weight
        self.metrics = metric_list

    @property
    def task_weight(self) -> Tensor:
        return self._w_t

    @task_weight.setter
    def task_weight(self, value: Tensor | None) -> None:
        if value is None:
            w_t = torch.ones(self.n_tasks)
        else:
            w_t = torch.tensor(value)
        self._w_t = nn.Parameter(w_t.unsqueeze(0), False)

    def update_network(self) -> Self:
        self.bn = (
            nn.BatchNorm1d(self.message_passing.output_dim)
            if self.batch_norm
            else nn.Identity()
        )
        return self

    def set_params(self, **params: Any) -> Self:
        super().set_params(**params)
        if "metrics" in params:
            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            self.metrics = params["metrics"]
        self.update_network()
        return self


class MeanAggregation(_MeanAggregation, BaseEstimator):
    pass


class SumAggregation(_SumAggregation, BaseEstimator):
    def __init__(self, dim: int = 0):
        super().__init__(dim)
