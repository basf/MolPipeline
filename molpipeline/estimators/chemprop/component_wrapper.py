"""Wrapper classes for the chemprop components to make them compatible with scikit-learn."""

import abc
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
from chemprop.nn.metrics import (
    BinaryAUROCMetric,
    CrossEntropyMetric,
    Metric,
    MSEMetric,
    SIDMetric,
)
from chemprop.nn.predictors import BinaryClassificationFFN as _BinaryClassificationFFN
from chemprop.nn.predictors import BinaryDirichletFFN as _BinaryDirichletFFN
from chemprop.nn.predictors import EvidentialFFN as _EvidentialFFN
from chemprop.nn.predictors import (
    MulticlassClassificationFFN as _MulticlassClassificationFFN,
)
from chemprop.nn.predictors import MulticlassDirichletFFN as _MulticlassDirichletFFN
from chemprop.nn.predictors import MveFFN as _MveFFN
from chemprop.nn.predictors import RegressionFFN as _RegressionFFN
from chemprop.nn.predictors import SpectralFFN as _SpectralFFN
from chemprop.nn.predictors import (
    _FFNPredictorBase as _Predictor,  # pylint: disable=protected-access
)
from chemprop.nn.transforms import UnscaleTransform
from chemprop.nn.utils import Activation, get_activation_function
from sklearn.base import BaseEstimator
from torch import Tensor, nn

from molpipeline.estimators.chemprop.loss_wrapper import (
    BCELoss,
    BinaryDirichletLoss,
    CrossEntropyLoss,
    EvidentialLoss,
    MSELoss,
    MulticlassDirichletLoss,
    MVELoss,
    SIDLoss,
)


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
        """Reinitialize the network with the current parameters.

        Returns
        -------
        Self
            The reinitialized network.
        """
        self.W_i, self.W_h, self.W_o, self.W_d = self.setup(
            self.d_v, self.d_e, self.d_h, self.d_vd, self.bias
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        if isinstance(self.activation, str):
            self.tau = get_activation_function(self.activation)
        else:
            self.tau = self.activation
        return self

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the class and update the network.

        Parameters
        ----------
        **params: Any
            The parameters to set.

        Returns
        -------
        Self
            The model with the new parameters.
        """
        super().set_params(**params)
        self.reinitialize_network()
        return self


# pylint: disable=too-many-ancestors, too-many-instance-attributes
class PredictorWrapper(_Predictor, BaseEstimator, abc.ABC):  # type: ignore
    """Abstract wrapper for the Predictor class."""

    _T_default_criterion: LossFunction
    _T_default_metric: Metric

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        criterion: LossFunction | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
        **kwargs: Any,
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
        task_weights : Tensor or None, optional (default=None)
            Task weights.
        threshold : float or None, optional (default=None)
            Threshold for binary classification.
        output_transform : UnscaleTransform or None, optional (default=None)
            Transformations to apply to the output. None defaults to UnscaleTransform.
        kwargs : Any
            Additional keyword arguments.
        """
        if task_weights is None:
            task_weights = torch.ones(n_tasks)
        super().__init__(
            n_tasks=n_tasks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            criterion=criterion,
            output_transform=output_transform,
            **kwargs,
        )
        self.n_tasks = n_tasks
        self._input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        self.task_weights = task_weights
        self.threshold = threshold

    @property
    def input_dim(self) -> int:
        """Get the dimension of input."""
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value: int) -> None:
        """Set the dimension of input.

        Parameters
        ----------
        value : int
            The dimension of input.
        """
        self._input_dim = value

    @property
    def n_tasks(self) -> int:
        """Get the number of tasks."""
        return self._n_tasks

    @n_tasks.setter
    def n_tasks(self, value: int) -> None:
        """Set the number of tasks.

        Parameters
        ----------
        value : int
            The number of tasks.
        """
        self._n_tasks = value

    def reinitialize_fnn(self) -> Self:
        """Reinitialize the feedforward network.

        Returns
        -------
        Self
            The reinitialized feedforward network.
        """
        self.ffn = MLP.build(
            input_dim=self.input_dim,
            output_dim=self.n_tasks * self.n_targets,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            activation=self.activation,
        )
        return self

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the class and reinitialize the feedforward network.

        Parameters
        ----------
        **params: Any
            The parameters to set.

        Returns
        -------
        Self
            The model with the new parameters.
        """
        super().set_params(**params)
        self.reinitialize_fnn()
        return self


class RegressionFFN(PredictorWrapper, _RegressionFFN):  # type: ignore
    """A wrapper for the RegressionFFN class."""

    n_targets: int = 1
    _T_default_criterion = MSELoss
    _T_default_metric = MSEMetric


class MveFFN(PredictorWrapper, _MveFFN):  # type: ignore
    """A wrapper for the MveFFN class."""

    n_targets: int = 2
    _T_default_criterion = MVELoss


class EvidentialFFN(PredictorWrapper, _EvidentialFFN):  # type: ignore
    """A wrapper for the EvidentialFFN class."""

    n_targets: int = 4
    _T_default_criterion = EvidentialLoss


class BinaryClassificationFFN(PredictorWrapper, _BinaryClassificationFFN):  # type: ignore
    """A wrapper for the BinaryClassificationFFN class."""

    n_targets: int = 1
    _T_default_criterion = BCELoss
    _T_default_metric = BinaryAUROCMetric


class BinaryDirichletFFN(PredictorWrapper, _BinaryDirichletFFN):  # type: ignore
    """A wrapper for the BinaryDirichletFFN class."""

    n_targets: int = 2
    _T_default_criterion = BinaryDirichletLoss
    _T_default_metric = BinaryAUROCMetric


class MulticlassClassificationFFN(PredictorWrapper, _MulticlassClassificationFFN):  # type: ignore
    """A wrapper for the MulticlassClassificationFFN class."""

    n_targets: int = 1
    _T_default_criterion = CrossEntropyLoss
    _T_default_metric = CrossEntropyMetric

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        n_classes: int,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        criterion: LossFunction | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        """Initialize the MulticlassClassificationFFN class.

        Parameters
        ----------
        n_classes : int
            how many classes are expected in the output
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
        task_weights : Tensor or None, optional (default=None)
            Task weights.
        threshold : float or None, optional (default=None)
            Threshold for binary classification.
        output_transform : UnscaleTransform or None, optional (default=None)
            Transformations to apply to the output. None defaults to UnscaleTransform.
        """
        super().__init__(
            n_tasks,
            input_dim,
            hidden_dim,
            n_layers,
            dropout,
            activation,
            criterion,
            task_weights,
            threshold,
            output_transform,
            n_classes=n_classes,
        )

        self.n_classes = n_classes


class MulticlassDirichletFFN(PredictorWrapper, _MulticlassDirichletFFN):  # type: ignore
    """A wrapper for the MulticlassDirichletFFN class."""

    n_targets: int = 1
    _T_default_criterion = MulticlassDirichletLoss
    _T_default_metric = CrossEntropyMetric


class SpectralFFN(PredictorWrapper, _SpectralFFN):  # type: ignore
    """A wrapper for the SpectralFFN class."""

    n_targets: int = 1
    _T_default_criterion = SIDLoss
    _T_default_metric = SIDMetric


class MPNN(_MPNN, BaseEstimator):
    """A wrapper for the MPNN class.

    The MPNN is the main model class in chemprop. It consists of a message passing network, an aggregation function,
    and a feedforward network for prediction.
    """

    bn: nn.BatchNorm1d | nn.Identity

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: PredictorWrapper,
        batch_norm: bool = True,
        metric_list: Iterable[Metric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        """Initialize the MPNN class.

        Parameters
        ----------
        message_passing : MessagePassing
            The message passing network.
        agg : Aggregation
            The aggregation function.
        predictor : Predictor
            The predictor function.
        batch_norm : bool, optional (default=True)
            Whether to use batch normalization.
        metric_list : Iterable[Metric] | None, optional (default=None)
            The metrics to use for evaluation.
        warmup_epochs : int, optional (default=2)
            The number of epochs to use for the learning rate warmup.
        init_lr : float, optional (default=1e-4)
            The initial learning rate.
        max_lr : float, optional (default=1e-3)
            The maximum learning rate.
        final_lr : float, optional (default=1e-4)
            The final learning rate.
        """
        super().__init__(
            message_passing=message_passing,
            agg=agg,
            predictor=predictor,
            batch_norm=batch_norm,
            metrics=metric_list,
            warmup_epochs=warmup_epochs,
            init_lr=init_lr,
            max_lr=max_lr,
            final_lr=final_lr,
        )
        self.metric_list = metric_list
        self.batch_norm = batch_norm

    def reinitialize_network(self) -> Self:
        """Reinitialize the network with the current parameters.

        Returns
        -------
        Self
            The reinitialized network.
        """
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(self.message_passing.output_dim)
        else:
            self.bn = nn.Identity()

        if self.metric_list is None:
            # pylint: disable=protected-access
            self.metrics = [self.predictor._T_default_metric, self.criterion]
        else:
            self.metrics = list(self.metric_list) + [self.criterion]

        return self

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the class and update the network.

        Parameters
        ----------
        **params: Any
            The parameters to set.

        Returns
        -------
        Self
            The model with the new parameters.
        """
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
