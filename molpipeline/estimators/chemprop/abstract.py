"""Parent class of wrapper for Chemprop to make it compatible with scikit-learn."""

import abc
from typing import Any, Sequence

# pylint: disable=duplicate-code
try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt

try:
    from chemprop.data import MoleculeDataset, build_dataloader
    from chemprop.models.model import MPNN
    from lightning import pytorch as pl
except ImportError:
    pass
from sklearn.base import BaseEstimator

from molpipeline.estimators.chemprop.lightning_wrapper import get_params_trainer

# pylint: enable=duplicate-code


class ABCChemprop(BaseEstimator, abc.ABC):
    """Wrap Chemprop in a sklearn compatible Estimator.

    Notes
    -----
    As the ChempropNeuralFP is a transformer and the ChempropModel is a predictor, this class
    provides the common functionality for both classes.
    Although this class does not implement abstract methods, it is marked as abstract to prevent
    instantiation. (without predict or transform methods this class is useless.)

    Attributes
    ----------
    model : MPNN
        The chemprop model to wrap.
    batch_size : int
        The batch size to use for training and prediction.
    n_jobs : int
        The number of jobs to use for processing the molecular graphs in the dataloader.
    lightning_trainer : pl.Trainer
        The lightning trainer to use for training the model.
    trainer_params : dict[str, Any]
        The parameters of the lightning trainer. This is used as the trainer is not compatible with the
        `get_params` method.
    model_ckpoint_params : dict[str, Any]
        The parameters of the model checkpoint callback. This is used as the callback is not compatible with the
        `get_params` method.
    """

    model: MPNN
    batch_size: int
    n_jobs: int
    lightning_trainer: pl.Trainer
    trainer_params: dict[str, Any]
    model_ckpoint_params: dict[str, Any]

    def __init__(
        self,
        model: MPNN,
        lightning_trainer: pl.Trainer | None = None,
        batch_size: int = 64,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the chemprop abstract model.

        Parameters
        ----------
        model : MPNN
            The chemprop model to wrap.
        lightning_trainer : pl.Trainer, optional
            The lightning trainer to use, by default None
        batch_size : int, optional (default=64)
            The batch size to use.
        n_jobs : int, optional (default=1)
            The number of jobs to use.
        kwargs : Any
            Parameters set using `set_params`.
            Can be used to modify components of the model.
        """
        self.model = model
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.model_ckpoint_params = {}
        if not lightning_trainer:
            lightning_trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                max_epochs=500,
                enable_model_summary=False,
                callbacks=[],
            )
        self.lightning_trainer = lightning_trainer
        self.trainer_params = get_params_trainer(self.lightning_trainer)
        self.set_params(**kwargs)

    def _update_trainer(
        self,
    ) -> None:
        """Update the trainer for the model."""
        trainer_params = dict(self.trainer_params)
        if self.model_ckpoint_params:
            trainer_params["callbacks"] = [
                pl.callbacks.ModelCheckpoint(**self.model_ckpoint_params)
            ]
        self.lightning_trainer = pl.Trainer(**trainer_params)

    def fit(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
    ) -> Self:
        """Fit the model to the data.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Sequence[int | float] | npt.NDArray[np.int_ | np.float64]
            The target data.

        Returns
        -------
        Self
            The fitted model.
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        X.Y = y
        training_data = build_dataloader(
            X, batch_size=self.batch_size, num_workers=self.n_jobs
        )
        self.lightning_trainer.fit(self.model, training_data)
        return self

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the model.

        Note
        ----
        Parameters for the trainer and the checkpoint callback are filtered out and added as attributes of the model.
        This is done due to incompatibility with the `get_params` method.

        Parameters
        ----------
        **params: Any
            The parameters to set.

        Returns
        -------
        Self
            The model with the new parameters.
        """
        params, trainer_params = self._filter_params(params, "lightning_trainer")
        params, model_ckpoint_params = self._filter_params(params, "callback_modelckpt")
        self.trainer_params.update(trainer_params)
        self.model_ckpoint_params.update(model_ckpoint_params)
        super().set_params(**params)
        self._update_trainer()
        return self

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        """Get the parameters of the model.

        Parameters
        ----------
        deep : bool, optional (default=False)
            Whether to get the parameters of the model.

        Notes
        -----
        The parameters of the trainer and the model checkpoint are added irrespective of the `deep` parameter.
        This is done due to their incompatibility with the `get_params` and `set_params` methods.

        Returns
        -------
        dict[str, Any]
            The parameters of the model.
        """
        params = super().get_params(deep)
        # Since the trainer and the model checkpoint are not compatible with the `get_params` and `set_params` methods
        # they are not passed as objects but as parameters. Hence, the `deep` parameter is ignored and the parameters
        # are always returned.
        for name, value in self.trainer_params.items():
            params[f"lightning_trainer__{name}"] = value
        for name, value in self.model_ckpoint_params.items():
            params[f"callback_modelckpt__{name}"] = value
        # set to none as the trainer is created from the parameters
        # Otherwise, the sklearn clone will fail as the trainer is updated by replacing the object
        params["lightning_trainer"] = None
        return params

    @staticmethod
    def _filter_params(
        params: dict[str, Any],
        prefix: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Filter the parameters for the specified prefix.

        Parameters
        ----------
        params : dict[str, Any]
            The parameters to filter.
        prefix : str
            The prefix to filter the parameters for.

        Returns
        -------
        dict[str, Any]
            The filtered parameters for the model.
        dict[str, Any]
            The filtered parameters for the trainer.
        """
        trainer_params = {}
        other_params = {}
        for key, value in params.items():
            if key.startswith(f"{prefix}__"):
                trainer_params[key.split("__")[1]] = value
            else:
                other_params[key] = value
        return other_params, trainer_params
