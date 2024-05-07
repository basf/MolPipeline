"""Parent class of wrapper for Chemprop to make it compatible with scikit-learn."""

import abc
from typing import Any, Iterable

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

# pylint: enable=duplicate-code


class ABCChemprop(BaseEstimator, abc.ABC):
    """Wrap Chemprop in a sklearn like object."""

    model: MPNN

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
        self.trainer_params = {}
        self.model_ckpoint_params = {}
        self.set_params(**kwargs)
        checkpoint_callback = []
        if self.model_ckpoint_params:
            checkpoint_callback = [
                pl.callbacks.ModelCheckpoint(**self.model_ckpoint_params)
            ]
        self.lightning_trainer = self._set_trainer(
            self.trainer_params, lightning_trainer, checkpoint_callback
        )

    def _set_trainer(self, trainer_params, lightning_trainer, checkpoint_callback):
        if self.trainer_params and lightning_trainer is not None:
            raise ValueError(
                "You must provide either trainer_params or lightning_trainer."
            )
        elif not trainer_params and lightning_trainer is None:
            lightning_trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                max_epochs=500,
                enable_model_summary=False,
                callbacks=checkpoint_callback,
            )
        elif trainer_params and lightning_trainer is None:
            lightning_trainer = pl.Trainer(
                **trainer_params, callbacks=checkpoint_callback
            )

        return lightning_trainer

    def fit(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
        y: Iterable[int | float] | npt.NDArray[np.int_ | np.float_],
    ) -> Self:
        """Fit the model to the data.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Iterable[int | float] | npt.NDArray[np.int_ | np.float_]
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

    def _update_trainer(args: Any) -> pl.Trainer:
        return pl.Trainer(**args)

    def set_params(self, **params) -> None:
        params, self.trainer_params = self._filter_params_trainer(params)
        params, self.model_ckpoint_params = self._filter_params_callback(params)
        super().set_params(**params)

    def get_params(self, deep: bool = False) -> None:
        params = super().get_params(deep)
        for name, value in self.trainer_params.items():
            params[f"lightning_trainer__{name}"] = value
        for name, value in self.model_ckpoint_params.items():
            params[f"callback_modelckpt__{name}"] = value
        params["lightning_trainer"] = (
            None  # set to none as we either have the trainer params or the non-parametrized trainer object (otherwise recursive from JSON fails as trainer + params are set)
        )
        return params

    def _filter_params_trainer(self, params: dict) -> dict:
        params_trainer = {
            k.split("__")[1]: v
            for k, v in params.items()
            if k.startswith("lightning_trainer")
        }
        params = {
            k: v for k, v in params.items() if not k.startswith("lightning_trainer")
        }
        return params, params_trainer

    def _filter_params_callback(self, params: dict) -> dict:
        params_ckpt = {
            k.split("__")[1]: v
            for k, v in params.items()
            if k.startswith("callback_modelckpt")
        }
        params = {
            k: v for k, v in params.items() if not k.startswith("callback_modelckpt")
        }
        return params, params_ckpt
