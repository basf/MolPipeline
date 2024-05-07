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

from molpipeline.estimators.chemprop.component_wrapper import (
    get_lightning_trainer_params,
)

# pylint: enable=duplicate-code


class ABCChemprop(BaseEstimator, abc.ABC):
    """Wrap Chemprop in a sklearn like object."""

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
        self.trainer_params = get_lightning_trainer_params(lightning_trainer)
        self.set_params(**kwargs)

    def _update_trainer(
        self,
    ) -> None:
        """Set the trainer for the model.

        Parameters
        ----------
        trainer_params : dict[str, Any]
            The parameters for the trainer.
        lightning_trainer : pl.Trainer | None
            The lightning trainer.
        checkpoint_callback : list[pl.callbacks.ModelCheckpoint]
            The checkpoint callback to use.
        """
        trainer_params = dict(self.trainer_params)
        if self.model_ckpoint_params:
            trainer_params["callbacks"] = [
                pl.callbacks.ModelCheckpoint(**self.model_ckpoint_params)
            ]
        self.lightning_trainer = pl.Trainer(**trainer_params)

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
        params, self.trainer_params = self._filter_params_trainer(params)
        params, self.model_ckpoint_params = self._filter_params_callback(params)
        super().set_params(**params)
        self._update_trainer()
        return self

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        """Get the parameters of the model.

        Parameters
        ----------
        deep : bool, optional (default=False)
            Whether to get the parameters of the model.

        Returns
        -------
        dict[str, Any]
            The parameters of the model.
        """
        params = super().get_params(deep)
        for name, value in self.trainer_params.items():
            params[f"lightning_trainer__{name}"] = value
        for name, value in self.model_ckpoint_params.items():
            params[f"callback_modelckpt__{name}"] = value
        # set to none as the trainer is created from the parameters
        return params

    @staticmethod
    def _filter_params_trainer(
        params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Filter the parameters for the trainer.

        Parameters
        ----------
        params : dict[str, Any]
            The parameters to filter.

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
            if key.startswith("lightning_trainer"):
                trainer_params[key.split("__")[1]] = value
            else:
                other_params[key] = value
        return other_params, trainer_params

    @staticmethod
    def _filter_params_callback(
        params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Filter the parameters for the checkpoint callback.

        Parameters
        ----------
        params : dict[str, Any]
            The parameters to filter.

        Returns
        -------
        dict[str, Any]
            The filtered parameters for the model.
        dict[str, Any]
            The filtered parameters for the checkpoint callback.
        """
        checkpoint_params = {}
        other_params = {}
        for key, value in params.items():
            if key.startswith("callback_modelckpt"):
                checkpoint_params[key.split("__")[1]] = value
            else:
                other_params[key] = value
        return params, checkpoint_params
