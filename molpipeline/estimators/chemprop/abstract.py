"""Parent class of wrapper for Chemprop to make it compatible with scikit-learn."""

import abc
from typing import Any, Iterable, Self

import numpy as np
import numpy.typing as npt
from chemprop.data import MoleculeDataset, MolGraphDataLoader
from chemprop.models.model import MPNN
from lightning import pytorch as pl
from sklearn.base import BaseEstimator


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
        default_trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            max_epochs=500,
            enable_model_summary=False,
            callbacks=False,
        )
        self.lightning_trainer = lightning_trainer or default_trainer
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.set_params(**kwargs)

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
        training_data = MolGraphDataLoader(
            X, batch_size=self.batch_size, num_workers=self.n_jobs
        )
        self.lightning_trainer.fit(self.model, training_data)
        return self
