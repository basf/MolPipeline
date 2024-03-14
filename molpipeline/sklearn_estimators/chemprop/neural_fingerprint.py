"""Wrap Chemprop in a sklearn like transformer returning the neural fingerprint as a numpy array."""

from typing import Iterable, Self

import numpy as np
import numpy.typing as npt
from chemprop.data import BatchMolGraph, MoleculeDataset
from chemprop.models.model import MPNN
from lightning import pytorch as pl

from molpipeline.sklearn_estimators.chemprop.abstract import ABCChemprop


class ChempropNeuralFP(ABCChemprop):
    """Wrap Chemprop in a sklearn like transformer returning the neural fingerprint as a numpy array."""

    def __init__(
        self,
        chemprop_model: MPNN,
        lightning_trainer: pl.Trainer | None = None,
        batch_size: int = 64,
        n_jobs: int = 1,
        disable_fitting: bool = False,
    ) -> None:
        """Initialize the chemprop neural fingerprint model.

        Parameters
        ----------
        chemprop_model : MPNN
            The chemprop model to wrap.
        lightning_trainer : pl.Trainer, optional
            The lightning trainer to use, by default None
        batch_size : int, optional (default=64)
            The batch size to use.
        n_jobs : int, optional (default=1)
            The number of jobs to use.
        disable_fitting : bool, optional (default=False)
            Whether to allow fitting or set to fixed encoding.
        """
        # pylint: disable=duplicate-code
        super().__init__(
            chemprop_model=chemprop_model,
            lightning_trainer=lightning_trainer,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )
        self.disable_fitting = disable_fitting

    def fit(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
        y: Iterable[int | float] | npt.NDArray[np.int_ | np.float_],
    ) -> Self:
        """Fit the model.

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
        if self.disable_fitting:
            return self
        return super().fit(X, y)

    def transform(
        self, X: MoleculeDataset  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Transform the input.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.

        Returns
        -------
        npt.NDArray[np.float_]
            The neural fingerprint of the input data.
        """
        self.model.eval()
        mol_data = [X[i].mg for i in range(len(X))]
        return self.model.fingerprint(BatchMolGraph(mol_data)).numpy()

    def fit_transform(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
        y: Iterable[int | float] | npt.NDArray[np.int_ | np.float_],
    ) -> npt.NDArray[np.float_]:
        """Fit the model and transform the input.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Iterable[int | float] | npt.NDArray[np.int_ | np.float_]
            The target data.

        Returns
        -------
        npt.NDArray[np.float_]
            The neural fingerprint of the input data.
        """
        self.fit(X, y)
        return self.transform(X)
