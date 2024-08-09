"""Wrap Chemprop in a sklearn like transformer returning the neural fingerprint as a numpy array."""

from typing import Any, Self, Sequence

import numpy as np
import numpy.typing as npt
from chemprop.data import BatchMolGraph, MoleculeDataset
from chemprop.models.model import MPNN
from lightning import pytorch as pl

from molpipeline.estimators.chemprop.abstract import ABCChemprop


class ChempropNeuralFP(ABCChemprop):
    """Wrap Chemprop in a sklearn like transformer returning the neural fingerprint as a numpy array."""

    def __init__(
        self,
        model: MPNN,
        lightning_trainer: pl.Trainer | None = None,
        batch_size: int = 64,
        n_jobs: int = 1,
        disable_fitting: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the chemprop neural fingerprint model.

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
        disable_fitting : bool, optional (default=False)
            Whether to allow fitting or set to fixed encoding.
        **kwargs: Any
            Parameters for components of the model.
        """
        # pylint: disable=duplicate-code
        super().__init__(
            model=model,
            lightning_trainer=lightning_trainer,
            batch_size=batch_size,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.disable_fitting = disable_fitting

    def fit(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
    ) -> Self:
        """Fit the model.

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
        if self.disable_fitting:
            return self
        return super().fit(X, y)

    def transform(
        self, X: MoleculeDataset  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Transform the input.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.

        Returns
        -------
        npt.NDArray[np.float64]
            The neural fingerprint of the input data.
        """
        self.model.eval()
        mol_data = [X[i].mg for i in range(len(X))]
        return self.model.fingerprint(BatchMolGraph(mol_data)).detach().numpy()

    def fit_transform(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
    ) -> npt.NDArray[np.float64]:
        """Fit the model and transform the input.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Sequence[int | float] | npt.NDArray[np.int_ | np.float64]
            The target data.

        Returns
        -------
        npt.NDArray[np.float64]
            The neural fingerprint of the input data.
        """
        self.fit(X, y)
        return self.transform(X)
