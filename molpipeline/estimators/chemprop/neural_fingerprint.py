"""Wrap Chemprop in a transformer returning the neural fingerprint as a numpy array."""

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from chemprop.data import BatchMolGraph, MoleculeDataset
from chemprop.models.model import MPNN
from lightning import pytorch as pl
from typing_extensions import override

from molpipeline.estimators.chemprop.abstract import ABCChemprop


class ChempropNeuralFP(ABCChemprop):
    """Wrap Chemprop in a transformer returning the neural fingerprint as a numpy array.

    This class is not a (grand-) child of MolToAnyPipelineElement, as it does not
    support the `pretransform_single` method. To maintain compatibility with the
    MolToAnyPipelineElement, the `output_type` property is implemented.
    It can be used as any other transformer in the pipeline, except in the
    `MolToConcatenatedVector`.

    """

    @property
    def output_type(self) -> str:
        """Return the output type of the transformer."""
        return "float"

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
        self.disable_fitting = disable_fitting
        super().__init__(
            model=model,
            lightning_trainer=lightning_trainer,
            batch_size=batch_size,
            n_jobs=n_jobs,
            **kwargs,
        )

    @override
    def fit(
        self,
        X: MoleculeDataset,
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
        *,
        sample_weight: Sequence[float] | npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Fit the model.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Sequence[int | float] | npt.NDArray[np.int_ | np.float64]
            The target data.
        sample_weight : npt.NDArray[np.float64] | None, optional
            Sample weights for the input data, by default None.

        Returns
        -------
        Self
            The fitted model.

        """
        if self.disable_fitting:
            return self
        return super().fit(X, y, sample_weight=sample_weight)

    def transform(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name  # noqa: N803
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
        X: MoleculeDataset,  # pylint: disable=invalid-name  # noqa: N803
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
        *,
        sample_weight: Sequence[float] | npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Fit the model and transform the input.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Sequence[int | float] | npt.NDArray[np.int_ | np.float64]
            The target data.
        sample_weight : npt.NDArray[np.float64] | None, optional
            Sample weights for the input data, by default None.

        Returns
        -------
        npt.NDArray[np.float64]
            The neural fingerprint of the input data.

        """
        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X)
