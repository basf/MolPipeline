"""Wrapper for Chemprop to make it compatible with scikit-learn."""

import abc
from typing import Iterable, Self

import numpy as np
import numpy.typing as npt
from chemprop.data import BatchMolGraph, MoleculeDataset, MolGraphDataLoader
from chemprop.models.model import MPNN
from chemprop.nn import metrics
from chemprop.nn.agg import Aggregation, SumAggregation
from chemprop.nn.predictors import (
    BinaryClassificationFFNBase,
    MulticlassClassificationFFN,
)
from lightning import pytorch as pl
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
from torch import Tensor


class ABCChemprop(BaseEstimator, abc.ABC):
    """Wrap Chemprop in a sklearn like object."""

    model: MPNN

    def __init__(
        self,
        chemprop_model: MPNN,
        lightning_trainer: pl.Trainer | None = None,
        batch_size: int = 64,
        n_jobs: int = 1,
    ) -> None:
        """Initialize the chemprop abstract model.

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
        """
        self.model = chemprop_model
        self.lightning_trainer = lightning_trainer or pl.Trainer(max_epochs=10)
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def fit(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
        y: Iterable[int | float] | npt.NDArray[np.int_ | np.float_],
    ) -> Self:
        """Fit the model."""
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


class Chemprop(ABCChemprop):
    """Wrap Chemprop in a sklearn like Estimator."""

    def _is_binary_classifier(self) -> bool:
        if isinstance(self.model.predictor, BinaryClassificationFFNBase):
            return True
        return False

    def _is_multiclass_classifier(self) -> bool:
        if isinstance(self.model.predictor, MulticlassClassificationFFN):
            return True
        return False

    def _is_classifier(self) -> bool:
        return self._is_binary_classifier() or self._is_multiclass_classifier()

    def _predict(
        self, X: MoleculeDataset  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Predict the labels."""
        self.model.eval()
        test_data = MolGraphDataLoader(X, num_workers=self.n_jobs, shuffle=False)
        predictions = self.lightning_trainer.predict(self.model, test_data)
        return np.array([pred.numpy() for pred in predictions])  # type: ignore

    def predict(
        self, X: MoleculeDataset  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Predict the output."""
        predictions = self._predict(X)
        if predictions.shape[0] != len(X):
            raise AssertionError(
                "Predictions should have the same length as the input dataset."
            )
        if self._is_binary_classifier():
            if predictions.shape[1] != 1 or predictions.shape[2] != 1:
                raise ValueError(
                    "Binary classification model should output a single probability."
                )
            predictions = predictions[:, 0, 0]
            pred = np.zeros(len(predictions))
            pred[predictions > 0.5] = 1
            return pred
        if self._is_multiclass_classifier():
            return np.argmax(predictions, axis=1)
        return predictions

    @available_if(_is_classifier)
    def predict_proba(
        self, X: MoleculeDataset  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Predict the probabilities."""
        if self._is_binary_classifier():
            predictions = self._predict(X)
            if predictions.shape[1] != 1 or predictions.shape[2] != 1:
                raise ValueError(
                    "Binary classification model should output a single probability."
                )
            proba_cls1 = predictions[:, 0, 0]
            return np.vstack([1 - proba_cls1, proba_cls1]).T
        return self._predict(X)


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
        """Fit the model."""
        if self.disable_fitting:
            return self
        return super().fit(X, y)

    def transform(
        self, X: MoleculeDataset  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Transform the input."""
        self.model.eval()
        mol_data = [X[i].mg for i in range(len(X))]
        return self.model.fingerprint(BatchMolGraph(mol_data)).numpy()

    def fit_transform(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
        y: Iterable[int | float] | npt.NDArray[np.int_ | np.float_],
    ) -> npt.NDArray[np.float_]:
        """Fit the model and transform the input."""
        self.fit(X, y)
        return self.transform(X)
