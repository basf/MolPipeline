"""Wrapper for Chemprop to make it compatible with scikit-learn."""

from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
from sklearn.utils.metaestimators import available_if

try:
    from chemprop.data import MoleculeDataset, MolGraphDataLoader
    from chemprop.nn.predictors import (
        BinaryClassificationFFNBase,
        MulticlassClassificationFFN,
    )
    from lightning import pytorch as pl
except ImportError:
    pass


from molpipeline.estimators.chemprop.abstract import ABCChemprop
from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    SumAggregation,
)
from molpipeline.estimators.chemprop.neural_fingerprint import ChempropNeuralFP


class ChempropModel(ABCChemprop):
    """Wrap Chemprop in a sklearn like Estimator."""

    def _is_binary_classifier(self) -> bool:
        """Check if the model is a binary classifier.

        Returns
        -------
        bool
            True if the model is a binary classifier, False otherwise.
        """
        if isinstance(self.model.predictor, BinaryClassificationFFNBase):
            return True
        return False

    def _is_multiclass_classifier(self) -> bool:
        """Check if the model is a multiclass classifier.

        Returns
        -------
        bool
            True if the model is a multiclass classifier, False otherwise.
        """
        if isinstance(self.model.predictor, MulticlassClassificationFFN):
            return True
        return False

    def _is_classifier(self) -> bool:
        """Check if the model is a classifier.

        Returns
        -------
        bool
            True if the model is a classifier, False otherwise.
        """
        return self._is_binary_classifier() or self._is_multiclass_classifier()

    def _predict(
        self, X: MoleculeDataset  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Predict the labels.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.

        Returns
        -------
        npt.NDArray[np.float_]
            The predictions for the input data.
        """
        self.model.eval()
        test_data = MolGraphDataLoader(X, num_workers=self.n_jobs, shuffle=False)
        predictions = self.lightning_trainer.predict(self.model, test_data)
        prediction_array = np.array([pred.numpy() for pred in predictions])  # type: ignore

        # Check if the predictions have the same length as the input dataset
        if prediction_array.shape[0] != len(X):
            raise AssertionError(
                "Predictions should have the same length as the input dataset."
            )

        # If the model is a binary classifier, return the probability of the positive class
        if self._is_binary_classifier():
            if prediction_array.shape[1] != 1 or prediction_array.shape[2] != 1:
                raise ValueError(
                    "Binary classification model should output a single probability."
                )
            prediction_array = prediction_array[:, 0, 0]
        return prediction_array

    def predict(
        self, X: MoleculeDataset  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Predict the output.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.

        Returns
        -------
        npt.NDArray[np.float_]
            The predictions for the input data.
        """
        predictions = self._predict(X)
        if self._is_binary_classifier():
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
        """Predict the probabilities.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.

        Returns
        -------
        npt.NDArray[np.float_]
            The probabilities of the input data.
        """
        if self._is_binary_classifier():
            proba_class_1 = self._predict(X)
            return np.vstack([1 - proba_class_1, proba_class_1]).T
        return self._predict(X)

    def to_encoder(self) -> ChempropNeuralFP:
        """Return the encoder for the model.

        Returns
        -------
        ChempropNeuralFP
            The encoder for the model.
        """
        return ChempropNeuralFP(
            model=clone(self.model),  # type: ignore
            lightning_trainer=self.lightning_trainer,
            batch_size=self.batch_size,
            n_jobs=self.n_jobs,
            disable_fitting=True,
        )


class ChempropClassifier(ChempropModel):
    """Wrap Chemprop in a sklearn like classifier."""

    def __init__(
        self,
        model: MPNN | None = None,
        lightning_trainer: pl.Trainer | None = None,
        batch_size: int = 64,
        n_jobs: int = 1,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        """Initialize the chemprop classifier model.

        Parameters
        ----------
        model : MPNN | None, optional
            The chemprop model to wrap. If None, a default model will be used.
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
        if model is None:
            bond_encoder = BondMessagePassing()
            agg = SumAggregation()
            predictor = BinaryClassificationFFN()
            model = MPNN(message_passing=bond_encoder, agg=agg, predictor=predictor)
        super().__init__(
            model=model,
            lightning_trainer=lightning_trainer,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )
        if not self._is_binary_classifier():
            raise ValueError("ChempropClassifier should be a binary classifier.")

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the model and check if it is a binary classifier.

        Parameters
        ----------
        **params
            The parameters to set.

        Returns
        -------
        Self
            The model with the new parameters.
        """
        super().set_params(**params)
        if not self._is_binary_classifier():
            raise ValueError("ChempropClassifier should be a binary classifier.")
        return self