"""Wrapper for Chemprop to make it compatible with scikit-learn."""

import warnings
from collections.abc import Sequence
from typing import Any, Literal, Self

try:
    from typing import override  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import override

import numpy as np
import numpy.typing as npt
from chemprop.data import MoleculeDataset, build_dataloader
from chemprop.nn.predictors import BinaryClassificationFFNBase
from lightning import pytorch as pl
from sklearn.base import clone
from sklearn.utils._tags import (
    ClassifierTags,  # noqa: PLC2701
    RegressorTags,  # noqa: PLC2701
    Tags,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.metaestimators import available_if

from molpipeline.estimators.chemprop.abstract import ABCChemprop
from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    MulticlassClassificationFFN,
    RegressionFFN,
    SumAggregation,
)
from molpipeline.estimators.chemprop.neural_fingerprint import ChempropNeuralFP


class ChempropModel(ABCChemprop):
    """Wrap Chemprop in a sklearn like Estimator."""

    _classes_: npt.NDArray[np.int_] | None

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
        super().__init__(
            model=model,
            lightning_trainer=lightning_trainer,
            batch_size=batch_size,
            n_jobs=n_jobs,
            **kwargs,
        )
        self._classes_ = None

    @property
    def classes_(self) -> npt.NDArray[np.int_]:
        """Return the classes.

        Raises
        ------
        ValueError
            If the model is not a classifier.
        ValueError
            If the classes are not set.

        """
        if not self._is_classifier():
            raise ValueError("Model is not a classifier.")
        if self._classes_ is None:
            raise ValueError("Classes are not set.")
        return self._classes_

    @property
    def _estimator_type(self) -> str:
        """Return the estimator type."""
        if self._is_classifier():
            return "classifier"
        return "regressor"

    def _is_binary_classifier(self) -> bool:
        """Check if the model is a binary classifier.

        Returns
        -------
        bool
            True if the model is a binary classifier, False otherwise.

        """
        return isinstance(self.model.predictor, BinaryClassificationFFNBase)

    def _is_multiclass_classifier(self) -> bool:
        """Check if the model is a multiclass classifier.

        Returns
        -------
        bool
            True if the model is a multiclass classifier, False otherwise.

        """
        return isinstance(self.model.predictor, MulticlassClassificationFFN)

    def _is_classifier(self) -> bool:
        """Check if the model is a classifier.

        Returns
        -------
        bool
            True if the model is a classifier, False otherwise.

        """
        return self._is_binary_classifier() or self._is_multiclass_classifier()

    def __sklearn_tags__(self) -> Tags:  # noqa: PLW3201
        """Return the sklearn tags.

        Returns
        -------
        Tags
            The sklearn tags for the model.

        """
        tags = super().__sklearn_tags__()
        if self._is_classifier():
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
        else:
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags

    def _predict(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name  # noqa: N803
    ) -> npt.NDArray[np.float64]:
        """Predict the labels.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.

        Raises
        ------
        AssertionError
            If the predictions do not have the same length as the input dataset.
        ValueError
            If the model is a binary classifier and the predictions do not have the
            correct shape.

        Returns
        -------
        npt.NDArray[np.float64]
            The predictions for the input data.

        """
        self.model.eval()
        test_data = build_dataloader(X, num_workers=self.n_jobs, shuffle=False)
        predictions = self.lightning_trainer.predict(self.model, test_data)
        prediction_array = np.vstack(predictions)  # type: ignore
        if prediction_array.shape[1] == 1:
            prediction_array = prediction_array.squeeze(axis=1)
        # Check if the predictions have the same length as the input dataset
        if prediction_array.shape[0] != len(X):
            raise AssertionError(
                "Predictions should have the same length as the input dataset.",
            )

        # If the model is a binary classifier, return the probability of the pos class
        if self._is_binary_classifier() and prediction_array.ndim != 1:
            raise ValueError(
                "Binary classification model should output a single probability.",
            )
        return prediction_array

    @override
    def fit(
        self,
        X: MoleculeDataset,
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
        *,
        sample_weight: Sequence[float] | npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Fit the model to the data.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Sequence[int | float] | npt.NDArray[np.int_ | np.float64]
            The target data.
        sample_weight : Sequence[float] |npt.NDArray[np.float64] | None, optional
            The sample weights.

        Returns
        -------
        Self
            The fitted model.

        """
        if self._is_classifier():
            self._classes_ = np.unique(y)
        return super().fit(X, y, sample_weight=sample_weight)

    @override
    def predict(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Predict the output.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.

        Returns
        -------
        npt.NDArray[np.float64]
            The predictions for the input data.

        """
        predictions = self._predict(X)
        if self._is_binary_classifier():
            pred = np.zeros(len(predictions))
            pred[predictions > 0.5] = 1  # noqa: PLR2004
            return pred

        if self._is_multiclass_classifier():
            return np.argmax(predictions, axis=1)

        return predictions

    @override
    @available_if(_is_classifier)
    def predict_proba(
        self,
        X: MoleculeDataset,  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Predict the probabilities.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.

        Returns
        -------
        npt.NDArray[np.float64]
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
    """Chemprop model with default parameters for binary classification tasks."""

    def __init__(
        self,
        model: MPNN | None = None,
        lightning_trainer: pl.Trainer | None = None,
        class_weight: Literal["balanced"] | dict[int, float] | None = None,
        batch_size: int = 64,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the chemprop classifier model.

        Parameters
        ----------
        model : MPNN | None, optional
            The chemprop model to wrap. If None, a default model will be used.
        lightning_trainer : pl.Trainer, optional
            The lightning trainer to use, by default None
        class_weight : Literal["balanced"] | dict[int, float] | None, optional
            The class weights to use, by default None
            If "balanced", the class weights will be calculated using the
            sklearn.utils.class_weight.compute_class_weight function.
            If a dict is provided, it should map class labels to weights.
            If sample_weight is provided during fitting, class_weight will be multiplied
            with sample_weight.
        batch_size : int, optional (default=64)
            The batch size to use.
        n_jobs : int, optional (default=1)
            The number of jobs to use.
        kwargs : Any
            Parameters set using `set_params`.
            Can be used to modify components of the model.

        Raises
        ------
        ValueError
            If the model's predictor is not a binary classifier.

        """
        if model is None:
            bond_encoder = BondMessagePassing()
            agg = SumAggregation()
            predictor = BinaryClassificationFFN()
            model = MPNN(message_passing=bond_encoder, agg=agg, predictor=predictor)
        self.class_weight = class_weight
        super().__init__(
            model=model,
            lightning_trainer=lightning_trainer,
            batch_size=batch_size,
            n_jobs=n_jobs,
            **kwargs,
        )
        if not self._is_binary_classifier():
            raise ValueError("ChempropClassifier should be a binary classifier.")

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Handle unpickling with backward compatibility.

        Parameters
        ----------
        state : dict[str, Any]
            The object's state dictionary.

        """
        if "class_weight" not in state:
            warnings.warn(
                "Loading old ChempropClassifier without class_weight."
                " Setting class_weight to None.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.class_weight = None
        super().__setstate__(state)

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the model and check if it is a binary classifier.

        Parameters
        ----------
        **params
            The parameters to set.

        Raises
        ------
        ValueError
            If the model's predictor is not a binary classifier.

        Returns
        -------
        Self
            The model with the new parameters.

        """
        super().set_params(**params)
        if not self._is_binary_classifier():
            raise ValueError("ChempropClassifier should be a binary classifier.")
        return self

    @override
    def fit(
        self,
        X: MoleculeDataset,
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
        *,
        sample_weight: Sequence[float] | npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Fit the model to the data.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Sequence[int | float] | npt.NDArray[np.int_ | np.float64]
            The target data.
        sample_weight : Sequence[float] | npt.NDArray[np.float64] | None, optional
            The sample weights. If class_weight is set, the sample weights will be
            multiplied with the class weights.

        Returns
        -------
        Self
            The fitted model.

        """
        if self.class_weight is not None:
            class_weights = compute_sample_weight(
                class_weight=self.class_weight,
                y=y,
            )
            if sample_weight is not None:
                sample_weight *= class_weights
            else:
                sample_weight = class_weights
        return super().fit(X, y, sample_weight=sample_weight)


class ChempropRegressor(ChempropModel):
    """Chemprop model with default parameters for regression tasks."""

    def __init__(
        self,
        model: MPNN | None = None,
        lightning_trainer: pl.Trainer | None = None,
        n_tasks: int = 1,
        batch_size: int = 64,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the chemprop regressor model.

        Parameters
        ----------
        model : MPNN | None, optional
            The chemprop model to wrap. If None, a default model will be used.
        lightning_trainer : pl.Trainer, optional
            The lightning trainer to use, by default None
        n_tasks : int
            The number of tasks for the regressor, e.g. number of target variables.
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
            predictor = RegressionFFN(n_tasks=n_tasks)
            model = MPNN(message_passing=bond_encoder, agg=agg, predictor=predictor)
        self.n_tasks = n_tasks
        super().__init__(
            model=model,
            lightning_trainer=lightning_trainer,
            batch_size=batch_size,
            n_jobs=n_jobs,
            **kwargs,
        )


class ChempropMulticlassClassifier(ChempropModel):
    """Chemprop model with default parameters for multiclass classification tasks."""

    def __init__(
        self,
        n_classes: int,
        model: MPNN | None = None,
        lightning_trainer: pl.Trainer | None = None,
        batch_size: int = 64,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the chemprop multiclass model.

        Parameters
        ----------
        n_classes : int
            The number of classes for the classifier.
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

        Raises
        ------
        AttributeError
            If the passed model.predictor does not have an attribute n_classes.
        ValueError
            If the number of classes in the predictor does not match the number of
            classes given as attribute.

        """
        if model is None:
            bond_encoder = BondMessagePassing()
            agg = SumAggregation()
            predictor = MulticlassClassificationFFN(n_classes=n_classes)
            model = MPNN(message_passing=bond_encoder, agg=agg, predictor=predictor)
        if not hasattr(model.predictor, "n_classes"):
            raise AttributeError(
                "The predictor does not have an attribute n_classes. Please use a "
                "MulticlassClassificationFFN predictor or define n_classes.",
            )
        if n_classes != model.predictor.n_classes:
            raise ValueError(
                "The number of classes in the predictor does not match the number of "
                "classes.",
            )
        super().__init__(
            model=model,
            lightning_trainer=lightning_trainer,
            batch_size=batch_size,
            n_jobs=n_jobs,
            **kwargs,
        )
        self._is_valid_multiclass_classifier()

    @property
    def n_classes(self) -> int:
        """Return the number of classes."""
        return self.model.predictor.n_classes

    @n_classes.setter
    def n_classes(self, n_classes: int) -> None:
        """Set the number of classes.

        Parameters
        ----------
        n_classes : int
            Number of classes.

        """
        self.model.predictor.n_classes = n_classes
        self.model.reinitialize_network()

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the model and check if it is a multiclass classifier.

        Parameters
        ----------
        **params
            The parameters to set.

        Raises
        ------
        ValueError
            If the model's predictor or the number of classes are invalid.

        Returns
        -------
        Self
            The model with the new parameters.

        """
        super().set_params(**params)
        if not self._is_valid_multiclass_classifier():
            raise ValueError(
                "The model's predictor or the number of classes are invalid. "
                "Use a multiclass predictor and more than 2 classes.",
            )
        return self

    @override
    def fit(
        self,
        X: MoleculeDataset,
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
        *,
        sample_weight: Sequence[float] | npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Fit the model to the data.

        Parameters
        ----------
        X : MoleculeDataset
            The input data.
        y : Sequence[int | float] | npt.NDArray[np.int_ | np.float64]
            The target data.
        sample_weight : Sequence[float] | npt.NDArray[np.float64] | None, optional
            The sample weights.

        Returns
        -------
        Self
            The fitted model.

        """
        self._check_correct_input(y)
        return super().fit(X, y, sample_weight=sample_weight)

    def _check_correct_input(
        self,
        y: Sequence[int | float] | npt.NDArray[np.int_ | np.float64],
    ) -> None:
        """Check if the input for the multi-class classifier is correct.

        Parameters
        ----------
        y : Sequence[int | float] | npt.NDArray[np.int_ | np.float64]
            Indended classes for the dataset

        Raises
        ------
        ValueError
            If the classes found in y are not matching n_classes or if the class labels
            do not start from 0 to n_classes-1.

        """
        unique_y = np.unique(y)
        log = []
        if self.n_classes != len(unique_y):
            log.append(
                f"Given number of classes in init (n_classes) does not match the "
                f"number of unique classes (found {unique_y}) in the target data.",
            )
        if sorted(unique_y) != list(range(self.n_classes)):
            err = (
                f"Classes need to be in the range from 0 to {self.n_classes - 1}. "
                f"Found {unique_y}. Please correct the input data accordingly."
            )
            log.append(err)
        if log:
            raise ValueError("\n".join(log))

    def _is_valid_multiclass_classifier(self) -> bool:
        """Check if a multiclass classifier is valid.

        Model FFN needs to be of the correct class and model needs to have more than
        2 classes.

        Returns
        -------
        bool
            True if is a valid multiclass classifier, False otherwise.

        """
        has_correct_model = isinstance(
            self.model.predictor,
            MulticlassClassificationFFN,
        )
        has_classes = self.n_classes > 2  # noqa: PLR2004
        return has_correct_model and has_classes
