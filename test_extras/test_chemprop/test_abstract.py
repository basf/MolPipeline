"""Tests for the abstract class ABCChemprop."""

import unittest

from chemprop.data import MoleculeDataset
from lightning import pytorch as pl

from molpipeline.any2mol import SmilesToMol
from molpipeline.estimators.chemprop.abstract import ABCChemprop
from molpipeline.mol2any.mol2chemprop import MolToChemprop
from test_extras.test_chemprop.chemprop_test_utils.default_models import (
    get_binary_classification_mpnn,
)


class TestABCChemprop(unittest.TestCase):
    """Test static methods of the Chemprop model."""

    @staticmethod
    def _get_training_data() -> tuple[MoleculeDataset, list[int]]:
        """Create a tiny classification dataset for callback fit tests.

        Returns
        -------
        MoleculeDataset
            Molecular dataset for a minimal training run.
        list[int]
            Corresponding labels for the dataset.

        """
        molecules = SmilesToMol().transform(["CCC", "CCO", "CCN"])
        dataset = MolToChemprop().transform(molecules)
        return dataset, [1, 0, 1]

    def test_filter_params_callback(self) -> None:
        """Test the filter_params_callback method."""
        dummy_params = {
            "callback_modelckpt__monitor": "val_loss",
            "other__param": "value",
        }
        other_params, callback_params = ABCChemprop._filter_params(
            dummy_params,
            "callback_modelckpt",
        )
        # pylint: enable=protected-access
        self.assertEqual(callback_params, {"monitor": "val_loss"})
        self.assertEqual(other_params, {"other__param": "value"})

    def test_filter_params_trainer(self) -> None:
        """Test the filter_params_trainer method."""
        dummy_params = {
            "lightning_trainer__max_epochs": 50,
            "other__param": "value",
        }
        other_params, trainer_params = ABCChemprop._filter_params(
            dummy_params,
            "lightning_trainer",
        )
        # pylint: enable=protected-access
        self.assertEqual(trainer_params, {"max_epochs": 50})
        self.assertEqual(other_params, {"other__param": "value"})

    def test_trainer_callbacks_direct(self) -> None:
        """Test callbacks passed through a Trainer are retained."""
        expected_callbacks = [
            pl.callbacks.Callback(),
            pl.callbacks.lr_monitor.LearningRateMonitor(),
        ]
        trainer = pl.Trainer(
            callbacks=expected_callbacks,
            accelerator="cpu",
            max_epochs=1,
            enable_model_summary=False,  # Adds an extra callback
            enable_progress_bar=False,  # Adds an extra callback
        )
        estimator = ABCChemprop(
            model=get_binary_classification_mpnn(),
            lightning_trainer=trainer,
        )
        actual_callbacks = list(estimator.lightning_trainer.callbacks)  # type: ignore[attr-defined]
        self.assertEqual(len(expected_callbacks), len(actual_callbacks))
        for expected_callback, actual_callback in zip(
            expected_callbacks,
            actual_callbacks,
            strict=True,
        ):
            self.assertIs(expected_callback, actual_callback)

        training_data, labels = self._get_training_data()
        estimator.fit(training_data, labels)

        actual_callbacks = list(estimator.lightning_trainer.callbacks)  # type: ignore[attr-defined]
        self.assertEqual(len(expected_callbacks), len(actual_callbacks))
        for expected_callback, actual_callback in zip(
            expected_callbacks,
            actual_callbacks,
            strict=True,
        ):
            self.assertIs(expected_callback, actual_callback)

    def test_trainer_callback_indirect(self) -> None:
        """Test callbacks passed as ABCChemprop trainer kwargs are retained."""
        expected_callbacks = [
            pl.callbacks.Callback(),
            pl.callbacks.LearningRateMonitor(),
        ]
        estimator = ABCChemprop(
            model=get_binary_classification_mpnn(),
            lightning_trainer__callbacks=expected_callbacks,
            lightning_trainer__accelerator="cpu",
            lightning_trainer__max_epochs=1,
        )
        actual_callbacks = list(estimator.lightning_trainer.callbacks)  # type: ignore[attr-defined]
        self.assertEqual(len(expected_callbacks), len(actual_callbacks))
        for expected_callback, actual_callback in zip(
            expected_callbacks,
            actual_callbacks,
            strict=True,
        ):
            self.assertIs(expected_callback, actual_callback)

        training_data, labels = self._get_training_data()
        estimator.fit(training_data, labels)

        actual_callbacks = list(estimator.lightning_trainer.callbacks)  # type: ignore[attr-defined]
        self.assertEqual(len(expected_callbacks), len(actual_callbacks))
        for expected_callback, actual_callback in zip(
            expected_callbacks,
            actual_callbacks,
            strict=True,
        ):
            self.assertIs(expected_callback, actual_callback)
