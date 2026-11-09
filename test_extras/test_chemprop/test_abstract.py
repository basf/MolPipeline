"""Tests for the abstract class ABCChemprop."""

import tempfile
import unittest

import torch
from chemprop.data import MoleculeDataset
from lightning import pytorch as pl

from molpipeline.any2mol import SmilesToMol
from molpipeline.estimators.chemprop.abstract import ABCChemprop
from molpipeline.estimators.chemprop.models import ChempropClassifier, ChempropRegressor
from molpipeline.mol2any import MolToChemprop
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
        self.assertEqual(trainer_params, {"max_epochs": 50})
        self.assertEqual(other_params, {"other__param": "value"})

    def test_trainer_callbacks_direct(self) -> None:
        """Test callbacks passed through a Trainer are retained."""
        learning_rate_callback = pl.callbacks.LearningRateMonitor()
        expected_callbacks = [pl.callbacks.Callback(), learning_rate_callback]
        trainer = pl.Trainer(
            callbacks=expected_callbacks,
            accelerator="cpu",
            max_epochs=5,
            log_every_n_steps=1,
            enable_model_summary=False,  # Adds an extra callback otherwise
            enable_progress_bar=False,  # Adds an extra callback otherwise
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
        self.assertEqual(len(learning_rate_callback.lrs["lr-Adam"]), 5)

    def test_trainer_callback_indirect(self) -> None:
        """Test callbacks passed as ABCChemprop trainer kwargs are retained."""
        learning_rate_callback = pl.callbacks.LearningRateMonitor()
        expected_callbacks = [pl.callbacks.Callback(), learning_rate_callback]

        estimator = ABCChemprop(
            model=get_binary_classification_mpnn(),
            lightning_trainer__callbacks=expected_callbacks,
            lightning_trainer__accelerator="cpu",
            lightning_trainer__logger=True,
            lightning_trainer__log_every_n_steps=1,
            lightning_trainer__max_epochs=5,
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
        self.assertEqual(len(learning_rate_callback.lrs["lr-Adam"]), 5)

    def test_load_best_checkpoint_success(self) -> None:
        """Loading the best checkpoint restores the saved model state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=str(tmp_dir),
                save_top_k=1,
            )
            trainer = pl.Trainer(
                max_epochs=10,
                enable_checkpointing=True,
                callbacks=[checkpoint],
                accelerator="cpu",
            )
            estimator = ChempropClassifier(
                lightning_trainer=trainer,
            )
            estimator.fit(*self._get_training_data())
            best_checkpoint = torch.load(  # nosec B614
                checkpoint.best_model_path,
                map_location="cpu",
                weights_only=False,
            )
            model_state = estimator.model.state_dict()
            self.assertEqual(model_state.keys(), best_checkpoint["state_dict"].keys())
            for key, value in model_state.items():
                self.assertTrue(torch.equal(value, best_checkpoint["state_dict"][key]))

    def test_load_best_checkpoint_raises_without_callbacks(self) -> None:
        """No checkpoint callback raises a RuntimeError."""
        trainer = pl.Trainer(accelerator="cpu", callbacks=[])
        estimator = ChempropRegressor(
            lightning_trainer=trainer,
            reload_best_checkpoint=True,
        )

        with self.assertRaisesRegex(RuntimeError, "no checkpoint callback"):
            estimator._restore_best_checkpoint()
