"""Tests for the abstract class ABCChemprop."""

import tempfile
import unittest

import torch
from chemprop.data import MoleculeDataset
from lightning import pytorch as pl
from sklearn.base import clone

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

    def test_clone_learning_rate_monitor_isolation(self) -> None:
        """Fitting an estimator does not change its clone's LearningRateMonitor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_monitor = pl.callbacks.LearningRateMonitor()
            estimator = ChempropClassifier(
                lightning_trainer=pl.Trainer(
                    callbacks=[original_monitor],
                    accelerator="cpu",
                    logger=pl.loggers.CSVLogger(tmp_dir),
                    max_epochs=2,
                    log_every_n_steps=1,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                ),
                n_jobs=0,
            )
            cloned_estimator = clone(estimator)

            # This should change the original LR monitor's state.
            estimator.fit(*self._get_training_data())

            # Check the original monitor for identity after fitting
            extracted_monitors = [
                callback
                for callback in estimator.lightning_trainer.callbacks  # type: ignore[attr-defined]
                if isinstance(callback, pl.callbacks.LearningRateMonitor)
            ]
            self.assertEqual(len(extracted_monitors), 1)
            extracted_monitor = extracted_monitors[0]
            self.assertIs(original_monitor, extracted_monitor)

            # Check if the cloned monitor is a different object.
            cloned_monitors = [
                callback
                for callback in cloned_estimator.lightning_trainer.callbacks  # type: ignore[attr-defined]
                if isinstance(callback, pl.callbacks.LearningRateMonitor)
            ]
            self.assertEqual(len(cloned_monitors), 1)
            cloned_monitor = cloned_monitors[0]
            self.assertIsNot(original_monitor, cloned_monitor)

            # Check if the monitors have different states.
            self.assertEqual(len(original_monitor.lrs["lr-Adam"]), 2)
            self.assertFalse(cloned_monitor.lrs)

    def test_clone_model_checkpoint_isolation(self) -> None:
        """Fitting an estimator does not change its clone's ModelCheckpoint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=tmp_dir,
                save_top_k=1,
            )
            estimator = ChempropClassifier(
                lightning_trainer=pl.Trainer(
                    callbacks=[original_checkpoint],
                    accelerator="cpu",
                    logger=False,
                    max_epochs=2,
                    enable_checkpointing=True,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                ),
                n_jobs=0,
            )
            cloned_estimator = clone(estimator)

            # This should change the original checkpoint's state.
            estimator.fit(*self._get_training_data())

            # Check the original checkpoint for identity after fitting
            extracted_checkpoints = [
                callback
                for callback in estimator.lightning_trainer.callbacks  # type: ignore[attr-defined]
                if isinstance(callback, pl.callbacks.ModelCheckpoint)
            ]
            self.assertEqual(len(extracted_checkpoints), 1)
            extracted_checkpoint = extracted_checkpoints[0]
            self.assertIs(original_checkpoint, extracted_checkpoint)

            # Check if the cloned checkpoint is a different object.
            cloned_checkpoints = [
                callback
                for callback in cloned_estimator.lightning_trainer.callbacks  # type: ignore[attr-defined]
                if isinstance(callback, pl.callbacks.ModelCheckpoint)
            ]
            self.assertEqual(len(cloned_checkpoints), 1)
            cloned_checkpoint = cloned_checkpoints[0]
            self.assertIsNot(original_checkpoint, cloned_checkpoint)

            # Check if the checkpoints have different states.
            self.assertTrue(original_checkpoint.best_model_path)
            self.assertFalse(cloned_checkpoint.best_model_path)

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
            estimator = ChempropClassifier(lightning_trainer=trainer)
            estimator.fit(*self._get_training_data())
            best_checkpoint = torch.load(  # nosec B614
                checkpoint.best_model_path,
                map_location=estimator.model.device,
                weights_only=False,
            )
            model_state = estimator.model.state_dict()
            self.assertEqual(model_state.keys(), best_checkpoint["state_dict"].keys())
            for key, value in model_state.items():
                self.assertTrue(torch.equal(value, best_checkpoint["state_dict"][key]))

    def test_load_best_checkpoint_raises_without_callbacks(self) -> None:
        """No checkpoint callback raises a RuntimeError."""
        trainer = pl.Trainer(
            accelerator="cpu",
            callbacks=[],
            enable_checkpointing=False,  # Adds a default ModelCheckpoint otherwise
        )
        estimator = ChempropRegressor(
            lightning_trainer=trainer,
            reload_best_checkpoint=True,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Best checkpoint was requested, but no checkpoint callback was found\.",
        ):
            estimator._restore_best_checkpoint()

    def test_load_best_checkpoint_raises_without_saved_checkpoint(self) -> None:
        """A checkpoint callback without a saved checkpoint raises a RuntimeError."""
        trainer = pl.Trainer(accelerator="cpu", enable_checkpointing=True)
        estimator = ChempropRegressor(
            lightning_trainer=trainer,
            reload_best_checkpoint=True,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Best checkpoint was requested, but no checkpoint callback has a best "
            r"model path\.",
        ):
            estimator._restore_best_checkpoint()

    def test_load_best_checkpoint_auto_without_best_path(self) -> None:
        """Automatic reloading is skipped when only the last checkpoint is saved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=tmp_dir,
                save_top_k=0,
                save_last=True,
            )
            trainer = pl.Trainer(
                max_epochs=1,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=True,
                callbacks=[checkpoint],
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            estimator = ChempropClassifier(lightning_trainer=trainer, n_jobs=0)
            estimator.fit(*self._get_training_data())
            self.assertFalse(checkpoint.best_model_path)

    def test_reload_best_checkpoint_survives_clone(self) -> None:
        """The reload_best_checkpoint option is exposed as a parameter and cloned."""
        for estimator in (
            ChempropClassifier(reload_best_checkpoint=False),
            ChempropRegressor(reload_best_checkpoint=False),
        ):
            with self.subTest(estimator=type(estimator).__name__):
                self.assertIn("reload_best_checkpoint", estimator.get_params())
                self.assertIs(clone(estimator).reload_best_checkpoint, False)
