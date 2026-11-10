"""Tests for the abstract class ABCChemprop."""

import abc
import tempfile
import unittest
from typing import Any, Generic, TypeVar, cast

import torch
from chemprop.data import MoleculeDataset
from lightning import pytorch as pl
from sklearn.base import clone
from typing_extensions import override

from molpipeline.any2mol import SmilesToMol
from molpipeline.estimators.chemprop.abstract import ABCChemprop
from molpipeline.estimators.chemprop.models import ChempropClassifier, ChempropRegressor
from molpipeline.mol2any import MolToChemprop
from test_extras.test_chemprop.chemprop_test_utils.default_models import (
    get_binary_classification_mpnn,
)

_CallbackVar = TypeVar("_CallbackVar", bound=pl.Callback)


class CallbackTestABC(abc.ABC, Generic[_CallbackVar]):
    """Callback-specific hooks for the clone-isolation test."""

    @abc.abstractmethod
    def get_callback(self, tmp_dir: str) -> _CallbackVar:
        """Create a fresh callback.

        Parameters
        ----------
        tmp_dir : str
            Temporary directory usable by the callback.

        Returns
        -------
        _CallbackVar
            New callback instance.

        """

    @staticmethod
    @abc.abstractmethod
    def get_callback_type() -> type[_CallbackVar]:
        """Return the callback class created by ``get_callback``.

        Returns
        -------
        type[_CallbackVar]
            Callback class.

        """

    def get_trainer_kwargs(self, tmp_dir: str) -> dict[str, Any]:  # ruff: ignore[unused-method-argument, no-self-use]
        """Return trainer kwargs required by the callback.

        Subclasses extend the returned dictionary via ``super()``.

        Parameters
        ----------
        tmp_dir : str
            Temporary directory usable by loggers.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for ``pl.Trainer``.

        """
        return {
            "accelerator": "cpu",
            "max_epochs": 2,
            "enable_model_summary": False,
            "enable_progress_bar": False,
        }

    def extract_callback_from_list(self, estimator: ABCChemprop) -> _CallbackVar:
        """Return the single callback of this test's type from the estimator's trainer.

        Parameters
        ----------
        estimator : ABCChemprop
            Estimator whose trainer callbacks are searched.

        Returns
        -------
        _CallbackVar
            The only callback of the requested type.

        Raises
        ------
        AssertionError
            If the trainer does not hold exactly one callback of that type.

        """
        callback_type = self.get_callback_type()
        matching_callbacks = [
            callback
            for callback in estimator.lightning_trainer.callbacks  # type: ignore[attr-defined]
            if isinstance(callback, callback_type)
        ]
        if len(matching_callbacks) != 1:
            msg = (
                f"Expected exactly one {callback_type.__name__}, "
                f"found {len(matching_callbacks)}."
            )
            raise AssertionError(msg)
        return matching_callbacks[0]

    @abc.abstractmethod
    def test_original_callback(
        self,
        test_object: unittest.TestCase,
        orig_callback: _CallbackVar,
    ) -> None:
        """Assert the original callback accumulated state during fitting.

        Parameters
        ----------
        test_object : unittest.TestCase
            Test case used for assertions.
        orig_callback : _CallbackVar
            Callback of the fitted estimator.

        """

    @abc.abstractmethod
    def test_cloned_callback(
        self,
        test_object: unittest.TestCase,
        cloned_callback: _CallbackVar,
    ) -> None:
        """Assert the cloned callback is still in its initial state.

        Parameters
        ----------
        test_object : unittest.TestCase
            Test case used for assertions.
        cloned_callback : _CallbackVar
            Callback of the unfitted clone.

        """


class LearningRateMonitorTest(CallbackTestABC[pl.callbacks.LearningRateMonitor]):
    """Clone-isolation hooks for LearningRateMonitor."""

    @override
    def get_callback(self, tmp_dir: str) -> pl.callbacks.LearningRateMonitor:
        """Create a LearningRateMonitor.

        Parameters
        ----------
        tmp_dir : str
            Unused.

        Returns
        -------
        pl.callbacks.LearningRateMonitor
            New callback instance.

        """
        return self.get_callback_type()()

    @staticmethod
    def get_callback_type() -> type[pl.callbacks.LearningRateMonitor]:
        """Return the LearningRateMonitor class.

        Returns
        -------
        type[pl.callbacks.LearningRateMonitor]
            Callback class.

        """
        return pl.callbacks.LearningRateMonitor

    @override
    def get_trainer_kwargs(self, tmp_dir: str) -> dict[str, Any]:
        """Return trainer kwargs enabling logging.

        Parameters
        ----------
        tmp_dir : str
            Directory for the CSV logger.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for ``pl.Trainer``.

        """
        trainer_kwargs = super().get_trainer_kwargs(tmp_dir)
        trainer_kwargs.update(
            {
                "logger": pl.loggers.CSVLogger(tmp_dir),
                "log_every_n_steps": 1,
                "enable_checkpointing": False,
            },
        )
        return trainer_kwargs

    @override
    def test_original_callback(
        self,
        test_object: unittest.TestCase,
        orig_callback: pl.callbacks.LearningRateMonitor,
    ) -> None:
        """Assert learning rates were logged for every epoch.

        Parameters
        ----------
        test_object : unittest.TestCase
            Test case used for assertions.
        orig_callback : pl.callbacks.LearningRateMonitor
            Callback of the fitted estimator.

        """
        test_object.assertEqual(len(orig_callback.lrs["lr-Adam"]), 2)

    @override
    def test_cloned_callback(
        self,
        test_object: unittest.TestCase,
        cloned_callback: pl.callbacks.LearningRateMonitor,
    ) -> None:
        """Assert no learning rates were logged.

        Parameters
        ----------
        test_object : unittest.TestCase
            Test case used for assertions.
        cloned_callback : pl.callbacks.LearningRateMonitor
            Callback of the unfitted clone.

        """
        test_object.assertFalse(cloned_callback.lrs)


class ModelCheckpointTest(CallbackTestABC[pl.callbacks.ModelCheckpoint]):
    """Clone-isolation hooks for ModelCheckpoint."""

    def get_callback(self, tmp_dir: str) -> pl.callbacks.ModelCheckpoint:
        """Create a ModelCheckpoint.

        Parameters
        ----------
        tmp_dir : str
            Directory to store checkpoints in.

        Returns
        -------
        pl.callbacks.ModelCheckpoint
            New callback instance.

        """
        return self.get_callback_type()(dirpath=tmp_dir, save_top_k=1)

    @staticmethod
    def get_callback_type() -> type[pl.callbacks.ModelCheckpoint]:
        """Return the ModelCheckpoint class.

        Returns
        -------
        type[pl.callbacks.ModelCheckpoint]
            Callback class.

        """
        return pl.callbacks.ModelCheckpoint

    def get_trainer_kwargs(self, tmp_dir: str) -> dict[str, Any]:
        """Return trainer kwargs enabling checkpointing.

        Parameters
        ----------
        tmp_dir : str
            Unused.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for ``pl.Trainer``.

        """
        trainer_kwargs = super().get_trainer_kwargs(tmp_dir)
        trainer_kwargs.update({"logger": False, "enable_checkpointing": True})
        return trainer_kwargs

    def test_original_callback(  # ruff: ignore[no-self-use]
        self,
        test_object: unittest.TestCase,
        orig_callback: pl.callbacks.ModelCheckpoint,
    ) -> None:
        """Assert a best checkpoint was saved.

        Parameters
        ----------
        test_object : unittest.TestCase
            Test case used for assertions.
        orig_callback : pl.callbacks.ModelCheckpoint
            Callback of the fitted estimator.

        """
        test_object.assertTrue(orig_callback.best_model_path)

    def test_cloned_callback(  # ruff: ignore[no-self-use]
        self,
        test_object: unittest.TestCase,
        cloned_callback: pl.callbacks.ModelCheckpoint,
    ) -> None:
        """Assert no checkpoint was saved.

        Parameters
        ----------
        test_object : unittest.TestCase
            Test case used for assertions.
        cloned_callback : pl.callbacks.ModelCheckpoint
            Callback of the unfitted clone.

        """
        test_object.assertFalse(cloned_callback.best_model_path)


_CALLBACK_TESTS: dict[str, CallbackTestABC[Any]] = {
    "LearningRateMonitor": LearningRateMonitorTest(),
    "ModelCheckpoint": ModelCheckpointTest(),
}


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

    def test_clone_callback_isolation(self) -> None:
        """Fitting an estimator does not change the callbacks of its clone."""
        for callback_name, callback_test in _CALLBACK_TESTS.items():
            with (
                self.subTest(callback=callback_name),
                tempfile.TemporaryDirectory() as tmp_dir,
            ):
                original_callback = callback_test.get_callback(tmp_dir)
                estimator = ChempropClassifier(
                    lightning_trainer=pl.Trainer(
                        callbacks=[original_callback],
                        **callback_test.get_trainer_kwargs(tmp_dir),
                    ),
                    n_jobs=0,
                )
                cloned_estimator = cast("ChempropClassifier", clone(estimator))

                # This should change the original callback's state.
                estimator.fit(*self._get_training_data())

                # The fitted estimator keeps the original callback object.
                fitted_callback = callback_test.extract_callback_from_list(estimator)
                self.assertIs(original_callback, fitted_callback)

                # The clone holds a different callback object.
                cloned_callback = callback_test.extract_callback_from_list(
                    cloned_estimator,
                )
                self.assertIsNot(original_callback, cloned_callback)

                # Only the original callback accumulated state.
                callback_test.test_original_callback(self, original_callback)
                callback_test.test_cloned_callback(self, cloned_callback)

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
