"""Tests for the abstract class ABCChemprop."""

import tempfile
import unittest

import torch
from lightning import pytorch as pl

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.estimators.chemprop.abstract import ABCChemprop
from molpipeline.estimators.chemprop.models import ChempropClassifier, ChempropRegressor
from molpipeline.mol2any import MolToChemprop


class TestABCChemprop(unittest.TestCase):
    """Test static methods of the Chemprop model."""

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
            pipeline = Pipeline(
                steps=[
                    ("auto2mol", AutoToMol()),
                    ("mol2chemprop", MolToChemprop()),
                    ("chemprop", estimator),
                ],
            )
            pipeline.fit(
                X=["CCO", "CCN", "CCC"],
                y=[0, 1, 0],
            )
            best_checkpoint = torch.load(
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
        estimator = ChempropRegressor(lightning_trainer__accelerator="cpu")

        with self.assertRaisesRegex(RuntimeError, "no checkpoint callback"):
            estimator._load_best_checkpoint([])
