"""Tests for the abstract class ABCChemprop."""

import unittest

from molpipeline.estimators.chemprop.abstract import ABCChemprop


class TestABCChemprop(unittest.TestCase):
    """Test static methods of the Chemprop model."""

    def test_filter_params_callback(self) -> None:
        """Test the filter_params_callback method."""
        dummy_params = {
            "callback_modelckpt__monitor": "val_loss",
            "other__param": "value",
        }
        # pylint: disable=protected-access
        other_params, callback_params = ABCChemprop._filter_params(
            dummy_params, "callback_modelckpt"
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
        # pylint: disable=protected-access
        other_params, trainer_params = ABCChemprop._filter_params(
            dummy_params, "lightning_trainer"
        )
        # pylint: enable=protected-access
        self.assertEqual(trainer_params, {"max_epochs": 50})
        self.assertEqual(other_params, {"other__param": "value"})
