"""Module for testing if the lightning wrapper functions work as intended."""

import unittest

import lightning as pl

from molpipeline.estimators.chemprop.lightning_wrapper import (
    get_non_default_params_trainer,
    get_params_trainer,
)


class TestLightningWrapper(unittest.TestCase):
    """Test the lightning wrapper functions.

    Note
    ----
    These tests are not exhaustive.
    """

    def test_setting_deterministic(self) -> None:
        """Test setting the deterministic parameter."""
        trainer_params = get_params_trainer(pl.Trainer(deterministic=True))
        self.assertTrue(trainer_params["deterministic"])

        trainer_params = get_params_trainer(pl.Trainer(deterministic=False))
        self.assertFalse(trainer_params["deterministic"])

        trainer_params = get_non_default_params_trainer(pl.Trainer(deterministic=True))
        self.assertIn("deterministic", trainer_params)
        self.assertTrue(trainer_params["deterministic"])

        trainer_params = get_non_default_params_trainer(pl.Trainer(deterministic=False))
        # deterministic is by default False and hence will not be listed in the parameters
        self.assertNotIn("deterministic", trainer_params)


if __name__ == "__main__":
    unittest.main()
