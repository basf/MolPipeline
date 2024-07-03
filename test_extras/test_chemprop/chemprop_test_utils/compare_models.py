"""Functions for comparing chemprop models."""

from typing import Sequence
from unittest import TestCase

import torch
from chemprop.nn.loss import LossFunction
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.profilers.base import PassThroughProfiler
from sklearn.base import BaseEstimator
from torch import nn


def compare_params(
    test_case: TestCase, model_a: BaseEstimator, model_b: BaseEstimator
) -> None:
    """Compare the parameters of two models.

    Parameters
    ----------
    test_case : TestCase
        The test case for which to raise the assertion.
    model_a : BaseEstimator
        The first model.
    model_b : BaseEstimator
        The second model.
    """
    model_a_params = model_a.get_params(deep=True)
    model_b_params = model_b.get_params(deep=True)
    test_case.assertSetEqual(set(model_a_params.keys()), set(model_b_params.keys()))
    for param_name, param_a in model_a_params.items():
        param_b = model_b_params[param_name]
        test_case.assertEqual(param_a.__class__, param_b.__class__)
        if hasattr(param_a, "get_params"):
            test_case.assertTrue(hasattr(param_b, "get_params"))
            test_case.assertNotEqual(id(param_a), id(param_b))
        elif isinstance(param_a, LossFunction):
            test_case.assertEqual(
                param_a.state_dict()["task_weights"],
                param_b.state_dict()["task_weights"],
            )
            test_case.assertEqual(type(param_a), type(param_b))
        elif isinstance(param_a, (nn.Identity, Accelerator, PassThroughProfiler)):
            test_case.assertEqual(type(param_a), type(param_b))
        elif isinstance(param_a, torch.Tensor):
            test_case.assertTrue(
                torch.equal(param_a, param_b), f"Test failed for {param_name}"
            )
        elif param_name == "lightning_trainer__callbacks":
            test_case.assertIsInstance(param_b, Sequence)
            for i, callback in enumerate(param_a):
                test_case.assertIsInstance(callback, type(param_b[i]))
        else:
            test_case.assertEqual(param_a, param_b, f"Test failed for {param_name}")
