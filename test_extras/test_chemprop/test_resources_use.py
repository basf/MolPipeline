"""Test resources changes of estimators with chemprop models."""

import unittest
from typing import Literal

import torch
from sklearn.calibration import CalibratedClassifierCV

from molpipeline import Pipeline
from molpipeline.utils.resources_use import (
    model_to_cpu,
    set_single_job,
)
from test_extras.test_chemprop.chemprop_test_utils.default_models import (
    get_classification_pipeline,
    get_neural_fp_classification_pipeline,
)


def _get_chemprop_pipeline(
    n_jobs: int = 1,
    accelerator: Literal["gpu", "cpu"] = "cpu",
) -> Pipeline:
    """Get a Chemprop classification pipeline for testing resource management.

    Wraps :func:`get_classification_pipeline` while exposing ``n_jobs`` and
    ``accelerator`` parameters needed by the resource-use tests.

    Parameters
    ----------
    n_jobs : int, default 1
        Number of cores to use.
    accelerator : Literal["gpu", "cpu"], default "cpu"
        The accelerator to use.

    Returns
    -------
    Pipeline
        A pipeline based on the Chemprop model.

    """
    chemprop_n_jobs = 0 if n_jobs == 1 else n_jobs
    return get_classification_pipeline(
        chemprop_kwargs={
            "n_jobs": chemprop_n_jobs,
            "lightning_trainer__accelerator": accelerator,
        },
        pipeline_n_jobs=n_jobs,
    )


def _get_neural_rf_pipeline(
    n_jobs: int = 1,
    accelerator: Literal["gpu", "cpu"] = "cpu",
) -> Pipeline:
    """Get a Neural FP + RF pipeline for testing resource management.

    Wraps :func:`get_neural_fp_classification_pipeline` while exposing
    ``n_jobs`` and ``accelerator`` parameters needed by the resource-use tests.

    Parameters
    ----------
    n_jobs : int, default 1
        Number of cores to use.
    accelerator : Literal["gpu", "cpu"], default "cpu"
        The accelerator to use.

    Returns
    -------
    Pipeline
        A pipeline based on the Neural RF model.

    """
    chemprop_n_jobs = 0 if n_jobs == 1 else n_jobs
    return get_neural_fp_classification_pipeline(
        neural_fp_kwargs={
            "n_jobs": chemprop_n_jobs,
            "lightning_trainer__accelerator": accelerator,
        },
        rf_kwargs={
            "n_jobs": n_jobs,
            "n_estimators": 2,
        },
        pipeline_n_jobs=n_jobs,
    )


class TestSetSingleJobChemprop(unittest.TestCase):
    """Test the function set_single_job with chemprop models."""

    def test_set_single_job_chemprop(self) -> None:
        """Test if the Chemprop model is set to single-job mode."""
        for n_jobs, change_expected in [(-1, True), (1, False)]:
            with self.subTest(n_jobs=n_jobs, change_expected=change_expected):
                model = _get_chemprop_pipeline(n_jobs=n_jobs)
                changed = set_single_job(model)
                self.assertEqual(changed, change_expected)
                self.assertEqual(model.n_jobs, 1)
                self.assertEqual(model.named_steps["model"].n_jobs, 0)

    def test_set_single_job_neural_rf(self) -> None:
        """Test if the neural Random Forest model is set to single-job mode."""
        for n_jobs, change_expected in [(-1, True), (1, False)]:
            with self.subTest(n_jobs=n_jobs, change_expected=change_expected):
                model = _get_neural_rf_pipeline(n_jobs=n_jobs)
                changed = set_single_job(model)
                self.assertEqual(changed, change_expected)
                self.assertEqual(model.n_jobs, 1)
                self.assertEqual(model.named_steps["neural_fp"].n_jobs, 0)
                self.assertEqual(model.named_steps["rf"].n_jobs, 1)

    def test_set_single_job_calibrated_neural_rf(self) -> None:
        """Test if the calibrated neural RF model is set to single-job mode."""
        params = [
            (-1, 1, True),
            (1, -1, True),
            (-1, -1, True),
            (1, 1, False),
        ]
        for wrapper_n_jobs, estimator_n_jobs, change_expected in params:
            with self.subTest(
                wrapper_n_jobs=wrapper_n_jobs,
                estimator_n_jobs=estimator_n_jobs,
                change_expected=change_expected,
            ):
                model = CalibratedClassifierCV(
                    _get_neural_rf_pipeline(n_jobs=estimator_n_jobs),
                    n_jobs=wrapper_n_jobs,
                )
                changed = set_single_job(model)

                self.assertEqual(changed, change_expected)

                # test estimator attribute
                estimator = model.estimator
                self.assertIsNotNone(estimator)
                self.assertEqual(model.n_jobs, 1)
                self.assertEqual(estimator.n_jobs, 1)
                self.assertEqual(estimator.named_steps["neural_fp"].n_jobs, 0)
                self.assertEqual(estimator.named_steps["rf"].n_jobs, 1)

    def test_set_single_job_fitted_calibrated_neural_rf(self) -> None:
        """Test if the fitted calibrated neural RF model is set to single-job mode."""
        smiles_list = [
            "C",  # methane
            "CC",  # ethane
            "CCC",  # propane
            "CCCC",  # butane
            "CCCCC",  # pentane
            "CCCCCC",  # hexane
            "c1ccccc1",  # benzene
            "c1ccc(O)cc1",  # phenol
            "c1ccc(N)cc1",  # aniline
            "CC(=O)O",  # acetic acid
        ]
        labels = [int(i % 2 == 0) for i in range(len(smiles_list))]

        params = [
            (2, 1, True),
            (1, 2, True),
            (1, 1, False),
        ]
        for wrapper_n_jobs, estimator_n_jobs, change_expected in params:
            with self.subTest(
                wrapper_n_jobs=wrapper_n_jobs,
                estimator_n_jobs=estimator_n_jobs,
                change_expected=change_expected,
            ):
                model = CalibratedClassifierCV(
                    _get_neural_rf_pipeline(n_jobs=estimator_n_jobs),
                    n_jobs=wrapper_n_jobs,
                    cv=2,
                )
                model.fit(smiles_list, labels)

                changed = set_single_job(model)

                self.assertEqual(changed, change_expected)
                self.assertEqual(model.n_jobs, 1)

                # test estimator attribute
                estimator = model.estimator
                self.assertIsNotNone(estimator)
                self.assertEqual(estimator.n_jobs, 1)
                self.assertEqual(estimator.named_steps["neural_fp"].n_jobs, 0)
                self.assertEqual(estimator.named_steps["rf"].n_jobs, 1)

                # test calibrated_classifiers_ attribute (used in pred_proba)
                self.assertTrue(hasattr(model, "calibrated_classifiers_"))
                self.assertGreater(len(model.calibrated_classifiers_), 0)  # type: ignore[arg-type]
                for clf in model.calibrated_classifiers_:
                    pipe = clf.estimator  # type: ignore[union-attr]
                    self.assertEqual(pipe.n_jobs, 1)
                    self.assertEqual(pipe.named_steps["neural_fp"].n_jobs, 0)
                    for name, step in pipe.steps:  # iterate the pipeline
                        if name != "neural_fp" and hasattr(step, "n_jobs"):
                            self.assertEqual(step.n_jobs, 1)


class TestModelToCPUChemprop(unittest.TestCase):
    """Test the function model_to_cpu with chemprop models."""

    def test_model_to_cpu_chemprop(self) -> None:
        """Test if the Chemprop model is transferred to CPU correctly."""
        for accelerator, change_expected in [("cpu", False), ("gpu", True)]:
            with self.subTest(accelerator=accelerator, change_expected=change_expected):
                if accelerator == "gpu" and not torch.cuda.is_available():
                    continue
                model = _get_chemprop_pipeline(n_jobs=-1, accelerator=accelerator)  # type: ignore[arg-type]
                model_params = model.get_params(deep=True)
                self.assertEqual(
                    model_params["model__lightning_trainer__accelerator"],
                    accelerator,
                )
                changed = model_to_cpu(model)
                self.assertEqual(changed, change_expected)
                model_params = model.get_params(deep=True)
                self.assertEqual(
                    model_params["model__lightning_trainer__accelerator"],
                    "cpu",
                )

    def test_model_to_cpu_neural_rf(self) -> None:
        """Test if the neural RF model is transferred to CPU correctly."""
        for accelerator, change_expected in [("cpu", False), ("gpu", True)]:
            with self.subTest(accelerator=accelerator, change_expected=change_expected):
                if accelerator == "gpu" and not torch.cuda.is_available():
                    continue
                model = _get_neural_rf_pipeline(n_jobs=-1, accelerator=accelerator)  # type: ignore[arg-type]
                model_params = model.get_params(deep=True)

                self.assertEqual(
                    model_params["neural_fp__lightning_trainer__accelerator"],
                    accelerator,
                )
                changed = model_to_cpu(model)
                self.assertEqual(changed, change_expected)
                model_params = model.get_params(deep=True)
                self.assertEqual(
                    model_params["neural_fp__lightning_trainer__accelerator"],
                    "cpu",
                )

    def test_model_to_cpu_calibrated_neural_rf(self) -> None:
        """Test if the cal. neural RF model is transferred to CPU correctly."""
        for accelerator, change_expected in [("cpu", False), ("gpu", True)]:
            with self.subTest(accelerator=accelerator, change_expected=change_expected):
                if accelerator == "gpu" and not torch.cuda.is_available():
                    continue
                model = CalibratedClassifierCV(
                    _get_neural_rf_pipeline(n_jobs=-1, accelerator=accelerator),  # type: ignore[arg-type]
                )
                model_params = model.get_params(deep=True)
                self.assertEqual(
                    model_params[
                        "estimator__neural_fp__lightning_trainer__accelerator"
                    ],
                    accelerator,
                )
                changed = model_to_cpu(model)
                self.assertEqual(changed, change_expected)
                model_params = model.get_params(deep=True)
                self.assertEqual(
                    model_params[
                        "estimator__neural_fp__lightning_trainer__accelerator"
                    ],
                    "cpu",
                )


if __name__ == "__main__":
    unittest.main()
