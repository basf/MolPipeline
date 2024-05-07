"""Test module for behavior of chemprop in a pipeline."""

import unittest
from io import BytesIO
from typing import TypeVar

import joblib
import numpy as np
import pandas as pd
from chemprop.nn.loss import LossFunction
from lightning import pytorch as pl
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.profilers.base import PassThroughProfiler
from sklearn.base import clone
from torch import nn

from molpipeline.any2mol import SmilesToMol
from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    SumAggregation,
)
from molpipeline.estimators.chemprop.models import (
    ChempropClassifier,
    ChempropModel,
    ChempropRegressor,
)
from molpipeline.mol2any.mol2chemprop import MolToChemprop
from molpipeline.pipeline import Pipeline
from molpipeline.post_prediction import PostPredictionWrapper


# pylint: disable=duplicate-code
def get_model() -> ChempropModel:
    """Get the Chemprop model.

    Returns
    -------
    ChempropModel
        The Chemprop model.
    """
    binary_clf_ffn = BinaryClassificationFFN()
    aggregate = SumAggregation()
    bond_message_passing = BondMessagePassing()
    mpnn = MPNN(
        message_passing=bond_message_passing,
        agg=aggregate,
        predictor=binary_clf_ffn,
    )
    chemprop_model = ChempropModel(model=mpnn)
    return chemprop_model


def get_model_pipeline() -> Pipeline:
    """Get the Chemprop model pipeline.

    Returns
    -------
    Pipeline
        The Chemprop model pipeline.
    """
    chemprop_model = get_model()
    mol2chemprop = MolToChemprop()
    smiles2mol = SmilesToMol()
    model_pipeline = Pipeline(
        steps=[
            ("smiles2mol", smiles2mol),
            ("mol2chemprop", mol2chemprop),
            ("model", chemprop_model),
        ],
    )
    return model_pipeline


DEFAULT_TRAINER = pl.Trainer(
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    max_epochs=5,
    enable_model_summary=False,
    enable_progress_bar=False,
    val_check_interval=0.0,
)


def get_regression_pipeline() -> Pipeline:
    """Get the Chemprop model pipeline for regression.

    Returns
    -------
    Pipeline
        The Chemprop model pipeline for regression.
    """

    smiles2mol = SmilesToMol()
    mol2chemprop = MolToChemprop()
    error_filter = ErrorFilter(filter_everything=True)
    filter_reinserter = FilterReinserter.from_error_filter(
        error_filter, fill_value=np.nan
    )
    chemprop_model = ChempropRegressor(lightning_trainer=DEFAULT_TRAINER)
    model_pipeline = Pipeline(
        steps=[
            ("smiles2mol", smiles2mol),
            ("mol2chemprop", mol2chemprop),
            ("error_filter", error_filter),
            ("model", chemprop_model),
            ("filter_reinserter", PostPredictionWrapper(filter_reinserter)),
        ],
    )
    return model_pipeline


def get_classification_pipeline() -> Pipeline:
    """Get the Chemprop model pipeline for classification.

    Returns
    -------
    Pipeline
        The Chemprop model pipeline for classification.
    """
    smiles2mol = SmilesToMol()
    mol2chemprop = MolToChemprop()
    error_filter = ErrorFilter(filter_everything=True)
    filter_reinserter = FilterReinserter.from_error_filter(
        error_filter, fill_value=np.nan
    )
    chemprop_model = ChempropClassifier(lightning_trainer=DEFAULT_TRAINER)
    model_pipeline = Pipeline(
        steps=[
            ("smiles2mol", smiles2mol),
            ("mol2chemprop", mol2chemprop),
            ("error_filter", error_filter),
            ("model", chemprop_model),
            ("filter_reinserter", PostPredictionWrapper(filter_reinserter)),
        ],
    )
    return model_pipeline


_T = TypeVar("_T")


def joblib_dump_load(obj: _T) -> _T:
    """Dump and load an object using joblib.

    Notes
    -----
    The object is not dumped to disk but to a BytesIO object.

    Parameters
    ----------
    obj : _T
        The object to dump and load.

    Returns
    -------
    _T
        The loaded object.
    """
    bytes_container = BytesIO()
    joblib.dump(obj, bytes_container)
    bytes_container.seek(0)  # update to enable reading
    bytes_model = bytes_container.read()

    return joblib.load(BytesIO(bytes_model))


class TestChempropPipeline(unittest.TestCase):
    """Test the Chemprop model pipeline."""

    def test_get_set_params(self) -> None:
        """Test the get_params and set_params methods of the pipeline."""
        model = get_model_pipeline()
        orig_params = model.get_params(deep=True)
        new_params = {
            "model__model__predictor__activation": "relu",
            "model__model__predictor__dropout": 0.5,
            "model__model__message_passing__activation": "relu",
        }
        # Check setting new parameters
        model.set_params(**new_params)
        model_params = model.get_params(deep=True)
        for param_name, param in new_params.items():
            self.assertEqual(param, model_params[param_name])
        model.set_params(**orig_params)
        model_params = model.get_params(deep=True)
        for param_name, param in orig_params.items():
            self.assertEqual(param, model_params[param_name])

    def test_clone(self) -> None:
        """Test the clone method of the pipeline."""
        model = get_model_pipeline()
        cloned_model = clone(model)
        for step_name, step in model.steps:
            cloned_step = cloned_model.named_steps[step_name]
            self.assertEqual(step.__class__, cloned_step.__class__)
            params = step.get_params(deep=True)  # type: ignore
            cloned_params = cloned_step.get_params(deep=True)
            for param_name, param in params.items():
                # If parm implements get_params, it was cloned as well and we need
                # to compare the parameters. Since all parameters are listed flat in
                # the params dicts, all objects are identical if param dicts are identical.
                if hasattr(param, "get_params"):
                    self.assertEqual(
                        param.__class__, cloned_params[param_name].__class__
                    )
                elif param_name == "lightning_trainer":
                    # Lightning trainer does not implement get_params so things are a bit tricky
                    # at the moment. We can only check if the classes are the same.
                    self.assertEqual(
                        param.__class__, cloned_params[param_name].__class__
                    )
                elif isinstance(param, LossFunction):
                    self.assertEqual(
                        param.state_dict()["task_weights"],
                        cloned_params[param_name].state_dict()["task_weights"],
                    )
                    self.assertEqual(type(param), type(cloned_params[param_name]))
                elif isinstance(param, (nn.Identity, Accelerator, PassThroughProfiler)):
                    self.assertEqual(type(param), type(cloned_params[param_name]))
                elif param_name == "lightning_trainer__callbacks":
                    self.assertIsInstance(cloned_params[param_name], list)
                    self.assertEqual(len(param), len(cloned_params[param_name]))
                    for callback, cloned_callback in zip(
                        param, cloned_params[param_name]
                    ):
                        self.assertEqual(type(callback), type(cloned_callback))
                else:
                    self.assertEqual(
                        param, cloned_params[param_name], f"Failed for {param_name}"
                    )

    def test_passing_smiles(self) -> None:
        """Test passing SMILES strings to the pipeline.

        Since the pipeline is not trained, we only check if the prediction is successful.
        """
        model = get_model_pipeline()
        smiles = ["CCO", "CNC", "CCN"]
        pred = model.predict(smiles)
        self.assertEqual(len(pred), 3)
        proba = model.predict_proba(smiles)
        self.assertEqual(len(proba), 3)

    def test_error_handling(self) -> None:
        """Test the error handling in the pipeline."""
        error_filter = ErrorFilter(filter_everything=True)
        filter_reinserter = FilterReinserter.from_error_filter(
            error_filter, fill_value=np.nan
        )
        pipeline = Pipeline(
            steps=[
                ("smiles2mol", SmilesToMol()),
                ("mol2chemprop", MolToChemprop()),
                ("error_filter", error_filter),
                ("model", get_model()),
                ("filter_reinserter", PostPredictionWrapper(filter_reinserter)),
            ]
        )
        smiles = ["CCO", "CNC", "CCN", "invalid"]
        pred = pipeline.predict(smiles)
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.isnan(pred[-1]))
        proba = pipeline.predict_proba(smiles)
        self.assertEqual(len(proba), 4)
        self.assertTrue(np.isnan(proba[-1]).all())


class TestRegressionPipeline(unittest.TestCase):
    """Test the Chemprop model pipeline for regression."""

    def test_prediction(self) -> None:
        """Test the prediction of the regression model."""

        molecule_net_logd_df = pd.read_csv(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
        ).head(1000)
        regression_model = get_regression_pipeline()
        regression_model.fit(
            molecule_net_logd_df["smiles"].tolist(),
            molecule_net_logd_df["exp"].to_numpy(),
        )
        pred = regression_model.predict(molecule_net_logd_df["smiles"].tolist())

        self.assertEqual(len(pred), len(molecule_net_logd_df))

        model_copy = joblib_dump_load(regression_model)
        pred_copy = model_copy.predict(molecule_net_logd_df["smiles"].tolist())
        self.assertTrue(np.allclose(pred, pred_copy))


class TestClassificationPipeline(unittest.TestCase):
    """Test the Chemprop model pipeline for classification."""

    def test_prediction(self) -> None:
        """Test the prediction of the classification model."""

        molecule_net_bbbp_df = pd.read_csv(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        ).head(1000)
        classification_model = get_classification_pipeline()
        classification_model.fit(
            molecule_net_bbbp_df["smiles"].tolist(),
            molecule_net_bbbp_df["p_np"].to_numpy(),
        )
        pred = classification_model.predict(molecule_net_bbbp_df["smiles"].tolist())
        proba = classification_model.predict_proba(
            molecule_net_bbbp_df["smiles"].tolist()
        )
        self.assertEqual(len(pred), len(molecule_net_bbbp_df))
        self.assertEqual(proba.shape[1], 2)
        self.assertEqual(proba.shape[0], len(molecule_net_bbbp_df))

        model_copy = joblib_dump_load(classification_model)
        pred_copy = model_copy.predict(molecule_net_bbbp_df["smiles"].tolist())
        proba_copy = model_copy.predict_proba(molecule_net_bbbp_df["smiles"].tolist())

        nan_indices = np.isnan(pred)
        self.assertListEqual(nan_indices.tolist(), np.isnan(pred_copy).tolist())
        self.assertTrue(np.allclose(pred[~nan_indices], pred_copy[~nan_indices]))

        self.assertEqual(proba.shape, proba_copy.shape)
        self.assertTrue(np.allclose(proba[~nan_indices], proba_copy[~nan_indices]))
