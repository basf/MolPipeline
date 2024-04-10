"""Test module for behavior of chemprop in a pipeline."""

import unittest

import numpy as np
from sklearn.base import clone

from molpipeline.any2mol import SmilesToMol
from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    SumAggregation,
)
from molpipeline.estimators.chemprop.models import ChempropModel
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
