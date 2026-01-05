"""Test module for behavior of chemprop in a pipeline."""

import unittest
from io import BytesIO
from typing import TypeVar
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from molpipeline.any2mol import SmilesToMol
from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.estimators.chemprop.abstract import ABCChemprop
from molpipeline.estimators.chemprop.component_wrapper import BondMessagePassing
from molpipeline.estimators.chemprop.metric_wrapper import BCELoss
from molpipeline.mol2any.mol2chemprop import MolToChemprop
from molpipeline.pipeline import Pipeline
from molpipeline.post_prediction import PostPredictionWrapper
from molpipeline.utils.file_loading.url_file_loading import URLFileLoader
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json
from test_extras.test_chemprop.chemprop_test_utils.compare_models import compare_params
from test_extras.test_chemprop.chemprop_test_utils.default_models import (
    get_classification_pipeline,
    get_model,
    get_model_pipeline,
    get_multiclass_classification_pipeline,
    get_regression_pipeline,
    get_smiles_checker_pipeline,
)
from test_extras.test_chemprop.chemprop_test_utils.randomization import (
    randomize_state_dict_weights,
)
from tests import TEST_DATA_DIR

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
            if isinstance(step, ABCChemprop):
                compare_params(self, step, cloned_step)
                continue
            for param_name, param in params.items():
                # If parm implements get_params, it was cloned as well and we need to
                # compare the parameters. Since all parameters are listed flat in the
                # params dicts, all objects are identical if param dicts are identical.
                if hasattr(param, "get_params"):
                    self.assertEqual(
                        param.__class__,
                        cloned_params[param_name].__class__,
                    )
                else:
                    self.assertEqual(
                        param,
                        cloned_params[param_name],
                        f"Failed for {param_name}",
                    )

    def test_passing_smiles(self) -> None:
        """Test passing SMILES strings to the pipeline.

        Since the pipeline is not trained, we only check if the prediction is
        successful.

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
            error_filter,
            fill_value=np.nan,
        )
        pipeline = Pipeline(
            steps=[
                ("smiles2mol", SmilesToMol()),
                ("mol2chemprop", MolToChemprop()),
                ("error_filter", error_filter),
                ("model", get_model()),
                ("filter_reinserter", PostPredictionWrapper(filter_reinserter)),
            ],
        )
        smiles = ["CCO", "CNC", "CCN", "invalid"]
        pred = pipeline.predict(smiles)
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.isnan(pred[-1]))
        proba = pipeline.predict_proba(smiles)
        self.assertEqual(len(proba), 4)
        self.assertTrue(np.isnan(proba[-1]).all())

    def test_state_dict_forwarding(self) -> None:
        """Test that the state_dict is properly forwarded to the model."""
        chemprop_classifier = get_classification_pipeline()
        chemprop_classifier.fit(["CCO", "CCN"], [0, 1])
        chemprop_model = chemprop_classifier.named_steps["model"]
        state_dict = chemprop_model.model.state_dict()

        new_chemprop_classifier = get_classification_pipeline()
        new_chemprop_classifier.set_params(
            model__model__state_dict_ref=state_dict,
        )

        orig_pred = chemprop_classifier.predict_proba(["CCO", "CCN"])
        new_pred = new_chemprop_classifier.predict_proba(["CCO", "CCN"])

        self.assertTrue(np.allclose(orig_pred, new_pred))

    def test_state_dict_from_url(self) -> None:
        """Test that the state_dict can be loaded from a URL."""
        chemeleon_url = "dummy_chemeleon_url.pt"
        chemeleon_dummy = BondMessagePassing(d_h=2048)
        dummy_state_dict = randomize_state_dict_weights(chemeleon_dummy.state_dict())

        with mock.patch(
            "molpipeline.utils.file_loading.url_file_loading.requests.get",
        ) as mock_get:
            mock_response = mock.Mock()
            buffer = BytesIO()
            torch.save(dummy_state_dict, buffer)
            buffer.seek(0)
            mock_response.content = buffer.read()
            mock_get.return_value = mock_response
            chemprop_classifier = get_classification_pipeline(
                chemprop_kwargs={
                    "model__message_passing__state_dict_ref": URLFileLoader(
                        chemeleon_url,
                    ),
                    "model__message_passing__d_h": 2048,
                    "model__predictor__input_dim": 2048,
                },
            )
            self.assertEqual(mock_get.call_count, 1)

        # Check that the state dict was loaded correctly
        chemprop_mpnn = chemprop_classifier.named_steps["model"].model
        loaded_state_dict = chemprop_mpnn.message_passing.state_dict()
        self.assertEqual(
            dummy_state_dict.keys(),
            loaded_state_dict.keys(),
        )
        for key, value in dummy_state_dict.items():
            self.assertTrue(torch.equal(value, loaded_state_dict[key]))

        # Check that the prediction works
        pred = chemprop_classifier.predict_proba(["CCO", "CCN"])
        self.assertEqual(len(pred), 2)

    def test_state_dict_from_url_serialization(self) -> None:
        """Test that the state_dict set via URLFileLoader can be serialized."""
        chemeleon_url = "https://dummy_chemeleon_url.pt"
        chemeleon_dummy = BondMessagePassing(d_h=2048)
        dummy_state_dict = randomize_state_dict_weights(chemeleon_dummy.state_dict())

        with mock.patch(
            "molpipeline.utils.file_loading.url_file_loading.requests.get",
        ) as mock_get:
            mock_response = mock.Mock()
            buffer = BytesIO()
            torch.save(dummy_state_dict, buffer)
            buffer.seek(0)
            mock_response.content = buffer.read()
            mock_get.return_value = mock_response
            chemprop_classifier = get_classification_pipeline(
                chemprop_kwargs={
                    "model__message_passing__state_dict_ref": URLFileLoader(
                        chemeleon_url,
                    ),
                    "model__message_passing__d_h": 2048,
                    "model__predictor__input_dim": 2048,
                },
            )
            self.assertEqual(mock_get.call_count, 1)

            # Serialize and deserialize the pipeline
            chemprop_classifier_json = recursive_to_json(chemprop_classifier)
            chemprop_classifier_loaded = recursive_from_json(chemprop_classifier_json)
            self.assertEqual(mock_get.call_count, 2)
        # Check that the state dict was loaded correctly
        chemprop_mpnn = chemprop_classifier_loaded.named_steps["model"].model
        loaded_state_dict = chemprop_mpnn.message_passing.state_dict()
        self.assertEqual(
            dummy_state_dict.keys(),
            loaded_state_dict.keys(),
        )
        for key, value in dummy_state_dict.items():
            self.assertTrue(torch.equal(value, loaded_state_dict[key]))

    def test_sample_weight_forwarding(self) -> None:
        """Test that the sample weights are properly forwarded to the loss function."""
        smiles = ["CCO", "CCN"] * 32
        y = [0, 1] * 32
        sample_weight = np.ones(64) * 0.5

        bce_loss = BCELoss()
        bce_loss.update = mock.MagicMock(side_effect=bce_loss.update)
        model = get_classification_pipeline(
            chemprop_kwargs={"model__predictor__criterion": bce_loss},
        )
        model.fit(smiles, y, model__sample_weight=sample_weight)
        for call_params in bce_loss.update.call_args_list:
            call_args = call_params[0]
            weight = call_args[3]
            self.assertTrue(np.allclose(weight, sample_weight))

    def test_class_weight_no_sample_weight(self) -> None:
        """Test that class_weight is used correctly."""
        # Use a small dummy dataset with 80% of class 0 and 20% of class 1
        smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1"] * 4
        y = np.array([0, 0, 0, 0, 1] * 4)
        # Test with class_weight='balanced'
        bce_loss = BCELoss()
        bce_loss.update = mock.MagicMock(side_effect=bce_loss.update)
        model = get_classification_pipeline(
            chemprop_kwargs={
                "class_weight": "balanced",
                "model__predictor__criterion": bce_loss,
            },
        )
        model.fit(smiles, y)
        for call_params in bce_loss.update.call_args_list:
            call_args = call_params[0]
            label = call_args[1].detach().cpu().numpy()
            weight = call_args[3].detach().cpu().numpy()
            # Check that weights correspond to class_weight='balanced'
            neg_label_weight_sum = np.sum(weight[label == 0])
            pos_label_weight_sum = np.sum(weight[label == 1])
            self.assertAlmostEqual(neg_label_weight_sum, pos_label_weight_sum)

    def test_class_weight_with_sample_weight(self) -> None:
        """Test that class_weight and sample_weight are combined correctly."""
        # Use a small dummy dataset with 80% of class 0 and 20% of class 1
        smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1"] * 4
        y = np.array([0, 0, 0, 0, 1] * 4)
        # The last component has double weight
        sample_weight = np.array([1.0, 1.0, 1.0, 1.0, 2.0] * 4)
        # Test with class_weight='balanced'
        bce_loss = BCELoss()
        bce_loss.update = mock.MagicMock(side_effect=bce_loss.update)
        model = get_classification_pipeline(
            chemprop_kwargs={
                "class_weight": "balanced",
                "model__predictor__criterion": bce_loss,
            },
        )
        model.fit(smiles, y, model__sample_weight=sample_weight)
        for call_params in bce_loss.update.call_args_list:
            call_args = call_params[0]
            label = call_args[1].detach().cpu().numpy()
            weight = call_args[3].detach().cpu().numpy()
            # Compute expected weights
            neg_label_weight_sum = np.sum(weight[label == 0])
            pos_label_weight_sum = np.sum(weight[label == 1])
            self.assertAlmostEqual(neg_label_weight_sum * 2, pos_label_weight_sum)


class TestRegressionPipeline(unittest.TestCase):
    """Test the Chemprop model pipeline for regression."""

    def test_prediction(self) -> None:
        """Test the prediction of the regression model."""
        molecule_net_logd_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_logd.tsv.gz",
            sep="\t",
            nrows=100,
        )
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

        # Test single prediction, this was causing an error before
        single_mol_pred = regression_model.predict(
            [molecule_net_logd_df["smiles"].iloc[0]],
        )
        self.assertEqual(single_mol_pred.shape, (1,))


class TestMultiRegressionPipeline(unittest.TestCase):
    """Test the Chemprop model pipeline for multi-target/multiple regression."""

    def test_prediction(self) -> None:
        """Test the prediction of the multiple regression model."""
        molecule_net_logd_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_logd.tsv.gz",
            sep="\t",
            nrows=100,
        )

        # add another target col with gaussian noise
        rng = np.random.default_rng(seed=123)
        molecule_net_logd_df["exp2"] = molecule_net_logd_df["exp"] + rng.normal(
            loc=0.0,
            scale=1.0,
            size=molecule_net_logd_df.shape[0],
        )
        target_cols = ["exp", "exp2"]

        regression_model = get_regression_pipeline(n_tasks=len(target_cols))
        regression_model.fit(
            molecule_net_logd_df["smiles"],
            molecule_net_logd_df[target_cols].to_numpy(),
        )
        pred = regression_model.predict(molecule_net_logd_df["smiles"])
        self.assertEqual(
            pred.shape,
            (
                len(molecule_net_logd_df),
                len(target_cols),
            ),
        )

        model_copy = joblib_dump_load(regression_model)
        pred_copy = model_copy.predict(molecule_net_logd_df["smiles"])
        self.assertTrue(np.allclose(pred, pred_copy))

        # Test single prediction, this was causing an error before
        single_mol_pred = regression_model.predict(
            [molecule_net_logd_df["smiles"].iloc[0]],
        )
        self.assertEqual(single_mol_pred.shape, (1, len(target_cols)))


class TestClassificationPipeline(unittest.TestCase):
    """Test the Chemprop model pipeline for classification."""

    def setUp(self) -> None:
        """Set up repeated variables."""
        molecule_net_bbbp_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_bbbp.tsv.gz",
            sep="\t",
            nrows=100,
        )
        smiles_pipeline = get_smiles_checker_pipeline()
        molecule_net_bbbp_df["smiles"] = smiles_pipeline.transform(
            molecule_net_bbbp_df["smiles"],
        )
        molecule_net_bbbp_df = molecule_net_bbbp_df.dropna(subset=["smiles", "p_np"])

        self.molecule_net_bbbp_df = molecule_net_bbbp_df

    def test_prediction(self) -> None:
        """Test the prediction of the classification model."""
        classification_model = get_classification_pipeline()
        classification_model.fit(
            self.molecule_net_bbbp_df["smiles"].tolist(),
            self.molecule_net_bbbp_df["p_np"].to_numpy(),
        )
        pred = classification_model.predict(
            self.molecule_net_bbbp_df["smiles"].tolist(),
        )
        proba = classification_model.predict_proba(
            self.molecule_net_bbbp_df["smiles"].tolist(),
        )
        self.assertEqual(len(pred), len(self.molecule_net_bbbp_df))
        self.assertEqual(proba.shape[1], 2)
        self.assertEqual(proba.shape[0], len(self.molecule_net_bbbp_df))

        model_copy = joblib_dump_load(classification_model)
        pred_copy = model_copy.predict(self.molecule_net_bbbp_df["smiles"].tolist())
        proba_copy = model_copy.predict_proba(
            self.molecule_net_bbbp_df["smiles"].tolist(),
        )

        nan_indices = np.isnan(pred)
        self.assertListEqual(nan_indices.tolist(), np.isnan(pred_copy).tolist())
        self.assertTrue(np.allclose(pred[~nan_indices], pred_copy[~nan_indices]))

        self.assertEqual(proba.shape, proba_copy.shape)
        self.assertTrue(np.allclose(proba[~nan_indices], proba_copy[~nan_indices]))

        # Test single prediction, this was causing an error before
        single_mol_pred = classification_model.predict(
            [self.molecule_net_bbbp_df["smiles"].iloc[0]],
        )
        self.assertEqual(single_mol_pred.shape, (1,))
        single_mol_proba = classification_model.predict_proba(
            [self.molecule_net_bbbp_df["smiles"].iloc[0]],
        )
        self.assertEqual(single_mol_proba.shape, (1, 2))

    def test_calibrated_classifier(self) -> None:
        """Test if the pipeline can be used with a CalibratedClassifierCV."""
        calibrated_pipeline = CalibratedClassifierCV(
            get_classification_pipeline(),
            cv=2,
            ensemble=True,
            method="isotonic",
        )
        calibrated_pipeline.fit(
            self.molecule_net_bbbp_df["smiles"].tolist(),
            self.molecule_net_bbbp_df["p_np"].to_numpy(),
        )
        predicted_value_array = calibrated_pipeline.predict(
            self.molecule_net_bbbp_df["smiles"].tolist(),
        )
        predicted_proba_array = calibrated_pipeline.predict_proba(
            self.molecule_net_bbbp_df["smiles"].tolist(),
        )
        self.assertIsInstance(predicted_value_array, np.ndarray)
        self.assertIsInstance(predicted_proba_array, np.ndarray)
        self.assertEqual(
            predicted_value_array.shape,
            (len(self.molecule_net_bbbp_df["smiles"].tolist()),),
        )
        self.assertEqual(
            predicted_proba_array.shape,
            (len(self.molecule_net_bbbp_df["smiles"].tolist()), 2),
        )


class TestMulticlassClassificationPipeline(unittest.TestCase):
    """Test the Chemprop model pipeline for multiclass classification."""

    def test_prediction(self) -> None:
        """Test the prediction of the multiclass classification model."""
        test_data_df = pd.read_csv(
            TEST_DATA_DIR / "multiclass_mock.tsv",
            sep="\t",
            index_col=False,
        )
        classification_model = get_multiclass_classification_pipeline(n_classes=3)
        mols = test_data_df["Molecule"].tolist()
        classification_model.fit(
            mols,
            test_data_df["Label"].to_numpy(),
        )
        pred = classification_model.predict(mols)
        proba = classification_model.predict_proba(mols)
        self.assertEqual(len(pred), len(test_data_df))
        self.assertEqual(proba.shape[1], 3)
        self.assertEqual(proba.shape[0], len(test_data_df))

        model_copy = joblib_dump_load(classification_model)
        pred_copy = model_copy.predict(mols)
        proba_copy = model_copy.predict_proba(mols)

        nan_mask = np.isnan(pred)
        self.assertListEqual(nan_mask.tolist(), np.isnan(pred_copy).tolist())
        self.assertTrue(np.allclose(pred[~nan_mask], pred_copy[~nan_mask]))

        self.assertEqual(proba.shape, proba_copy.shape)
        self.assertEqual(pred.shape, pred_copy.shape)
        self.assertTrue(np.allclose(proba[~nan_mask], proba_copy[~nan_mask]))

        # Test single prediction, this was causing an error before
        single_mol_pred = classification_model.predict(
            [test_data_df["Molecule"].iloc[0]],
        )
        self.assertEqual(single_mol_pred.shape, (1,))
        single_mol_proba = classification_model.predict_proba(
            [test_data_df["Molecule"].iloc[0]],
        )
        self.assertEqual(single_mol_proba.shape, (1, 3))

        with self.assertRaises(ValueError):
            classification_model.fit(
                mols,
                test_data_df["Label"].add(1).to_numpy(),
            )
        with self.assertRaises(ValueError):
            classification_model = get_multiclass_classification_pipeline(n_classes=2)
            classification_model.fit(
                mols,
                test_data_df["Label"].to_numpy(),
            )
