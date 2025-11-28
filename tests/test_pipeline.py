"""Test functionality of the pipeline class."""

from __future__ import annotations

import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import AutoToMol, SmilesToMol
from molpipeline.mol2any import MolToMorganFP, MolToRDKitPhysChem, MolToSmiles
from molpipeline.mol2mol import (
    ChargeParentExtractor,
    EmptyMoleculeFilter,
    MetalDisconnector,
    SaltRemover,
)
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json
from molpipeline.utils.matrices import are_equal
from tests import TEST_DATA_DIR
from tests.utils.execution_count import get_exec_counted_rf_regressor
from tests.utils.fingerprints import make_sparse_fp

TEST_SMILES = ["CC", "CCO", "COC", "CCCCC", "CCC(-O)O", "CCCN"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]
FAULTY_TEST_SMILES = ["CCCXAS", "", "O=C(O)C(F)(F)F"]
FAULTY_CONTAINS_OX = [0, 0, 1]

FP_RADIUS = 2
FP_SIZE = 2048
EXPECTED_OUTPUT = make_sparse_fp(TEST_SMILES, FP_RADIUS, FP_SIZE)

_RANDOM_STATE = 67056


class PipelineTest(unittest.TestCase):
    """Unit test for the functionality of the pipeline class."""

    def test_fit_transform_single_core(self) -> None:
        """Test if the generation of the fingerprint matrix works as expected."""
        # Create pipeline
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
            ],
        )

        # Run pipeline
        matrix = pipeline.fit_transform(TEST_SMILES)

        # Compare with expected output
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix))

    def test_sklearn_pipeline(self) -> None:
        """Test if the pipeline can be used in a sklearn pipeline."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        d_tree = DecisionTreeClassifier()
        s_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
                ("decision_tree", d_tree),
            ],
        )
        s_pipeline.fit(TEST_SMILES, CONTAINS_OX)
        predicted_value_array = s_pipeline.predict(TEST_SMILES)
        for pred_val, true_val in zip(predicted_value_array, CONTAINS_OX, strict=True):
            self.assertEqual(pred_val, true_val)

    def test_sklearn_pipeline_parallel(self) -> None:
        """Test if the pipeline can be used in a sklearn pipeline."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        d_tree = DecisionTreeClassifier()
        s_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
                ("decision_tree", d_tree),
            ],
            n_jobs=2,
        )
        s_pipeline.fit(TEST_SMILES, CONTAINS_OX)
        out = s_pipeline.predict(TEST_SMILES)
        self.assertEqual(len(out), len(CONTAINS_OX))
        for pred_val, true_val in zip(out, CONTAINS_OX, strict=True):
            self.assertEqual(pred_val, true_val)

    def test_salt_removal(self) -> None:
        """Test if salts are correctly removed from molecules."""
        smiles_with_salt_list = ["CCO-[Na]", "CCC(=O)[O-].[Li+]", "CCC(=O)-O-[K]"]
        smiles_without_salt_list = ["CCO", "CCC(=O)O", "CCC(=O)O"]

        smi2mol = SmilesToMol()
        disconnect_metal = MetalDisconnector()
        salt_remover = SaltRemover()
        empty_mol_filter = EmptyMoleculeFilter()
        remove_charge = ChargeParentExtractor()
        mol2smi = MolToSmiles()

        salt_remover_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("disconnect_metal", disconnect_metal),
                ("salt_remover", salt_remover),
                ("empty_mol_filter", empty_mol_filter),
                ("remove_charge", remove_charge),
                ("mol2smi", mol2smi),
            ],
        )
        generated_smiles = salt_remover_pipeline.transform(smiles_with_salt_list)
        for generated_smi, smiles_without_salt in zip(
            generated_smiles,
            smiles_without_salt_list,
            strict=True,
        ):
            self.assertEqual(generated_smi, smiles_without_salt)

    def test_json_generation(self) -> None:
        """Test that the json representation of a pipeline can be loaded back.

        This test verifies that a pipeline can be loaded back into a pipeline.
        """
        # Create pipeline
        smi2mol = SmilesToMol()
        metal_disconnector = MetalDisconnector()
        salt_remover = SaltRemover()
        physchem = MolToRDKitPhysChem()
        pipeline_element_list = [
            smi2mol,
            metal_disconnector,
            salt_remover,
            physchem,
        ]
        m_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("metal_disconnector", metal_disconnector),
                ("salt_remover", salt_remover),
                ("physchem", physchem),
            ],
        )

        # Convert pipeline to json
        json_str = recursive_to_json(m_pipeline)
        # Recreate pipeline from json
        loaded_pipeline: Pipeline = recursive_from_json(json_str)
        self.assertIsInstance(loaded_pipeline, Pipeline)
        # Compare pipeline elements
        for loaded_element, original_element in zip(
            loaded_pipeline.steps,
            pipeline_element_list,
            strict=True,
        ):
            if loaded_element[1] == "passthrough":
                self.assertEqual(loaded_element[1], original_element)
                continue
            loaded_params = loaded_element[1].get_params()
            original_params = original_element.get_params()
            for key, value in loaded_params.items():
                if isinstance(value, BaseEstimator):
                    self.assertEqual(type(value), type(original_params[key]))
                else:
                    self.assertEqual(value, original_params[key])

    def test_fit_transform_record_remove_nones(self) -> None:
        """Test if the generation of the fingerprint matrix works as expected."""
        smi2mol = SmilesToMol()
        salt_remover = SaltRemover()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        empty_mol_filter = EmptyMoleculeFilter()
        remove_none = ErrorFilter.from_element_list(
            [smi2mol, salt_remover, mol2morgan, empty_mol_filter],
        )
        # Create pipeline
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("salt_remover", salt_remover),
                ("empty_mol_filter", empty_mol_filter),
                ("morgan", mol2morgan),
                ("remove_none", remove_none),
            ],
        )

        # Run pipeline
        matrix = pipeline.fit_transform(TEST_SMILES + FAULTY_TEST_SMILES)
        # Compare with expected output
        # (Which is the same as the output without the faulty smiles)
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix))

    def test_caching(self) -> None:
        """Test if the caching gives the same results is faster on the second run."""
        molecule_net_logd_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_logd.tsv.gz",
            sep="\t",
            nrows=20,
        )
        prediction_list = []
        for cache_activated in [False, True]:
            pipeline = get_exec_counted_rf_regressor(_RANDOM_STATE)
            with tempfile.TemporaryDirectory() as temp_dir:
                if cache_activated:
                    cache_dir = Path(temp_dir) / ".cache"
                    mem = Memory(location=cache_dir, verbose=0)
                else:
                    mem = Memory(location=None, verbose=0)
                pipeline.memory = mem
                # Run fitting 1
                pipeline.fit(
                    molecule_net_logd_df["smiles"].tolist(),
                    molecule_net_logd_df["exp"].tolist(),
                )
                # Get predictions
                prediction = pipeline.predict(molecule_net_logd_df["smiles"].tolist())
                prediction_list.append(prediction)

                # Reset the last step with an untrained model
                pipeline.steps[-1] = (
                    "rf",
                    RandomForestRegressor(random_state=_RANDOM_STATE, n_jobs=1),
                )

                # Run fitting 2
                pipeline.fit(
                    molecule_net_logd_df["smiles"].tolist(),
                    molecule_net_logd_df["exp"].tolist(),
                )
                # Get predictions
                prediction = pipeline.predict(molecule_net_logd_df["smiles"].tolist())
                prediction_list.append(prediction)

                n_transformations = pipeline.named_steps["mol2concat"].n_transformations
                if cache_activated:
                    # Fit is called twice, but the transform is only called once,
                    # since the second run is cached
                    self.assertEqual(n_transformations, 1)
                else:
                    self.assertEqual(n_transformations, 2)

                mem.clear(warn=False)
            for pred1, pred2 in combinations(prediction_list, 2):
                self.assertTrue(np.allclose(pred1, pred2))

    def test_input_data_types_fit_transform(self) -> None:
        """Test the pipeline's fit_transform can handle different input data types."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
            ],
        )

        # Test fit_transform with list of SMILES
        matrix_list = pipeline.fit_transform(TEST_SMILES)
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix_list))

        # Test fit_transform with numpy array of SMILES
        matrix_array = pipeline.fit_transform(np.array(TEST_SMILES))
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix_array))

        # Test fit_transform with pandas Series of SMILES
        matrix_series = pipeline.fit_transform(pd.Series(TEST_SMILES))
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix_series))

    def test_input_data_types_fit_transform_error_handling(self) -> None:
        """Test pipeline's fit_transform can handles input data types with errors."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense")
        error_filter = ErrorFilter(filter_everything=True)
        filter_reinserter = PostPredictionWrapper(
            FilterReinserter.from_error_filter(error_filter, np.nan),
        )
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("error_filter", error_filter),
                ("morgan", mol2morgan),
                ("filter_reinserter", filter_reinserter),
            ],
        )

        # Test fit_transform with list of SMILES
        fp_matrix = pipeline.fit_transform(FAULTY_TEST_SMILES)
        self.assertIsInstance(fp_matrix, np.ndarray)
        self.assertEqual(fp_matrix.shape, (len(FAULTY_TEST_SMILES), FP_SIZE))
        self.assertTrue(
            np.array_equal(
                np.isnan(fp_matrix).sum(axis=1),
                [FP_SIZE, 0, 0],
            ),
        )

        # Test fit_transform with np.ndarray of SMILES
        fp_matrix = pipeline.fit_transform(np.array(FAULTY_TEST_SMILES))
        self.assertIsInstance(fp_matrix, np.ndarray)
        self.assertEqual(fp_matrix.shape, (len(FAULTY_TEST_SMILES), FP_SIZE))
        self.assertTrue(
            np.array_equal(
                np.isnan(fp_matrix).sum(axis=1),
                [FP_SIZE, 0, 0],
            ),
        )

        # Test fit_transform with pd.Series of SMILES
        fp_matrix = pipeline.fit_transform(pd.Series(FAULTY_TEST_SMILES))
        self.assertIsInstance(fp_matrix, np.ndarray)
        self.assertEqual(fp_matrix.shape, (len(FAULTY_TEST_SMILES), FP_SIZE))
        self.assertTrue(
            np.array_equal(
                np.isnan(fp_matrix).sum(axis=1),
                [FP_SIZE, 0, 0],
            ),
        )

    def test_input_data_types_transform(self) -> None:
        """Test the pipeline's transform can handle different input data types."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
            ],
        )

        # Test transform with list of SMILES
        matrix_list = pipeline.transform(TEST_SMILES)
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix_list))

        # Test transform with numpy array of SMILES
        matrix_array = pipeline.transform(np.array(TEST_SMILES))
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix_array))

        # Test transform with pandas Series of SMILES
        matrix_series = pipeline.transform(pd.Series(TEST_SMILES))
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix_series))

    def test_input_data_types_transform_error_handling(self) -> None:
        """Test pipeline's transform can handles input data types with errors."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense")
        error_filter = ErrorFilter(filter_everything=True)
        filter_reinserter = PostPredictionWrapper(
            FilterReinserter.from_error_filter(error_filter, np.nan),
        )
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("error_filter", error_filter),
                ("morgan", mol2morgan),
                ("filter_reinserter", filter_reinserter),
            ],
        )

        # Test transform with list of SMILES
        fp_matrix = pipeline.transform(FAULTY_TEST_SMILES)
        self.assertIsInstance(fp_matrix, np.ndarray)
        self.assertEqual(fp_matrix.shape, (len(FAULTY_TEST_SMILES), FP_SIZE))
        self.assertTrue(
            np.array_equal(
                np.isnan(fp_matrix).sum(axis=1),
                [FP_SIZE, 0, 0],
            ),
        )

        # Test transform with np.ndarray of SMILES
        fp_matrix = pipeline.transform(np.array(FAULTY_TEST_SMILES))
        self.assertIsInstance(fp_matrix, np.ndarray)
        self.assertEqual(fp_matrix.shape, (len(FAULTY_TEST_SMILES), FP_SIZE))
        self.assertTrue(
            np.array_equal(
                np.isnan(fp_matrix).sum(axis=1),
                [FP_SIZE, 0, 0],
            ),
        )

        # Test transform with pd.Series of SMILES
        fp_matrix = pipeline.transform(pd.Series(FAULTY_TEST_SMILES))
        self.assertIsInstance(fp_matrix, np.ndarray)
        self.assertEqual(fp_matrix.shape, (len(FAULTY_TEST_SMILES), FP_SIZE))
        self.assertTrue(
            np.array_equal(
                np.isnan(fp_matrix).sum(axis=1),
                [FP_SIZE, 0, 0],
            ),
        )

    def test_input_data_types_error_handling_issue_193_valid_smiles(self) -> None:
        """Test if the pipeline works with error handling for expected data types."""
        error_filter = ErrorFilter(filter_everything=True)
        filter_reinserter = PostPredictionWrapper(
            FilterReinserter.from_error_filter(error_filter, np.nan),
        )

        pipeline_base = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("descriptor", MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)),
                ("error_filter", error_filter),
                ("model", RandomForestRegressor(n_estimators=2, random_state=42)),
                (
                    "reinserter",
                    filter_reinserter,
                ),
            ],
        )

        # test with a type list: valid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(TEST_SMILES, CONTAINS_OX)
        predictions_list = pipeline.predict(TEST_SMILES)
        self.assertIsInstance(predictions_list, np.ndarray)
        self.assertEqual(predictions_list.shape, (len(TEST_SMILES),))

        # test with type np.ndarray: valid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(np.array(TEST_SMILES), np.array(CONTAINS_OX))
        predictions_array = pipeline.predict(np.array(TEST_SMILES))
        self.assertIsInstance(predictions_array, np.ndarray)
        self.assertEqual(predictions_array.shape, (len(TEST_SMILES),))
        self.assertTrue(np.allclose(predictions_array, predictions_list))

        # test with a type pd.Series: valid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(pd.Series(TEST_SMILES), pd.Series(CONTAINS_OX))
        predictions_series = pipeline.predict(pd.Series(TEST_SMILES))
        self.assertIsInstance(predictions_series, np.ndarray)
        self.assertEqual(predictions_series.shape, (len(TEST_SMILES),))
        self.assertTrue(np.allclose(predictions_series, predictions_list))

    def test_input_data_types_error_handling_issue_193_invalid_smiles(self) -> None:
        """Test if the pipeline works with error handling for expected data types."""
        error_filter = ErrorFilter(filter_everything=True)
        filter_reinserter = PostPredictionWrapper(
            FilterReinserter.from_error_filter(error_filter, np.nan),
        )

        pipeline_base = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("descriptor", MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)),
                ("error_filter", error_filter),
                ("model", RandomForestRegressor(n_estimators=2, random_state=42)),
                (
                    "reinserter",
                    filter_reinserter,
                ),
            ],
        )

        test_smiles_only_faulty = ["CCXSA", "NO_VALID_smiles:)"]
        test_smiles_only_faulty_y = [0, 1]

        # test with type list: invalid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(FAULTY_TEST_SMILES, FAULTY_CONTAINS_OX)
        predictions_list_faulty = pipeline.predict(FAULTY_TEST_SMILES)
        self.assertIsInstance(predictions_list_faulty, np.ndarray)
        self.assertEqual(predictions_list_faulty.shape, (len(FAULTY_TEST_SMILES),))

        # test with type list: only invalid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(
            test_smiles_only_faulty,
            test_smiles_only_faulty_y,
        )
        predictions_list_only_faulty = pipeline.predict(test_smiles_only_faulty)
        # note that for all nan values, the output is a list and not np.array
        self.assertIsInstance(predictions_list_only_faulty, list)
        self.assertEqual(
            len(predictions_list_only_faulty),
            (len(test_smiles_only_faulty)),
        )
        self.assertTrue(np.isnan(predictions_list_only_faulty).all())

        # test with type np.ndarray: invalid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(
            np.array(FAULTY_TEST_SMILES),
            np.array(FAULTY_CONTAINS_OX),
        )
        predictions_array_faulty = pipeline.predict(np.array(FAULTY_TEST_SMILES))
        self.assertIsInstance(predictions_array_faulty, np.ndarray)
        self.assertEqual(
            predictions_array_faulty.shape,
            (len(FAULTY_TEST_SMILES),),
        )
        self.assertTrue(
            np.allclose(
                predictions_array_faulty,
                predictions_list_faulty,
                equal_nan=True,
            ),
        )

        # test with type np.ndarray: only invalid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(
            np.array(test_smiles_only_faulty),
            np.array(test_smiles_only_faulty_y),
        )
        predictions_array_only_faulty = pipeline.predict(
            np.array(test_smiles_only_faulty),
        )
        # note that for all nan values, the output is a list and not np.array
        self.assertIsInstance(predictions_array_only_faulty, list)
        self.assertEqual(
            len(predictions_array_only_faulty),
            (len(test_smiles_only_faulty)),
        )
        self.assertTrue(np.isnan(predictions_array_only_faulty).all())

        # test with a type pd.Series: invalid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(
            pd.Series(FAULTY_TEST_SMILES),
            pd.Series(FAULTY_CONTAINS_OX),
        )
        predictions_series_faulty = pipeline.predict(pd.Series(FAULTY_TEST_SMILES))
        self.assertIsInstance(predictions_series_faulty, np.ndarray)
        self.assertEqual(
            predictions_series_faulty.shape,
            (len(FAULTY_TEST_SMILES),),
        )
        self.assertTrue(
            np.allclose(
                predictions_series_faulty,
                predictions_list_faulty,
                equal_nan=True,
            ),
        )

        # test with a type pd.Series: only invalid SMILES
        pipeline = clone(pipeline_base)
        pipeline.fit(
            pd.Series(test_smiles_only_faulty),
            pd.Series(test_smiles_only_faulty_y),
        )
        predictions_series_only_faulty = pipeline.predict(
            pd.Series(test_smiles_only_faulty),
        )
        # note that for all nan values, the output is a list and not np.array
        self.assertIsInstance(predictions_series_only_faulty, list)
        self.assertEqual(
            len(predictions_series_only_faulty),
            (len(test_smiles_only_faulty)),
        )
        self.assertTrue(np.isnan(predictions_series_only_faulty).all())


class PipelineCompatibilityTest(unittest.TestCase):
    """Test if the pipeline is compatible with other sklearn functionalities."""

    def test_gridsearchcv(self) -> None:
        """Test if the MolPipeline can be used in sklearn's GridSearchCV."""
        descriptor_elements_to_test: list[dict[str, Any]] = [
            {
                "name": "morgan",
                "element": MolToMorganFP(),
                "param_grid": {"morgan__n_bits": [64, 128], "morgan__radius": [1, 2]},
            },
            {
                "name": "physchem",
                "element": MolToRDKitPhysChem(),
                "param_grid": {
                    "physchem__descriptor_list": [
                        ["HeavyAtomMolWt"],
                        ["HeavyAtomMolWt", "HeavyAtomCount"],
                    ],
                },
            },
        ]

        for test_data_dict in descriptor_elements_to_test:
            name = test_data_dict["name"]
            element = test_data_dict["element"]
            param_grid = test_data_dict["param_grid"]

            # set up a pipeline that trains
            # a random forest classifier on morgan fingerprints
            pipeline = Pipeline(
                [
                    ("auto2mol", AutoToMol()),
                    (name, element),
                    ("estimator", RandomForestClassifier()),
                ],
                n_jobs=1,
            )

            # define the hyperparameter space to try out
            grid_space = {
                "estimator__n_estimators": [1, 5],
                "estimator__random_state": [_RANDOM_STATE],
            }
            grid_space.update(param_grid)

            grid_search_cv = GridSearchCV(
                estimator=pipeline,
                param_grid=grid_space,
                cv=2,
                scoring="roc_auc",
                n_jobs=1,
            )

            grid_search_cv.fit(
                X=TEST_SMILES,
                y=CONTAINS_OX,
            )

            for k, value in param_grid.items():
                self.assertIn(grid_search_cv.best_params_[k], value)

    def test_gridsearch_cache(self) -> None:
        """Run GridSearchCV and check caching vs not caching gives same results."""
        h_params = {
            "rf__n_estimators": [1, 2],
        }
        # First without caching
        data_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_logd.tsv.gz",
            sep="\t",
            nrows=20,
        )
        best_param_dict = {}
        prediction_dict = {}
        for cache_activated in [True, False]:
            pipeline = get_exec_counted_rf_regressor(_RANDOM_STATE)
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = Path(temp_dir) / ".cache"
                if cache_activated:
                    mem = Memory(location=cache_dir, verbose=0)
                else:
                    mem = Memory(location=None, verbose=0)
                pipeline.memory = mem
                grid_search_cv = GridSearchCV(
                    estimator=pipeline,
                    param_grid=h_params,
                    cv=2,
                    scoring="neg_mean_squared_error",
                    n_jobs=1,
                    error_score="raise",
                    refit=True,
                    pre_dispatch=1,
                )
                grid_search_cv.fit(data_df["smiles"].tolist(), data_df["exp"].tolist())
                best_param_dict[cache_activated] = grid_search_cv.best_params_
                prediction_dict[cache_activated] = grid_search_cv.predict(
                    data_df["smiles"].tolist(),
                )
                mem.clear(warn=False)
        self.assertEqual(best_param_dict[True], best_param_dict[False])
        self.assertTrue(np.allclose(prediction_dict[True], prediction_dict[False]))

    def test_calibrated_classifier(self) -> None:
        """Test if the pipeline can be used with a CalibratedClassifierCV."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        d_tree = DecisionTreeClassifier()
        error_filter = ErrorFilter(filter_everything=True)
        s_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
                ("error_filter", error_filter),
                ("decision_tree", d_tree),
                (
                    "error_replacer",
                    PostPredictionWrapper(
                        FilterReinserter.from_error_filter(error_filter, np.nan),
                    ),
                ),
            ],
        )
        calibrated_pipeline = CalibratedClassifierCV(
            s_pipeline,
            cv=2,
            ensemble=True,
            method="isotonic",
        )
        calibrated_pipeline.fit(TEST_SMILES, CONTAINS_OX)
        predicted_value_array = calibrated_pipeline.predict(TEST_SMILES)
        predicted_proba_array = calibrated_pipeline.predict_proba(TEST_SMILES)
        self.assertIsInstance(predicted_value_array, np.ndarray)
        self.assertIsInstance(predicted_proba_array, np.ndarray)
        self.assertEqual(predicted_value_array.shape, (len(TEST_SMILES),))
        self.assertEqual(predicted_proba_array.shape, (len(TEST_SMILES), 2))


if __name__ == "__main__":
    unittest.main()
