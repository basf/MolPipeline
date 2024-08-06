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
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from molpipeline import ErrorFilter, Pipeline
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
FAULTY_TEST_SMILES = ["CCCXAS", "", "O=C(O)C(F)(F)F"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]
FP_RADIUS = 2
FP_SIZE = 2048
EXPECTED_OUTPUT = make_sparse_fp(TEST_SMILES, FP_RADIUS, FP_SIZE)

_RANDOM_STATE = 67056


class PipelineTest(unittest.TestCase):
    """Unit test for the functionality of the pipeline class."""

    def test_fit_transform_single_core(self) -> None:
        """Test if the generation of the fingerprint matrix works as expected.

        Returns
        -------
        None
        """
        # Create pipeline
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
            ]
        )

        # Run pipeline
        matrix = pipeline.fit_transform(TEST_SMILES)

        # Compare with expected output
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix))

    def test_sklearn_pipeline(self) -> None:
        """Test if the pipeline can be used in a sklearn pipeline.

        Returns
        -------
        None
        """
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        d_tree = DecisionTreeClassifier()
        s_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
                ("decision_tree", d_tree),
            ]
        )
        s_pipeline.fit(TEST_SMILES, CONTAINS_OX)
        predicted_value_array = s_pipeline.predict(TEST_SMILES)
        for pred_val, true_val in zip(predicted_value_array, CONTAINS_OX):
            self.assertEqual(pred_val, true_val)

    def test_sklearn_pipeline_parallel(self) -> None:
        """Test if the pipeline can be used in a sklearn pipeline.

        Returns
        -------
        None
        """
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
        for pred_val, true_val in zip(out, CONTAINS_OX):
            self.assertEqual(pred_val, true_val)

    def test_salt_removal(self) -> None:
        """Test if salts are correctly removed from molecules.

        Returns
        -------
        None
        """
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
            ]
        )
        generated_smiles = salt_remover_pipeline.transform(smiles_with_salt_list)
        for generated_smiles, smiles_without_salt in zip(
            generated_smiles, smiles_without_salt_list
        ):
            self.assertEqual(generated_smiles, smiles_without_salt)

    def test_json_generation(self) -> None:
        """Test that the json representation of a pipeline can be loaded back into a pipeline.

        Returns
        -------
        None
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
            ]
        )

        # Convert pipeline to json
        json_str = recursive_to_json(m_pipeline)
        # Recreate pipeline from json
        loaded_pipeline: Pipeline = recursive_from_json(json_str)
        self.assertTrue(isinstance(loaded_pipeline, Pipeline))
        # Compare pipeline elements
        for loaded_element, original_element in zip(
            loaded_pipeline.steps, pipeline_element_list
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
                    self.assertEqual(loaded_params[key], original_params[key])

    def test_fit_transform_record_remove_nones(self) -> None:
        """Test if the generation of the fingerprint matrix works as expected.
        Returns
        -------
        None
        """
        smi2mol = SmilesToMol()
        salt_remover = SaltRemover()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE)
        empty_mol_filter = EmptyMoleculeFilter()
        remove_none = ErrorFilter.from_element_list(
            [smi2mol, salt_remover, mol2morgan, empty_mol_filter]
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
        # Compare with expected output (Which is the same as the output without the faulty smiles)
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix))

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
                    ]
                },
            },
        ]

        for test_data_dict in descriptor_elements_to_test:

            name = test_data_dict["name"]
            element = test_data_dict["element"]
            param_grid = test_data_dict["param_grid"]

            # set up a pipeline that trains a random forest classifier on morgan fingerprints
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

    def test_caching(self) -> None:
        """Test if the caching gives the same results and is faster on the second run."""

        molecule_net_logd_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_logd.tsv.gz", sep="\t", nrows=20
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
                    # Fit is called twice, but the transform is only called once, since the second run is cached
                    self.assertEqual(n_transformations, 1)
                else:
                    self.assertEqual(n_transformations, 2)

                mem.clear(warn=False)
            for pred1, pred2 in combinations(prediction_list, 2):
                self.assertTrue(np.allclose(pred1, pred2))

    def test_gridsearch_cache(self) -> None:
        """Run a short GridSearchCV and check if the caching and not caching gives the same results."""
        h_params = {
            "rf__n_estimators": [1, 2],
        }
        # First without caching
        data_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_logd.tsv.gz", sep="\t", nrows=20
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
                    data_df["smiles"].tolist()
                )
                mem.clear(warn=False)
        self.assertEqual(best_param_dict[True], best_param_dict[False])
        self.assertTrue(np.allclose(prediction_dict[True], prediction_dict[False]))


if __name__ == "__main__":
    unittest.main()
