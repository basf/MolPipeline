"""Test functionality of the pipeline class."""

from __future__ import annotations

import time
import unittest
from typing import Any

from loguru import logger
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
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
    TautomerCanonicalizer,
)
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json
from molpipeline.utils.matrices import are_equal
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
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
        ).head(1000)
        smi2mol = SmilesToMol()
        tautomerizer = TautomerCanonicalizer()
        mol2smi = MolToSmiles()

        mem = Memory(location="./cache")
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("tautomerizer", tautomerizer),
                ("mol2smi", mol2smi),
            ],
            memory=mem,
            n_jobs=-1,
        )

        # Run pipeline

        start_time = time.time()
        smiles1 = pipeline.fit_transform(molecule_net_logd_df["smiles"].tolist())
        time1 = time.time() - start_time
        start_time = time.time()
        smiles2 = pipeline.fit_transform(molecule_net_logd_df["smiles"].tolist())
        time2 = time.time() - start_time
        self.assertListEqual(smiles1, smiles2)
        self.assertLess(time2, time1)
        logger.info(f"Original run took {time1} seconds, cached run took {time2} seconds")


if __name__ == "__main__":
    unittest.main()
