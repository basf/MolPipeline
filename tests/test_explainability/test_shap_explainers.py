"""Test SHAP's TreeExplainer wrapper."""

import unittest
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from rdkit import Chem, rdBase
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.abstract_pipeline_elements.core import RDKitMol
from molpipeline.any2mol import SmilesToMol
from molpipeline.explainability.explainer import SHAPKernelExplainer, SHAPTreeExplainer
from molpipeline.explainability.explanation import (
    AtomExplanationMixin,
    SHAPFeatureAndAtomExplanation,
    SHAPFeatureExplanation,
)
from molpipeline.mol2any import (
    MolToConcatenatedVector,
    MolToMorganFP,
    MolToRDKitPhysChem,
)
from molpipeline.mol2mol import SaltRemover
from molpipeline.utils.subpipeline import (
    SubpipelineExtractor,
    get_featurization_subpipeline,
)

TEST_SMILES = ["CC", "CCO", "COC", "c1ccccc1(N)", "CCC(-O)O", "CCCN"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]

TEST_SMILES_WITH_BAD_SMILES = [
    "CC",
    "CCO",
    "COC",
    "MY_FIRST_BAD_SMILES",
    "c1ccccc1(N)",
    "CCC(-O)O",
    "CCCN",
    "BAD_SMILES_2",
]
CONTAINS_OX_BAD_SMILES = [0, 1, 1, 0, 0, 1, 0, 1]

_RANDOM_STATE = 67056


def _construct_kernel_shap_kwargs(
    pipeline: Pipeline, data: list[str]
) -> dict[str, Any]:
    """Construct the kwargs for SHAPKernelExplainer.

    Convert sparse matrix to dense array because SHAPKernelExplainer does not
    support sparse matrix as `data` and then explain dense matrices.
    We stick to dense matrices for simplicity.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline used for featurization.
    data : list[str]
        The input data, e.g. SMILES strings.

    Returns
    -------
    dict[str, Any]
        The kwargs for SHAPKernelExplainer
    """
    featurization_subpipeline = get_featurization_subpipeline(pipeline)
    data_transformed = featurization_subpipeline.transform(data)
    if scipy.sparse.issparse(data_transformed):
        data_transformed = data_transformed.toarray()
    return {"data": data_transformed}


class TestSHAPExplainers(unittest.TestCase):
    """Test SHAP's Explainer wrappers."""

    def _test_valid_explanation(
        self,
        explanation: SHAPFeatureExplanation | SHAPFeatureAndAtomExplanation,
        estimator: BaseEstimator,
        molecule_reader_subpipeline: Pipeline,
        nof_features: int,
        test_smiles: str,
        explainer: SHAPKernelExplainer | SHAPTreeExplainer,
    ) -> None:
        """Helper method to test if the explanation is valid and has the correct shape and content.

        Parameters
        ----------
        explanation : Explanation
            The explanation to be tested.
        estimator : BaseEstimator
            The estimator used in the pipeline.
        molecule_reader_subpipeline : Pipeline
            The subpipeline that extracts the molecule from the input data.
        nof_features : int
            The number of features in the feature vector.
        test_smiles : str
            The SMILES string of the molecule.
        explainer : SHAPKernelExplainer | SHAPTreeExplainer
            The explainer used to generate the explanation.
        """
        self.assertTrue(explanation.is_valid())

        self.assertIsInstance(explanation.feature_vector, np.ndarray)
        self.assertEqual(
            (nof_features,), explanation.feature_vector.shape  # type: ignore[union-attr]
        )

        # feature names are not implemented yet
        self.assertIsNone(explanation.feature_names)
        # self.assertEqual(len(explanation.feature_names), explanation.feature_vector.shape[0])

        self.assertIsInstance(explanation.molecule, RDKitMol)
        self.assertEqual(
            Chem.MolToInchi(*molecule_reader_subpipeline.transform([test_smiles])),
            Chem.MolToInchi(explanation.molecule),
        )

        self.assertIsInstance(explanation.prediction, np.ndarray)
        self.assertIsInstance(explanation.feature_weights, np.ndarray)
        if is_regressor(estimator):
            self.assertTrue((1,), explanation.prediction.shape)  # type: ignore[union-attr]
            self.assertEqual(
                (nof_features,), explanation.feature_weights.shape  # type: ignore[union-attr]
            )
        elif is_classifier(estimator):
            self.assertTrue((2,), explanation.prediction.shape)  # type: ignore[union-attr]
            if isinstance(explainer, SHAPTreeExplainer) and isinstance(
                estimator, GradientBoostingClassifier
            ):
                # there is currently a bug in SHAP's TreeExplainer for GradientBoostingClassifier
                # https://github.com/shap/shap/issues/3177 returning only one feature weight
                # which is also based on log odds. This check is a workaround until the bug is fixed.
                self.assertEqual(
                    (nof_features,), explanation.feature_weights.shape  # type: ignore[union-attr]
                )
            elif isinstance(estimator, SVC):
                # SVC seems to be handled differently by SHAP. It returns only a one dimensional
                # feature array for binary classification.
                self.assertTrue(
                    (1,), explanation.prediction.shape
                )  # type: ignore[union-attr]
                self.assertEqual(
                    (nof_features,), explanation.feature_weights.shape  # type: ignore[union-attr]
                )
            else:
                # normal binary classification case
                self.assertEqual(
                    (nof_features, 2), explanation.feature_weights.shape  # type: ignore[union-attr]
                )
        else:
            raise ValueError("Error in unittest. Unsupported estimator.")

        if issubclass(type(explainer), AtomExplanationMixin):
            self.assertIsInstance(explanation.atom_weights, np.ndarray)
            self.assertEqual(
                explanation.atom_weights.shape,  # type: ignore[union-attr]
                (explanation.molecule.GetNumAtoms(),),  # type: ignore[union-attr]
            )

    def test_explanations_fingerprint_pipeline(self) -> None:
        """Test SHAP's TreeExplainer wrapper on MolPipeline's pipelines with fingerprints."""

        tree_estimators = [
            RandomForestClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            RandomForestRegressor(n_estimators=2, random_state=_RANDOM_STATE),
            GradientBoostingClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            GradientBoostingRegressor(n_estimators=2, random_state=_RANDOM_STATE),
        ]
        other_estimators = [
            SVC(kernel="rbf", probability=False, random_state=_RANDOM_STATE),
            SVR(kernel="linear"),
            LogisticRegression(random_state=_RANDOM_STATE),
            LinearRegression(),
        ]
        n_bits = 64

        explainer_types = [SHAPKernelExplainer, SHAPTreeExplainer]
        explainer_estimators = [tree_estimators + other_estimators, tree_estimators]

        for estimators, explainer_type in zip(explainer_estimators, explainer_types):

            # test explanations with different estimators
            for estimator in estimators:
                pipeline = Pipeline(
                    [
                        ("smi2mol", SmilesToMol()),
                        ("morgan", MolToMorganFP(radius=1, n_bits=n_bits)),
                        ("model", estimator),
                    ]
                )
                pipeline.fit(TEST_SMILES, CONTAINS_OX)

                # some explainers require additional kwargs
                explainer_kwargs = {}
                if explainer_type == SHAPKernelExplainer:
                    explainer_kwargs = _construct_kernel_shap_kwargs(
                        pipeline, TEST_SMILES
                    )

                explainer = explainer_type(pipeline, **explainer_kwargs)
                explanations = explainer.explain(TEST_SMILES)
                self.assertEqual(len(explanations), len(TEST_SMILES))

                self.assertTrue(explainer.has_atom_weights_)
                self.assertTrue(
                    issubclass(explainer.return_element_type_, AtomExplanationMixin)
                )

                # get the subpipeline that extracts the molecule from the input data
                mol_reader_subpipeline = SubpipelineExtractor(
                    pipeline
                ).get_molecule_reader_subpipeline()
                self.assertIsInstance(mol_reader_subpipeline, Pipeline)

                for i, explanation in enumerate(explanations):
                    self._test_valid_explanation(
                        explanation,
                        estimator,
                        mol_reader_subpipeline,  # type: ignore[arg-type]
                        n_bits,
                        TEST_SMILES[i],
                        explainer=explainer,
                    )

    # pylint: disable=too-many-locals
    def test_explanations_pipeline_with_invalid_inputs(self) -> None:
        """Test SHAP's TreeExplainer wrapper with invalid inputs."""

        # estimators to test
        estimators = [
            RandomForestClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            RandomForestRegressor(n_estimators=2, random_state=_RANDOM_STATE),
            GradientBoostingClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            GradientBoostingRegressor(n_estimators=2, random_state=_RANDOM_STATE),
        ]

        # fill values considered invalid predictions
        invalid_fill_values = [None, np.nan, pd.NA]
        # fill values considered valid predictions (although outside the valid range)
        valid_fill_values = [0, 999]
        # fill values to test
        fill_values = invalid_fill_values + valid_fill_values

        n_bits = 64

        for estimator in estimators:
            for fill_value in fill_values:

                # pipeline with ErrorFilter
                error_filter1 = ErrorFilter()
                pipeline1 = Pipeline(
                    [
                        ("smi2mol", SmilesToMol()),
                        ("salt_remover", SaltRemover()),
                        ("error_filter", error_filter1),
                        ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                        ("model", estimator),
                    ]
                )

                # pipeline with ErrorFilter and FilterReinserter
                error_filter2 = ErrorFilter()
                error_reinserter2 = PostPredictionWrapper(
                    FilterReinserter.from_error_filter(error_filter2, fill_value)
                )
                pipeline2 = Pipeline(
                    [
                        ("smi2mol", SmilesToMol()),
                        ("salt_remover", SaltRemover()),
                        ("error_filter", error_filter2),
                        ("morgan", MolToMorganFP(radius=1, n_bits=n_bits)),
                        ("model", estimator),
                        ("error_reinserter", error_reinserter2),
                    ]
                )

                for pipeline in [pipeline1, pipeline2]:

                    pipeline.fit(TEST_SMILES_WITH_BAD_SMILES, CONTAINS_OX_BAD_SMILES)

                    explainer = SHAPTreeExplainer(pipeline)
                    log_block = rdBase.BlockLogs()  # pylint: disable=unused-variable
                    explanations = explainer.explain(TEST_SMILES_WITH_BAD_SMILES)
                    del log_block
                    self.assertEqual(
                        len(explanations), len(TEST_SMILES_WITH_BAD_SMILES)
                    )

                    # get the subpipeline that extracts the molecule from the input data
                    mol_reader_subpipeline = SubpipelineExtractor(
                        pipeline
                    ).get_molecule_reader_subpipeline()
                    self.assertIsNotNone(mol_reader_subpipeline)

                    for i, explanation in enumerate(explanations):
                        if i in [3, 7]:
                            self.assertFalse(explanation.is_valid())
                            continue

                        self._test_valid_explanation(
                            explanation,
                            estimator,
                            mol_reader_subpipeline,  # type: ignore[arg-type]
                            n_bits,
                            TEST_SMILES_WITH_BAD_SMILES[i],
                            explainer=explainer,
                        )

    def test_explanations_pipeline_with_physchem(self) -> None:
        """Test SHAP's TreeExplainer wrapper on physchem feature vector."""

        estimators = [
            RandomForestClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            RandomForestRegressor(n_estimators=2, random_state=_RANDOM_STATE),
            GradientBoostingClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            GradientBoostingRegressor(n_estimators=2, random_state=_RANDOM_STATE),
        ]

        # test explanations with different estimators
        for estimator in estimators:
            pipeline = Pipeline(
                [
                    ("smi2mol", SmilesToMol()),
                    ("physchem", MolToRDKitPhysChem()),
                    ("model", estimator),
                ]
            )

            pipeline.fit(TEST_SMILES, CONTAINS_OX)

            explainer = SHAPTreeExplainer(pipeline)
            explanations = explainer.explain(TEST_SMILES)
            self.assertEqual(len(explanations), len(TEST_SMILES))

            # get the subpipeline that extracts the molecule from the input data
            mol_reader_subpipeline = SubpipelineExtractor(
                pipeline
            ).get_molecule_reader_subpipeline()
            self.assertIsNotNone(mol_reader_subpipeline)

            for i, explanation in enumerate(explanations):
                self._test_valid_explanation(
                    explanation,
                    estimator,
                    mol_reader_subpipeline,  # type: ignore[arg-type]
                    pipeline.named_steps["physchem"].n_features,
                    TEST_SMILES[i],
                    explainer=explainer,
                )

    def test_explanations_pipeline_with_concatenated_features(self) -> None:
        """Test SHAP's TreeExplainer wrapper on concatenated feature vector."""

        estimators = [
            RandomForestClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            RandomForestRegressor(n_estimators=2, random_state=_RANDOM_STATE),
            GradientBoostingClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            GradientBoostingRegressor(n_estimators=2, random_state=_RANDOM_STATE),
        ]

        n_bits = 64

        # test explanations with different estimators
        for estimator in estimators:
            pipeline = Pipeline(
                [
                    ("smi2mol", SmilesToMol()),
                    (
                        "features",
                        MolToConcatenatedVector(
                            [
                                (
                                    "RDKitPhysChem",
                                    MolToRDKitPhysChem(),
                                ),
                                (
                                    "MorganFP",
                                    MolToMorganFP(radius=1, n_bits=n_bits),
                                ),
                            ]
                        ),
                    ),
                    ("model", estimator),
                ]
            )

            pipeline.fit(TEST_SMILES, CONTAINS_OX)

            explainer = SHAPTreeExplainer(pipeline)
            explanations = explainer.explain(TEST_SMILES)
            self.assertEqual(len(explanations), len(TEST_SMILES))

            # get the subpipeline that extracts the molecule from the input data
            mol_reader_subpipeline = SubpipelineExtractor(
                pipeline
            ).get_molecule_reader_subpipeline()
            self.assertIsNotNone(mol_reader_subpipeline)

            for i, explanation in enumerate(explanations):
                self._test_valid_explanation(
                    explanation,
                    estimator,
                    mol_reader_subpipeline,  # type: ignore[arg-type]
                    pipeline.named_steps["features"].n_features,
                    TEST_SMILES[i],
                    explainer=explainer,
                )
