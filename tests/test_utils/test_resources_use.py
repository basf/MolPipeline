"""Test resources changes of estimators."""

import unittest
from unittest.mock import MagicMock

from sklearn.calibration import (
    CalibratedClassifierCV,
    _CalibratedClassifier,  # noqa: PLC2701
)
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToConcatenatedVector
from molpipeline.mol2any.mol2morgan_fingerprint import MolToMorganFP
from molpipeline.utils.resources_use import (
    iterate_components,
    model_to_cpu,
    set_n_job_estimator,
    set_single_job,
)
from tests.utils.default_models import get_morgan_rf_pipeline


class TestSetSingleJob(unittest.TestCase):
    """Test the function set_single_job."""

    def test_set_single_job_none(self) -> None:
        """Test if the model is set to single job mode."""
        model = None
        changed = set_single_job(model)
        self.assertIs(changed, False)

    def test_set_single_job_rf(self) -> None:
        """Test if the Random Forest model is set to single-job mode."""
        for n_jobs, change_expected in [(-1, True), (1, False)]:
            with self.subTest(n_jobs=n_jobs, change_expected=change_expected):
                model = get_morgan_rf_pipeline(n_jobs=n_jobs)
                changed = set_single_job(model)
                self.assertEqual(changed, change_expected)
                self.assertEqual(model.n_jobs, 1)
                self.assertEqual(model.named_steps["rf"].n_jobs, 1)

    def test_set_single_job_fitted_rf(self) -> None:
        """Test if the fitted Random Forest model is set to single-job mode."""
        for n_jobs, change_expected in [(-1, True), (1, False)]:
            with self.subTest(n_jobs=n_jobs, change_expected=change_expected):
                model = get_morgan_rf_pipeline(n_jobs=n_jobs)
                model.fit(["C", "CC"], [1, 0])
                changed = set_single_job(model)
                self.assertEqual(changed, change_expected)
                self.assertEqual(model.n_jobs, 1)
                self.assertEqual(model.named_steps["rf"].n_jobs, 1)

    def test_fully_parallelized_pipeline(self) -> None:
        """Test if a fully parallelized pipeline is set to single-job mode.

        Tests that n_jobs is correctly set:
        1. On the pipeline level
        2. For MolPipeline elements, e.g. AutoToMol
        3. For Sklearn elements, e.g. RandomForestClassifier

        """
        for n_jobs, change_expected in [(2, True), (-1, True), (1, False)]:
            with self.subTest(n_jobs=n_jobs, change_expected=change_expected):
                model = Pipeline(
                    [
                        ("auto2mol", AutoToMol(n_jobs=n_jobs)),
                        ("rf", RandomForestClassifier(n_estimators=2, n_jobs=n_jobs)),
                    ],
                    n_jobs=n_jobs,
                )
                changed = set_single_job(model)
                self.assertEqual(changed, change_expected)

                self.assertEqual(model.n_jobs, 1)
                self.assertEqual(model.named_steps["auto2mol"].n_jobs, 1)
                self.assertEqual(model.named_steps["rf"].n_jobs, 1)

    def test_set_n_job_estimator_circular_estimator_attribute(self) -> None:
        """Test a circular ``estimator`` reference doesn't cause infinite recursion.

        Simulates an estimator whose ``.estimator`` attribute points back to
        itself, which would loop forever without cycle detection.

        """
        rf = RandomForestClassifier(n_estimators=2, n_jobs=-1)
        rf.estimator = rf  # create circular ref
        changed = set_n_job_estimator(rf, n_jobs=1, n_jobs_chemprop=None)
        self.assertIs(changed, True)
        self.assertEqual(rf.n_jobs, 1)

    def test_set_n_job_estimator_circular_estimators_list(self) -> None:
        """Test that a circular ``estimators_`` list does not cause infinite recursion.

        Simulates an estimator whose fitted ``.estimators_`` list contains
        a reference back to itself.

        """
        rf = RandomForestClassifier(n_estimators=2, n_jobs=-1)
        rf.estimators_ = [rf]  # circular ref in list
        changed = set_n_job_estimator(rf, n_jobs=1, n_jobs_chemprop=None)
        self.assertIs(changed, True)
        self.assertEqual(rf.n_jobs, 1)

    def test_set_n_job_estimator_shared_reference_visited_once(self) -> None:
        """Test that a shared estimator referenced multiple times is only visited once.

        When two pipeline steps point to the same object, cycle detection
        should cause the second visit to be a no-op, but the first visit
        should still apply changes.

        """
        shared_rf = RandomForestClassifier(n_estimators=2, n_jobs=-1)
        model = Pipeline(
            [
                ("rf1", shared_rf),
                ("rf2", shared_rf),  # same object referenced twice
            ],
        )
        changed = set_n_job_estimator(model, n_jobs=1, n_jobs_chemprop=None)
        self.assertIs(changed, True)
        self.assertEqual(shared_rf.n_jobs, 1)

    def test_set_n_job_estimator_rejects_magic_mock(self) -> None:
        """Test that a MagicMock does not cause infinite recursion.

        MagicMock dynamically generates attributes on ``hasattr`` calls,
        which would cause ``set_n_job_estimator`` to recurse endlessly
        through ``mock.estimator.estimator.estimator...`` without a
        ``BaseEstimator`` isinstance guard.

        """
        mock = MagicMock()
        changed = set_n_job_estimator(mock, n_jobs=1, n_jobs_chemprop=0)
        self.assertIs(changed, False)

    def test_set_single_job_rejects_magic_mock(self) -> None:
        """Test that set_single_job returns False for a MagicMock."""
        mock = MagicMock()
        changed = set_single_job(mock)
        self.assertIs(changed, False)


class TestModelToCPU(unittest.TestCase):
    """Test the function model_to_cpu."""

    def test_model_to_cpu_none(self) -> None:
        """Test if the model is set to CPU mode."""
        model = None
        changed = model_to_cpu(model)
        self.assertIs(changed, False)


class TestIterateComponents(unittest.TestCase):
    """Test the iterate_components function."""

    def test_iterate_simple_rf(self) -> None:
        """Test iterate_components with a simple Random Forest classifier."""
        rf = RandomForestClassifier(n_estimators=2, n_jobs=-1)
        components = [c for c, _n in iterate_components(rf)]
        # Should yield: RF itself + base estimator (DecisionTreeClassifier)
        self.assertEqual(len(components), 2)
        self.assertIn(rf, components)
        self.assertIsInstance(components[1], DecisionTreeClassifier)

    def test_iterate_pipeline(self) -> None:
        """Test iterate_components with a Pipeline."""
        model = get_morgan_rf_pipeline(n_jobs=-1)
        components = [c for c, _n in iterate_components(model)]
        self.assertEqual(len(components), 7)
        self.assertIn(model, components)
        self.assertIn(model.named_steps["smiles_to_mol"], components)
        self.assertIn(model.named_steps["morgan_fp"], components)
        self.assertIn(model.named_steps["error_filter"], components)
        self.assertIn(model.named_steps["rf"], components)
        self.assertIn(model.named_steps["error_reinserter"], components)
        self.assertIn(model.named_steps["rf"].estimator, components)

    def test_iterate_fitted_rf(self) -> None:
        """Test iterate_components with a fitted Random Forest.

        After fitting, RandomForestClassifier has an estimators_ list containing the
        individual decision trees, plus the unfitted base estimator.

        """
        n_estimators = 3
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 0])

        components = [c for c, _n in iterate_components(rf)]
        # Should yield: base estimator + 3 individual trees + the RF classifier itself
        self.assertEqual(len(components), 5)
        self.assertIn(rf, components)
        self.assertIn(rf.estimator, components)
        self.assertEqual(len(rf.estimators_), n_estimators)
        for est in rf.estimators_:
            self.assertIn(est, components)

    def test_iterate_calibrated_classifier_ensemble(self) -> None:
        """Test iterate_components with a fitted ensemble CalibratedClassifierCV."""
        model = CalibratedClassifierCV(DecisionTreeClassifier(), cv=2, ensemble=True)
        model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 1, 0, 1])
        components = [c for c, _n in iterate_components(model)]

        # Should include the wrapper and the nested pipeline and its steps
        self.assertEqual(len(components), 6)
        self.assertIn(model, components)
        self.assertIn(model.estimator, components)
        self.assertEqual(len(list(model.calibrated_classifiers_)), 2)
        for estimator in model.calibrated_classifiers_:
            self.assertIn(estimator, components)
            self.assertIsInstance(estimator, _CalibratedClassifier)
            self.assertIn(estimator.estimator, components)

    def test_iterate_calibrated_classifier_no_ensemble(self) -> None:
        """Test iterate_components with a fitted non-ensemble CalibratedClassifierCV."""
        model = CalibratedClassifierCV(DecisionTreeClassifier(), cv=2, ensemble=False)
        model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 1, 0, 1])
        components = [c for c, _n in iterate_components(model)]

        # Should include the wrapper and the nested pipeline and its steps
        self.assertEqual(len(components), 4)
        self.assertIn(model, components)
        self.assertIn(model.estimator, components)
        classifier_list = model.calibrated_classifiers_
        self.assertIsInstance(classifier_list, list)
        self.assertEqual(len(classifier_list), 1)  # not ensemble
        cal_est = classifier_list[0]
        self.assertIn(cal_est, components)
        self.assertIsInstance(cal_est, _CalibratedClassifier)
        self.assertIn(cal_est.estimator, components)

    def test_iterate_circular_estimator_attribute(self) -> None:
        """Test that circular references in estimator attribute are handled."""
        bc = BaggingClassifier(n_estimators=2, n_jobs=-1)
        bc.estimator = bc

        components = [c for c, _n in iterate_components(bc)]
        self.assertEqual(len(components), 1)
        self.assertIn(bc, components)

    def test_iterate_shared_reference(self) -> None:
        """Test that shared estimators referenced multiple times are yielded once."""
        shared_dt = DecisionTreeClassifier()
        model = VotingClassifier([("rf1", shared_dt), ("rf2", shared_dt)])

        components = [c for c, _n in iterate_components(model)]
        # Should yield: VotingClassifier + shared_dt (only once)
        self.assertEqual(len(components), 2)
        self.assertIn(model, components)
        self.assertIn(shared_dt, components)

    def test_iterate_magic_mock(self) -> None:
        """Test that MagicMock doesn't cause infinite recursion."""
        components = [c for c, _n in iterate_components(MagicMock())]
        # Should return empty list for MagicMock
        self.assertEqual(len(components), 0)

    def test_iterate_seen_parameter(self) -> None:
        """Test that iterate_components accepts a seen parameter."""
        dt = DecisionTreeClassifier()
        seen_set: set[int] = set()
        components = [c for c, _n in iterate_components(dt, seen=seen_set)]
        # Should yield the DecisionTreeClassifier only
        self.assertEqual(len(components), 1)
        self.assertIn(dt, components)
        self.assertIn(id(dt), seen_set)  # The seen set should be modified in place

    def test_iterate_nested_pipelines(self) -> None:
        """Test iterate_components with nested pipelines."""
        inner_pipeline = Pipeline(
            [
                ("auto2mol", AutoToMol(n_jobs=1)),
                ("dt", DecisionTreeClassifier()),
            ],
        )
        outer_pipeline = Pipeline([("inner", inner_pipeline)])

        components = [c for c, _n in iterate_components(outer_pipeline)]

        self.assertEqual(len(components), 4)
        self.assertIn(outer_pipeline, components)
        self.assertIn(inner_pipeline, components)
        self.assertIn(inner_pipeline.named_steps["auto2mol"], components)
        self.assertIn(inner_pipeline.named_steps["dt"], components)

    def test_iterate_mol_to_concatenated_vector(self) -> None:
        """Test iterate_components recurses into MolToConcatenatedVector."""
        morgan_fp = MolToMorganFP(n_bits=512)
        morgan_fp2 = MolToMorganFP(n_bits=1024)
        concat_vec = MolToConcatenatedVector(
            [("morgan1", morgan_fp), ("morgan2", morgan_fp2)],
        )

        components, names = zip(*iterate_components(concat_vec), strict=True)
        # Should yield: concat_vec itself + morgan_fp + morgan_fp2
        self.assertEqual(len(components), 3)
        self.assertIn(concat_vec, components)
        self.assertIn(morgan_fp, components)
        self.assertIn(morgan_fp2, components)
        self.assertIn("morgan1", names)
        self.assertIn("morgan2", names)


if __name__ == "__main__":
    unittest.main()
