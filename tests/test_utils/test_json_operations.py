import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from molpipeline.utils.json_operations import sklearn_model_from_json, sklearn_model_to_json


class JsonConversionTest(unittest.TestCase):
    def test_rf_reconstruction(self) -> None:
        rf = RandomForestClassifier(n_estimators=200)
        recreated_rf = sklearn_model_from_json(sklearn_model_to_json(rf))
        self.assertEqual(rf.get_params(), recreated_rf.get_params())

    def test_svc_reconstruction(self) -> None:
        svc = SVC()
        recreated_svc = sklearn_model_from_json(sklearn_model_to_json(svc))
        self.assertEqual(svc.get_params(), recreated_svc.get_params())

    def test_pipeline_reconstruction(self) -> None:
        rf = RandomForestClassifier(n_estimators=200)
        svc = SVC()
        pipeline = Pipeline([("rf", rf), ("svc", svc)])
        recreated_pipeline = sklearn_model_from_json(sklearn_model_to_json(pipeline))

        original_params = pipeline.get_params()
        recreated_params = recreated_pipeline.get_params()
        original_steps = original_params.pop("steps")
        recreated_steps = recreated_params.pop("steps")

        # Separate comparison of the steps as models cannot be compared directly
        for (orig_name, orig_obj), (recreated_name, recreated_obj) in zip(original_steps, recreated_steps):
            # Remove the model from the original params
            del original_params[orig_name]
            del recreated_params[recreated_name]
            self.assertEqual(orig_name, recreated_name)
            self.assertEqual(orig_obj.get_params(), recreated_obj.get_params())
            self.assertEqual(type(orig_obj), type(recreated_obj))
        self.assertEqual(original_params, recreated_params)

if __name__ == '__main__':
    unittest.main()
