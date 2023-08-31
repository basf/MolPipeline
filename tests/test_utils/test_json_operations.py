import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from molpipeline.utils.json_operations import (
    transform_functions2string,
    transform_string2function,
    sklearn_model_from_json,
    sklearn_model_to_json,
)
from molpipeline.utils.multi_proc import check_available_cores
from molpipeline.pipeline import Pipeline


class JsonConversionTest(unittest.TestCase):
    def test_rf_reconstruction(self) -> None:
        """Test if the sklearn-rf can be reconstructed from json.

        Returns
        -------
        None
        """
        rf = RandomForestClassifier(n_estimators=200)
        recreated_rf = sklearn_model_from_json(sklearn_model_to_json(rf))
        self.assertEqual(rf.get_params(), recreated_rf.get_params())

    def test_svc_reconstruction(self) -> None:
        """Test if the sklearn-svc can be reconstructed from json.

        Returns
        -------
        None
        """
        svc = SVC()
        recreated_svc = sklearn_model_from_json(sklearn_model_to_json(svc))
        self.assertEqual(svc.get_params(), recreated_svc.get_params())

    def test_pipeline_reconstruction(self) -> None:
        """Test if the sklearn-pipleine can be reconstructed from json.

        Returns
        -------
        None
        """
        rf = RandomForestClassifier(n_estimators=200)
        svc = SVC()
        pipeline = Pipeline([("rf", rf), ("svc", svc)])
        recreated_pipeline = sklearn_model_from_json(sklearn_model_to_json(pipeline))

        original_params = pipeline.get_params()
        recreated_params = recreated_pipeline.get_params()
        original_steps = original_params.pop("steps")
        recreated_steps = recreated_params.pop("steps")

        # Separate comparison of the steps as models cannot be compared directly
        for (orig_name, orig_obj), (recreated_name, recreated_obj) in zip(
            original_steps, recreated_steps
        ):
            # Remove the model from the original params
            del original_params[orig_name]
            del recreated_params[recreated_name]
            self.assertEqual(orig_name, recreated_name)
            self.assertEqual(orig_obj.get_params(), recreated_obj.get_params())
            self.assertEqual(type(orig_obj), type(recreated_obj))
        self.assertEqual(original_params, recreated_params)

    def test_function_dict_json(self) -> None:
        """Test if a dict with objects can be reconstructed from json.

        Returns
        -------
        None
        """
        function_dict = {
            "dummy1": {"check_available_cores": check_available_cores},
            "dummy2": str,
            "dummy3": 1,
            "dummy4": [check_available_cores, check_available_cores, "test"],
        }
        function_json = transform_functions2string(function_dict)
        recreated_function_dict = transform_string2function(function_json)
        self.assertEqual(function_dict, recreated_function_dict)


if __name__ == "__main__":
    unittest.main()
