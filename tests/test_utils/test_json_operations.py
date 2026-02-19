"""Tests conversion of sklearn models to json and back."""

import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from molpipeline import Pipeline
from molpipeline.utils.json_operations import (
    recursive_from_json,
    recursive_to_json,
)
from molpipeline.utils.multi_proc import check_available_cores


class JsonConversionTest(unittest.TestCase):
    """Unittest for conversion of sklearn models to json and back."""

    def test_rf_reconstruction(self) -> None:
        """Test if the sklearn-rf can be reconstructed from json."""
        random_forest = RandomForestClassifier(n_estimators=200)
        recreated_rf = recursive_from_json(recursive_to_json(random_forest))
        self.assertEqual(random_forest.get_params(), recreated_rf.get_params())

    def test_svc_reconstruction(self) -> None:
        """Test if the sklearn-svc can be reconstructed from json."""
        svc = SVC()
        recreated_svc = recursive_from_json(recursive_to_json(svc))
        self.assertEqual(svc.get_params(), recreated_svc.get_params())

    def test_pipeline_reconstruction(self) -> None:
        """Test if the sklearn-pipleine can be reconstructed from json."""
        random_forest = RandomForestClassifier(n_estimators=200)
        svc = SVC()
        pipeline = Pipeline([("rf", random_forest), ("svc", svc)])
        recreated_pipeline = recursive_from_json(recursive_to_json(pipeline))

        original_params = pipeline.get_params()
        recreated_params = recreated_pipeline.get_params()
        original_steps = original_params.pop("steps")
        recreated_steps = recreated_params.pop("steps")

        # Separate comparison of the steps as models cannot be compared directly
        for (orig_name, orig_obj), (recreated_name, recreated_obj) in zip(
            original_steps,
            recreated_steps,
            strict=True,
        ):
            # Remove the model from the original params
            del original_params[orig_name]
            del recreated_params[recreated_name]
            self.assertEqual(orig_name, recreated_name)
            self.assertEqual(orig_obj.get_params(), recreated_obj.get_params())
            self.assertEqual(type(orig_obj), type(recreated_obj))
        self.assertEqual(original_params, recreated_params)


    def test_set_transformation(self) -> None:
        """Test if a set can be reconstructed from json."""
        test_set = {1, "a", (1, "a")}
        test_set_json = recursive_to_json(test_set)
        recreated_set = recursive_from_json(test_set_json)
        self.assertEqual(test_set, recreated_set)

    def test_numpy_dtype_roundtrip(self) -> None:
        """Test if a numpy dtype can be reconstructed from json."""
        dtype_dict = {
            "int64": np.int64,
            "float32": np.float32,
            "object": np.object_,
        }
        for dtype_name, dtype in dtype_dict.items():
            with self.subTest(dtype=dtype_name):
                json_data = recursive_to_json(dtype)
                deserialized_dtype = recursive_from_json(json_data)
                self.assertEqual(dtype, deserialized_dtype)

    def test_array_roundtrip(self) -> None:
        """Test if a numpy array can be reconstructed from json."""
        dtype_dict = {
            "int64": np.int64,
            "float32": np.float32,
            "object": np.object_,
        }
        for dtype_name, dtype in dtype_dict.items():
            with self.subTest(dtype=dtype_name):
                original_array = np.array([[1, 2], [3, 4]], dtype=dtype)
                json_data = recursive_to_json(original_array)
                deserialized_array = recursive_from_json(json_data)
                self.assertTrue(np.array_equal(original_array, deserialized_array))
                self.assertEqual(original_array.dtype, deserialized_array.dtype)


if __name__ == "__main__":
    unittest.main()
