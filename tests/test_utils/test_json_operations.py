"""Tests conversion of sklearn models to json and back."""

import unittest
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from molpipeline import Pipeline
from molpipeline.utils.json_operations import (
    get_init_params,
    recursive_from_json,
    recursive_to_json,
)


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

    def test_function_roundtrip(self) -> None:
        """Test if a function can be reconstructed from json."""
        json_data = recursive_to_json(balanced_accuracy_score)
        deserialized_function = recursive_from_json(json_data)
        self.assertIs(deserialized_function, balanced_accuracy_score)


class ReconstructibleObject:  # pylint: disable=too-few-public-methods
    """Simple class with stable reconstruction state."""

    def __init__(self, required: int, optional: int = 2) -> None:
        """Initialize the object.

        Parameters
        ----------
        required : int
            A required parameter.
        optional : int, default=2
            An optional parameter with a default value.

        """
        self.required = required
        self.optional = optional
        self.sum = required + optional


class MissingRequiredStateParamObj(ReconstructibleObject):  # pylint: disable=too-few-public-methods
    """Class that omits a required init parameter from state."""

    def __getstate__(self) -> dict[str, Any]:
        """Mock the get_state and omit the parameter `required`.

        Returns
        -------
        dict[str, Any]
            State dictionary missing the required parameter.

        """
        state_dict = dict(self.__dict__)
        state_dict.pop("required")  # Remove the required parameter
        return state_dict


class ModifiedParamObj(ReconstructibleObject):  # pylint: disable=too-few-public-methods
    """Class that modifies a required init parameter in state."""

    def __init__(self, required: int, optional: int = 2) -> None:
        """Initialize the object.

        Parameters
        ----------
        required : int
            A required parameter.
        optional : int, default=2
            An optional parameter with a default value.

        """
        super().__init__(required * 2, optional)


class GetInitParamsTest(unittest.TestCase):
    """Test the get_init_params function."""

    def test_estimator(self) -> None:
        """Test extracting init params from a sklearn estimator."""
        estimator = RandomForestClassifier(n_estimators=15, random_state=7)
        init_params = get_init_params(estimator, validation="raise")
        self.assertEqual(init_params, estimator.get_params(deep=False))

    def test_splitter(self) -> None:
        """Test extracting init params from a sklearn splitter."""
        orig_params = {"n_splits": 4, "shuffle": True, "random_state": 13}
        splitter = StratifiedKFold(**orig_params)
        init_params = get_init_params(splitter, validation="raise")
        self.assertEqual(init_params, orig_params)

    def test_non_sklearn_object(self) -> None:
        """Test extracting init params from a custom serializable class."""
        custom_obj = ReconstructibleObject(required=3, optional=5)
        init_params = get_init_params(custom_obj, validation="raise")
        self.assertEqual(init_params, {"required": 3, "optional": 5})

    def test_return_none_for_missing_required_params(self) -> None:
        """Test return_none validation strategy for missing required params."""
        custom_obj = MissingRequiredStateParamObj(required=5, optional=9)
        init_params = get_init_params(custom_obj, validation="return_none")
        self.assertIsNone(init_params)

    def test_raises_for_missing_required_params(self) -> None:
        """Test raise validation strategy for missing required params."""
        custom_obj = MissingRequiredStateParamObj(required=5, optional=9)
        with self.assertRaises(ValueError):
            get_init_params(custom_obj, validation="raise")

    def test_missing_required_params_skip(self) -> None:
        """Test skip validation strategy for missing required params."""
        custom_obj = MissingRequiredStateParamObj(required=5, optional=9)
        init_params = get_init_params(custom_obj, validation="skip")
        self.assertIsNotNone(init_params)
        self.assertEqual(init_params, {"optional": 9})

    def test_modified_param_obj(self) -> None:
        """Test that modified parameters are correctly extracted."""
        custom_obj = ModifiedParamObj(required=4, optional=6)
        with self.assertRaisesRegex(ValueError, "Reconstructing the object failed."):
            get_init_params(custom_obj, validation="raise")


if __name__ == "__main__":
    unittest.main()
