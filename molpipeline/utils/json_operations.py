"""Contains functions for loading and saving pipeline elements and models to json files."""

from __future__ import annotations
import types
import warnings
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from molpipeline.abstract_pipeline_elements.core import ABCPipelineElement


def pipeline_element_from_json(json_dict: dict[str, Any]) -> ABCPipelineElement:
    """Find the class for the given PipelineElement and loads it with specified parameters.

    Parameters
    ----------
    json_dict: dict[str, Any]
        Dictionary containing the PipelineElement parameters.

    Returns
    -------
    ABCPipelineElement
        PipelineElement specified in the json file with correspondingly specified parameters.
    """
    module_str: str = json_dict["__module__"]
    class_str: str = json_dict["__name__"]
    class_module = __import__(module_str, fromlist=[class_str])
    pipeline_element_class = getattr(class_module, class_str)
    return pipeline_element_class.from_json(json_dict)


def transform_functions2string(value: Any) -> Any:
    """Transform functions to string representation.

    If the value is a function, it is transformed to a dictionary containing the module and the class name.
    If the value is a dictionary, the function is called recursively for each value.
    If the value is a list, the function is called recursively for each value.
    Else the value is returned as is.

    Parameters
    ----------
    value: Any
        Value which is transformed.

    Returns
    -------
    Any
        Json file containing the dictionary.
    """
    if callable(value):
        out_dict = {
            "load_from_constructor": True,
            "__name__": value.__name__,
            "__module__": value.__module__,
        }
        return out_dict

    if isinstance(value, dict):
        out_dict = {}
        for key, value in value.items():
            out_dict[key] = transform_functions2string(value)
        return out_dict

    elif isinstance(value, list):
        out_list = []
        for value in value:
            out_list.append(transform_functions2string(value))
        return out_list

    return value


def transform_string2function(value: Any) -> Any:
    """Transform string representation of functions to actual functions.

    If the value is a dictionary containing the key "load_from_constructor" and the value is True,
    the function is loaded from the module and class name.
    If the value is a dictionary, the function is called recursively for each value.
    If the value is a list, the function is called recursively for each value.
    Else the value is returned as is.

    Parameters
    ----------
    value: Any
        Object to be transformed

    Returns
    -------
    Any
        Json file containing the dictionary.
    """
    if isinstance(value, dict):
        if "load_from_constructor" in value:
            if value["load_from_constructor"]:
                module_str: str = value["__module__"]
                class_str: str = value["__name__"]
                class_module = __import__(module_str, fromlist=[class_str])
                return getattr(class_module, class_str)
            else:
                return value
        else:
            out_dict = {}
            for key, value in value.items():
                out_dict[key] = transform_string2function(value)
            return out_dict

    if isinstance(value, list):
        out_list = []
        for value in value:
            out_list.append(transform_string2function(value))
        return out_list

    return value


def sklearn_model_to_json(model: BaseEstimator) -> dict[str, Any]:
    """Extract the parameters of a sklearn model (or pipeline) and transform them to a json file.

    Parameters
    ----------
    model: BaseEstimator
        Model of which the parameters are extracted.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the model parameters.
    """
    json_dict = {
        "__name__": model.__class__.__name__,
        "__module__": model.__class__.__module__,
    }
    if isinstance(model, Pipeline):
        json_dict["steps"] = [
            (step_name, sklearn_model_to_json(step_model))
            for (step_name, step_model) in model.steps
        ]
        json_dict["memory"] = model.memory
        json_dict["verbose"] = model.verbose
        return json_dict
    model_params = model.get_params()
    json_dict.update(transform_functions2string(model_params))
    return json_dict


def sklearn_model_from_json(model_dict: dict[str, Any]) -> BaseEstimator:
    """Create a parameterized but untrained model from a json file.

    Parameters
    ----------
    model_dict: dict[str, Any]
        Dictionary containing the model parameters.

    Returns
    -------
    BaseEstimator
        Sklearn model with the parameters specified in the json file.
    """
    model_module_str: str = model_dict.pop("__module__")
    model_class_str: str = model_dict.pop("__name__")
    _ = model_dict.pop("__init__", None)  # For compatibility with new json files
    class_module = __import__(model_module_str, fromlist=[model_class_str])
    model_class = getattr(class_module, model_class_str)
    if model_class is Pipeline:
        steps = [
            (step_name, sklearn_model_from_json(step_model))
            for (step_name, step_model) in model_dict["steps"]
        ]
        return Pipeline(
            steps, memory=model_dict["memory"], verbose=model_dict["verbose"]
        )
    model_params = transform_string2function(model_dict)
    return model_class(**model_params)


def recursive_to_json(obj: Any) -> Any:
    """Recursively transform an object to a json file.

    Parameters
    ----------
    obj: Any
        Object to be transformed. Can be a string, int, float, bool, list, tuple, dict, callable or a sklearn model.
        A sklearn model is defined as an object with a get_params method.

    Returns
    -------
    dict[str, Any]
        Json file containing the dictionary.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, types.FunctionType):
        return {
            "__name__": obj.__name__,
            "__module__": obj.__module__,
            "__init__": False,
        }
    if isinstance(obj, dict):
        return {key: recursive_to_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        iter_list = [recursive_to_json(value) for value in obj]
        iterable_type = type(obj)
        return iterable_type(iter_list)

    # If neither of the above, it is assumed to be an object.
    object_dict: dict[str, Any] = {
        "__name__": obj.__class__.__name__,
        "__module__": obj.__class__.__module__,
        "__init__": True,
    }
    # If the object is a sklearn model, the parameters for initialization are extracted.
    if hasattr(obj, "get_params"):
        model_params = obj.get_params()
        for key, value in model_params.items():
            object_dict[key] = recursive_to_json(value)
        return object_dict

    # If the object is not a sklearn model, a warning is raised
    # as it might not be possible to recreate the object.
    warnings.warn(
        f"{type(obj)} has no get_params method. No parameters for initialization are retained."
    )
    return object_dict


def recursive_from_json(obj: Any) -> Any:
    """Recursively transform a json file to an object.

    Parameters
    ----------
    obj: Any
        Object to be transformed

    Returns
    -------
    Any
        Object specified in the json file.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        # Create copy
        object_params_copy = dict(obj)
        # For functions of classes
        obj_module_str = object_params_copy.pop("__module__", None)
        obj_class_str = object_params_copy.pop("__name__", None)
        initialize = object_params_copy.pop("__init__", False)

        # Remaining Values
        converted_dict = {}
        for key, value in object_params_copy.items():
            converted_dict[key] = recursive_from_json(value)

        # If the object is a function or a class
        if obj_module_str and obj_class_str:
            class_module = __import__(obj_module_str, fromlist=[obj_class_str])
            obj_class = getattr(class_module, obj_class_str)
            if not initialize:  # If the object is a function
                return obj_class
            # If the object is a class, but has no parameters
            if not converted_dict:
                return obj_class()
            # If the object is a class, and has parameters
            return obj_class(**converted_dict)
        return converted_dict

    if isinstance(obj, (list, tuple)):
        iter_list = [recursive_from_json(value) for value in obj]
        iterable_type = type(obj)
        return iterable_type(iter_list)

    raise TypeError(f"Unexpected Type: {type(obj)}")
