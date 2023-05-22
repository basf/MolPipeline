"""Contains functions for loading and saving pipeline elements and models to json files."""

from __future__ import annotations
from typing import Any, Callable, Union

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
    module_str: str = json_dict["module"]
    class_str: str = json_dict["type"]
    class_module = __import__(module_str, fromlist=[class_str])
    pipeline_element_class = getattr(class_module, class_str)
    return pipeline_element_class.from_json(json_dict)


def dict_with_function_to_str_dict(json_dict: dict[str, Any]) -> dict[str, Any]:
    """Find functions in the dictionary and replace them with their string representation

    Parameters
    ----------
    json_dict: dict[str, Any]
        Dictionary to be transformed to a json file.

    Returns
    -------
    str
        Json file containing the dictionary.
    """
    out_dict = {}
    for key, value in json_dict.items():
        # If there is a good method to detect classes this could be extended to classes
        # May need a method to get class attributes.
        if callable(value):
            out_dict[key] = "load_from_constructor"
            out_dict[key + "_construction_config"] = {
                "module": value.__module__,
                "type": value.__name__,
            }
        elif isinstance(value, dict):
            out_dict[key] = dict_with_function_to_str_dict(value)
        else:
            out_dict[key] = value
    return out_dict


def dict_with_function_from_str_dict(json_dict: dict[str, Any]) -> dict[str, Any]:
    """Find string representation of functions in the dictionary and replace them with their implementation.

    Parameters
    ----------
    json_dict: dict[str, Any]
        Dictionary with string representations of functions.

    Returns
    -------
    dict[str, Any]
        Dictionary with proper functions.
    """
    out_dict = {}
    for key, value in json_dict.items():
        if key.endswith("_construction_config"):
            continue
        if value == "load_from_constructor":
            module_str: str = json_dict[key + "_construction_config"]["module"]
            obj_str: str = json_dict[key + "_construction_config"]["type"]
            obj_module = __import__(module_str, fromlist=[obj_str])
            out_dict[key] = getattr(obj_module, obj_str)
        elif isinstance(value, dict):
            out_dict[key] = dict_with_function_from_str_dict(value)
        else:
            out_dict[key] = value
    return out_dict


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
        "type": model.__class__.__name__,
        "module": model.__class__.__module__,
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
    json_dict.update(dict_with_function_to_str_dict(model_params))
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
    model_module_str: str = model_dict.pop("module")
    model_class_str: str = model_dict.pop("type")
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
    model_params = dict_with_function_from_str_dict(model_dict)
    return model_class(**model_params)
