"""Contains functions for loading and saving pipeline elements and models to json files."""

from __future__ import annotations
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
    module_str: str = json_dict["module"]
    class_str: str = json_dict["type"]
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
            "type": value.__name__,
            "module": value.__module__,
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
                module_str: str = value["module"]
                class_str: str = value["type"]
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
    model_params = transform_string2function(model_dict)
    return model_class(**model_params)
