from __future__ import annotations
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from molpipeline.abstract_pipeline_elements.core import ABCPipelineElement


def pipeline_element_from_json(json_dict: dict[str, Any]) -> ABCPipelineElement:
    """Finds the class for the given PipelineElement and loads it with specified parameters.

    Parameters
    ----------
    json_dict: dict[str, Any]
        Dictionary containing the PipelineElement parameters.

    Returns
    -------
    ABCPipelineElement
    """
    module_str: str = json_dict["module"]
    class_str: str = json_dict["type"]
    class_module = __import__(module_str, fromlist=[class_str])
    pipeline_element_class = getattr(class_module, class_str)
    return pipeline_element_class.from_json(json_dict)


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
        json_dict["steps"] = [(step_name, sklearn_model_to_json(step_model)) for (step_name, step_model) in model.steps]
        json_dict["memory"] = model.memory
        json_dict["verbose"] = model.verbose
        return json_dict
    json_dict.update(model.get_params())
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
        steps = [(step_name, sklearn_model_from_json(step_model)) for (step_name, step_model) in model_dict["steps"]]
        return Pipeline(steps, memory=model_dict["memory"], verbose=model_dict["verbose"])

    return model_class(**model_dict)
