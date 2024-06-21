"""Contains functions for loading and saving pipeline elements and models to json files."""

from __future__ import annotations

import types
import typing
import warnings
from typing import Any

from molpipeline.pipeline import Pipeline
from molpipeline.utils.json_operations_torch import tensor_to_json

__all__ = [
    "builtin_to_json",
    "decode_dict",
    "recursive_from_json",
    "recursive_to_json",
    "transform_functions2string",
    "transform_string2function",
]


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
        for dict_key, dict_value in value.items():
            out_dict[dict_key] = transform_functions2string(dict_value)
        return out_dict

    if isinstance(value, list):
        out_list = []
        for list_value in value:
            out_list.append(transform_functions2string(list_value))
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
            return value
        out_dict = {}
        for dict_key, dict_value in value.items():
            out_dict[dict_key] = transform_string2function(dict_value)
        return out_dict

    if isinstance(value, list):
        out_list = []
        for list_value in value:
            out_list.append(transform_string2function(list_value))
        return out_list

    return value


_U = typing.TypeVar("_U", str, int, float, bool, None)


@typing.overload
def builtin_to_json(obj: _U) -> _U:
    """Transform a builtin object to an object savable as json file.

    Parameters
    ----------
    obj: str | int | float | bool | None
        Object which would be transformed, but these types are just returned as is.

    Returns
    -------
    str | int | float | bool | None
        The same object as the input.
    """


@typing.overload
def builtin_to_json(obj: list[Any]) -> list[Any]:
    """Transform a builtin object to an object savable as json file.

    Parameters
    ----------
    obj: list[Any]
        List of objects transformed recursively to json compatible objects.

    Returns
    -------
    list[Any]
        List of transformed objects.
    """


@typing.overload
def builtin_to_json(obj: tuple[Any, ...]) -> tuple[Any, ...]:
    """Transform a builtin object to an object savable as json file.

    Parameters
    ----------
    obj: tuple[Any]
        Tuple of objects transformed recursively to json compatible objects.

    Returns
    -------
    tuple[Any]
        Tuple of transformed objects.
    """


@typing.overload
def builtin_to_json(
    obj: types.FunctionType | set[Any] | dict[Any, Any]
) -> dict[str, Any]:
    """Transform a builtin object to an object savable as json file.

    Parameters
    ----------
    obj: types.FunctionType | set[Any] | dict[Any, Any]
        Object which are encoded as a dictionary in order to be json compatible.

    Returns
    -------
    tuple[Any]
        Tuple of transformed objects.
    """


def builtin_to_json(obj: Any) -> Any:
    """Transform a builtin object to an object savable as json file.

    Parameters
    ----------
    obj: PythonNative
        Object to be transformed. Can be a string, int, float, bool, list, tuple, dict, callable or a set.

    Returns
    -------
    Any
        Json file containing the dictionary.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, types.FunctionType):
        return {
            "__name__": obj.__name__,
            "__module__": obj.__module__,
            "__init__": False,
        }
    if isinstance(obj, dict):
        return {
            recursive_to_json(key): recursive_to_json(value)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        iter_list = [recursive_to_json(value) for value in obj]
        iterable_type = type(obj)
        return iterable_type(iter_list)

    object_dict: dict[str, Any] = {
        "__name__": obj.__class__.__name__,
        "__module__": obj.__class__.__module__,
        "__init__": True,
    }
    # If the object is a sklearn model, the parameters for initialization are extracted.
    if isinstance(obj, set):
        object_dict["__set_items__"] = [recursive_to_json(value) for value in obj]
        return object_dict
    raise TypeError(f"Unexpected Type: {type(obj)}")


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

    if isinstance(
        obj, (str, int, float, bool, dict, types.FunctionType, list, set, tuple)
    ):
        return_value = builtin_to_json(obj)
        return return_value

    # If neither of the above, it is assumed to be an object.
    object_dict: dict[str, Any] = {
        "__name__": obj.__class__.__name__,
        "__module__": obj.__class__.__module__,
        "__init__": True,
    }
    # If the object is a sklearn model, the parameters for initialization are extracted.
    if hasattr(obj, "get_params"):
        if isinstance(obj, Pipeline):
            object_dict["steps"] = [
                (step_name, recursive_to_json(step_model))
                for (step_name, step_model) in obj.steps
            ]
            object_dict["memory"] = obj.memory
            object_dict["verbose"] = obj.verbose
            object_dict["n_jobs"] = obj.n_jobs
        else:
            model_params = obj.get_params(deep=False)
            for key, value in model_params.items():
                object_dict[key] = recursive_to_json(value)
    else:
        obj_dict, success = tensor_to_json(obj)
        # Either not a tensor or torch is not available
        if success:
            return obj_dict
        # If the object is not a sklearn model, a warning is raised
        # as it might not be possible to recreate the object.
        warnings.warn(
            f"{type(obj)} has no get_params method. No parameters for initialization are retained."
        )

    return object_dict


def decode_dict(obj: dict[str, Any]) -> Any:
    """Decode a dictionary to an object.

    Parameters
    ----------
    obj: dict[str, Any]
        Dictionary to be transformed

    Returns
    -------
    Any
        Object specified in the dictionary.
    """
    # Create copy
    object_params_copy = dict(obj)

    # For functions of classes
    obj_module_str = object_params_copy.pop("__module__", None)
    obj_class_str = object_params_copy.pop("__name__", None)
    initialize = object_params_copy.pop("__init__", False)

    # Convert remaining Values
    converted_dict = {}
    for key, value in object_params_copy.items():
        converted_dict[key] = recursive_from_json(value)

    # If the object is a function or a class
    if obj_module_str and obj_class_str:
        class_module = __import__(obj_module_str, fromlist=[obj_class_str])
        obj_class = getattr(class_module, obj_class_str)
        if not initialize:  # If the object is a function or should not be initialized
            return obj_class
        # If the object is a class, but has no parameters
        if not converted_dict:
            return obj_class()
        if obj_class is set:
            return set(converted_dict["__set_items__"])
        # If the object is a class, and has parameters
        return obj_class(**converted_dict)
    return converted_dict


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
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return decode_dict(obj)

    if isinstance(obj, (list, tuple)):
        iter_list = [recursive_from_json(value) for value in obj]
        iterable_type = type(obj)
        return iterable_type(iter_list)

    raise TypeError(f"Unexpected Type: {type(obj)}")
