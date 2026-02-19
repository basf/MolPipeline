"""Contains functions for loading and saving objects to/from json files."""

import types
import typing
import warnings
from typing import Any, Literal, TypeVar

import numpy as np

from molpipeline.pipeline import Pipeline

_T = TypeVar("_T")


__all__ = [
    "builtin_to_json",
    "decode_dict",
    "recursive_from_json",
    "recursive_to_json",
    "transform_functions2string",
    "transform_string2function",
]

# The following functions are defined in a try-except block to avoid import errors if
# torch is not installed, which comes as an extra-dependency via molpipeline[chemprop].
# In case torch is not installed, the functions will just return the original object and
# a boolean indicating that the conversion was not successful.
try:
    import torch

    def _tensor_to_json(
        obj: _T,
    ) -> tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]:
        """Recursively convert a PyTorch model to a JSON-serializable object.

        Parameters
        ----------
        obj : object
            The object to convert.

        Returns
        -------
        tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]
            If the object is a PyTorch model, a tuple containing the JSON-serializable
            dictionary and True is returned.
            Else a tuple containing the original object and False is returned.

        """
        if not isinstance(obj, torch.Tensor):
            return obj, False

        object_dict: dict[str, Any] = {
            "__name__": obj.__class__.__name__,
            "__module__": obj.__class__.__module__,
            "__init__": True,
            "data": recursive_to_json(obj.cpu().numpy()),
        }
        return object_dict, True

    def _tensor_from_json(  # pylint: disable=unused-argument
        obj: type,
        *args: Any,  # noqa: ARG001
        **kwargs: Any,
    ) -> tuple[Any, Literal[True]] | tuple[type, Literal[False]]:
        """Recursively convert a JSON-serializable object to a PyTorch model.

        Parameters
        ----------
        obj : type
            The object to initialize.
        args : Any
            Positional arguments for initialization.
        kwargs : Any
            Keyword arguments for initialization.

        Returns
        -------
        tuple[Any, Literal[True]] | tuple[type, Literal[False]]
            If the object is a JSON-serializable PyTorch model, a tuple containing the
            PyTorch model and True is returned.
            Else a tuple containing the original object and False is returned.

        """
        if obj is torch.Tensor:
            return torch.from_numpy(kwargs["data"]), True
        return obj, False


except ImportError:

    def _tensor_to_json(
        obj: _T,
    ) -> tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]:
        """Recursively convert a PyTorch model to a JSON-serializable object.

        Parameters
        ----------
        obj : object
            The object to convert.

        Returns
        -------
        tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]
            Same signature as the function when torch is installed, but always returns
            the original object and False, as the conversion is not successful when
            torch is not installed.

        """
        return obj, False

    def _tensor_from_json(  # pylint: disable=unused-argument
        obj: type,
        *args: Any,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> tuple[Any, Literal[True]] | tuple[type, Literal[False]]:
        """Recursively convert a JSON-serializable object to a PyTorch model.

        Parameters
        ----------
        obj : type
            The object to initialize.
        args: Any
            Positional arguments for initialization.
        kwargs: Any
            Keyword arguments for initialization.

        Returns
        -------
        tuple[Any, Literal[True]] | tuple[type, Literal[False]]
            Same signature as the function when torch is installed, but always returns
            the original object and False, as the conversion is not successful when
            torch is not installed.

        """
        return obj, False


def np_array_to_json(
    obj: _T,
) -> tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]:
    """Convert a vector to a JSON-serializable object.

    Parameters
    ----------
    obj : object
        The vector to convert.

    Returns
    -------
     tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]
        If the object is a numpy array, a tuple containing the JSON-serializable
        dictionary and True is returned.
        Else a tuple containing the original object and False is returned.

    """
    if not isinstance(obj, np.ndarray):
        return obj, False

    obj_dict = {
        "__name__": "array",
        "__module__": obj.__class__.__module__,
        "__init__": True,
        "__args__": [recursive_to_json(obj.tolist())],
        "dtype": recursive_to_json(obj.dtype),
    }
    return obj_dict, True


def np_dtype_to_json(
    obj: _T,
) -> tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]:
    """Convert a numpy dtype to a JSON-serializable object.

    Parameters
    ----------
    obj : object
        The numpy dtype to convert.

    Returns
    -------
    tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]
        If the object is a numpy dtype, a tuple containing the JSON-serializable
        dictionary and True is returned.
        Else a tuple containing the original object and False is returned.

    """
    if not isinstance(obj, np.dtype):
        return obj, False

    obj_dict = {
        "__name__": type(obj).__name__,
        "__module__": obj.__class__.__module__,
        "__init__": False,
    }
    return obj_dict, True


_OBJECT_SPECIF_TO_JSON_FUNTIONS = [_tensor_to_json, np_array_to_json, np_dtype_to_json]
_OBJECT_SPECIF_FROM_JSON_FUNCTIONS = [_tensor_from_json]


def transform_functions2string(value: Any) -> Any:
    """Transform functions to string representation.

    If the value is a function, it is transformed to a dictionary containing the module
    and the class name. If the value is a dictionary, the function is called recursively
    for each value. If the value is a list, the function is called recursively for each
    value. Else the value is returned as is.

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


@deprecated(
    "`transform_string2function` is deprecated and will be removed in a future version."
    " Please use `recursive_from_json` instead."
)
def transform_string2function(value: Any) -> Any:
    """Transform string representation of functions to actual functions.

    If the value is a dictionary containing the key "load_from_constructor" and the
    value is True, the function is loaded from the module and class name.
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


_U = typing.TypeVar("_U", str, int, float, bool, None, type)


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
    obj: types.FunctionType | set[Any] | dict[Any, Any],
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
        Object to be transformed.
        Can be a string, int, float, bool, list, tuple, dict, callable or a set.

    Raises
    ------
    TypeError
        If the object is not a string, int, float, bool, list, tuple or dict.

    Returns
    -------
    Any
        Json file containing the dictionary.

    """
    if isinstance(obj, (str, int, float, bool, type)) or obj is None:
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
        Object to be transformed. Can be a string, int, float, bool, list, tuple, dict,
        callable or a sklearn model.
        A sklearn model is defined as an object with a get_params method.

    Returns
    -------
    dict[str, Any]
        Json file containing the dictionary.

    """
    if obj is None:
        return None

    if isinstance(
        obj,
        (str, int, float, bool, dict, types.FunctionType, list, set, tuple, type),
    ):
        return builtin_to_json(obj)

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
        for to_json_function in _OBJECT_SPECIF_TO_JSON_FUNTIONS:
            obj_dict, success = to_json_function(obj)
            if success:
                return obj_dict
        # If the object is not a sklearn model, a warning is raised
        # as it might not be possible to recreate the object.
        warnings.warn(
            f"{type(obj)} has no get_params method. "
            f"No parameters for initialization are retained.",
            stacklevel=2,
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
    args = object_params_copy.pop("__args__", [])

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
        if obj_class is set:
            return set(converted_dict["__set_items__"])
        for from_json_function in _OBJECT_SPECIF_FROM_JSON_FUNCTIONS:
            obj_class, success = from_json_function(obj_class, *args, **converted_dict)
            if success:
                return obj_class

        # If the object is a class, and has parameters
        return obj_class(*args, **converted_dict)
    return converted_dict


def recursive_from_json(obj: Any) -> Any:
    """Recursively transform a json file to an object.

    Parameters
    ----------
    obj: Any
        Object to be transformed

    Raises
    ------
    TypeError
        If the object is not a string, int, float, bool, list, tuple or dict.

    Returns
    -------
    Any
        Object specified in the json file.

    """
    if isinstance(obj, (str, int, float, bool, type)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return decode_dict(obj)

    if isinstance(obj, (list, tuple)):
        iter_list = [recursive_from_json(value) for value in obj]
        iterable_type = type(obj)
        return iterable_type(iter_list)

    raise TypeError(f"Unexpected Type: {type(obj)}")
