"""Contains functions for loading and saving objects to/from json files."""

import importlib
import inspect
import types
import typing
import warnings
from typing import Any, Literal, TypeVar

import joblib
import numpy as np
from loguru import logger

from molpipeline.pipeline import Pipeline

_T = TypeVar("_T")


__all__ = [
    "builtin_to_json",
    "decode_dict",
    "recursive_from_json",
    "recursive_to_json",
]

# The following functions are defined in a try-except block to avoid import errors if
# torch is not installed, which comes as an extra-dependency via molpipeline[chemprop].
# In case torch is not installed, the functions will just return the original object and
# a boolean indicating that the conversion was not successful.
if importlib.util.find_spec("torch") is not None:
    import torch

    def _tensor_to_json(
        obj: _T,
    ) -> tuple[dict[str, Any] | _T, bool]:
        """Recursively convert a PyTorch model to a JSON-serializable object.

        Parameters
        ----------
        obj : object
            The object to convert.

        Returns
        -------
        tuple[dict[str, Any] | _T, bool]
            If the object is a PyTorch model, a tuple containing the JSON-serializable
            dictionary and True is returned.
            Else a tuple containing the original object and False is returned.

        """
        if not isinstance(obj, torch.Tensor):
            return obj, False

        obj_dict = get_object_import_header(obj)
        obj_dict["data"] = recursive_to_json(obj.cpu().numpy())
        return obj_dict, True

    def _tensor_from_json(  # pylint: disable=unused-argument
        obj: type,
        *args: Any,  # noqa: ARG001
        **kwargs: Any,
    ) -> tuple[Any | type, bool]:
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
        tuple[Any | type, bool]
            If the object is a JSON-serializable PyTorch model, a tuple containing the
            PyTorch model and True is returned.
            Else a tuple containing the original object and False is returned.

        """
        if obj is torch.Tensor:
            return torch.from_numpy(kwargs["data"]), True
        return obj, False

else:

    def _tensor_to_json(
        obj: _T,
    ) -> tuple[dict[str, Any] | _T, bool]:
        """Recursively convert a PyTorch model to a JSON-serializable object.

        Parameters
        ----------
        obj : object
            The object to convert.

        Returns
        -------
        tuple[dict[str, Any] | _T, bool]
            Same signature as the function when torch is installed, but always returns
            the original object and False, as the conversion is not successful when
            torch is not installed.

        """
        return obj, False

    def _tensor_from_json(  # pylint: disable=unused-argument
        obj: type,
        *args: Any,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> tuple[Any | type, bool]:
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
        tuple[Any | type, bool]
            Same signature as the function when torch is installed, but always returns
            the original object and False, as the conversion is not successful when
            torch is not installed.

        """
        return obj, False


def get_object_import_header(obj: Any) -> dict[str, Any]:
    """Get the import header for an object.

    Parameters
    ----------
    obj: Any
        Object for which the import header is extracted.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the module and class name of the object.

    """
    if isinstance(obj, types.FunctionType):
        return {
            "__name__": obj.__name__,
            "__module__": obj.__module__,
            "__init__": False,
        }
    return {
        "__name__": obj.__class__.__name__,
        "__module__": obj.__class__.__module__,
        "__init__": True,
    }


def np_array_to_json(
    obj: _T,
) -> tuple[dict[str, Any] | _T, bool]:
    """Convert a vector to a JSON-serializable object.

    Parameters
    ----------
    obj : object
        The vector to convert.

    Returns
    -------
     tuple[dict[str, Any] | _T, bool]
        If the object is a numpy array, a tuple containing the JSON-serializable
        dictionary and True is returned.
        Else a tuple containing the original object and False is returned.

    """
    if not isinstance(obj, np.ndarray):
        return obj, False
    obj_dict = get_object_import_header(obj)
    obj_dict["__name__"] = "array"  # Override name to initialize as array
    obj_dict["__args__"] = [recursive_to_json(obj.tolist())]
    obj_dict["dtype"] = recursive_to_json(obj.dtype)
    return obj_dict, True


def np_dtype_to_json(
    obj: _T,
) -> tuple[dict[str, Any] | _T, bool]:
    """Convert a numpy dtype to a JSON-serializable object.

    Parameters
    ----------
    obj : object
        The numpy dtype to convert.

    Returns
    -------
    tuple[dict[str, Any] | _T, bool]
        If the object is a numpy dtype, a tuple containing the JSON-serializable
        dictionary and True is returned.
        Else a tuple containing the original object and False is returned.

    """
    if not isinstance(obj, np.dtype):
        return obj, False

    obj_dict = get_object_import_header(obj)
    obj_dict["__init__"] = False  # Numpy dtypes need no initialization
    return obj_dict, True


_OBJECT_SPECIFIC_TO_JSON_FUNCTIONS = [
    _tensor_to_json,
    np_array_to_json,
    np_dtype_to_json,
]
_OBJECT_SPECIFIC_FROM_JSON_FUNCTIONS = [_tensor_from_json]


_U = typing.TypeVar("_U", str, int, float, bool, None, type)


@typing.overload
def builtin_to_json(obj: _U) -> _U: ...


@typing.overload
def builtin_to_json(obj: list[Any]) -> list[Any]: ...


@typing.overload
def builtin_to_json(obj: tuple[Any, ...]) -> tuple[Any, ...]: ...


@typing.overload
def builtin_to_json(
    obj: types.FunctionType | set[Any] | dict[Any, Any],
) -> dict[str, Any]: ...


def builtin_to_json(obj: Any) -> Any:
    """Transform a builtin object to an object savable as json file.

    Parameters
    ----------
    obj : PythonNative
        Object to be transformed.
        Can be a string, int, float, bool, list, tuple, dict, callable or a set.

    Returns
    -------
    Any
        Json file containing the dictionary.

    Raises
    ------
    TypeError
        If the object is not a string, int, float, bool, list, tuple or dict.

    """
    if isinstance(obj, (str, int, float, bool, type)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {
            recursive_to_json(key): recursive_to_json(value)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        iter_list = [recursive_to_json(value) for value in obj]
        iterable_type = type(obj)
        return iterable_type(iter_list)

    obj_dict = get_object_import_header(obj)
    if isinstance(obj, types.FunctionType):
        obj_dict["__init__"] = False  # Functions need no initialization
        return obj_dict

    # If the object is a sklearn model, the parameters for initialization are extracted.
    if isinstance(obj, set):
        obj_dict["__set_items__"] = [recursive_to_json(value) for value in obj]
        return obj_dict
    raise TypeError(f"Unexpected Type: {type(obj)}")


def recursive_to_json(obj: Any) -> Any:
    """Recursively transform an object to a json file.

    Parameters
    ----------
    obj : Any
        Object to be transformed. Can be a string, int, float, bool, list, tuple, dict,
        callable or a sklearn model.
        A sklearn model is defined as an object with a get_params method.

    Returns
    -------
    Any
        Json file containing the dictionary.
        If the object cannot be transformed to a json file, a warning is raised and
        the import header of the object is returned, which might not be sufficient to
        recreate the object.
        Build-in objects (str, int, float, ...) are returned as is.

    """
    if obj is None:
        return None

    if isinstance(
        obj,
        (str, int, float, bool, dict, types.FunctionType, list, set, tuple, type),
    ):
        return builtin_to_json(obj)

    # If neither of the above, it is assumed to be an object.
    object_dict: dict[str, Any] = get_object_import_header(obj)
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
        for to_json_function in _OBJECT_SPECIFIC_TO_JSON_FUNCTIONS:
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
    obj : dict[str, Any]
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
        for from_json_function in _OBJECT_SPECIFIC_FROM_JSON_FUNCTIONS:
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
    obj : Any
        Object to be transformed

    Returns
    -------
    Any
        Object specified in the json file.

    Raises
    ------
    TypeError
        If the object is not a string, int, float, bool, list, tuple or dict.

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


def get_init_params(
    obj: Any,
    validation: Literal["raise", "warn", "skip", "return_none"],
) -> dict[str, Any] | None:
    """Get the parameters for initialization of an object.

    Parameters
    ----------
    obj : Any
        The object to get the parameters for.
    validation : Literal["raise", "warn", "skip", "return_none"]
        The validation strategy.

    Returns
    -------
    dict[str, Any] | None
        The parameters for initialization of the object, or None if the validation
        strategy is "return_none" and the object cannot be initialized with the
        extracted parameters.

    Raises
    ------
    ValueError


    """
    if hasattr(obj, "get_params"):
        return obj.get_params(deep=False)
    state_dict = obj.__getstate__()
    init_params = dict(inspect.signature(obj.__init__).parameters)
    init_params = {k: v for k, v in init_params.items() if k != "self"}
    allowed_params = init_params.keys()
    required_params = [
        key for key, param in init_params.items() if param.default is param.empty
    ]

    obj_params = {k: v for k, v in state_dict.items() if k in allowed_params}

    if validation == "skip":
        return obj_params

    missing_params = set(required_params) - set(obj_params.keys())
    if missing_params:
        msg = f"Missing required parameters: {missing_params}"
        if validation == "raise":
            raise ValueError(msg)
        if validation == "warn":
            logger.warning(msg)
            return obj_params
        if validation == "return_none":
            return None

    expected_state_hash = joblib.hash(recursive_to_json(state_dict))

    reconstructed_obj = obj.__class__(**obj_params)
    reconstructed_state = recursive_to_json(reconstructed_obj.__getstate__())
    reconstructed_state_hash = joblib.hash(reconstructed_state)

    if expected_state_hash != reconstructed_state_hash:
        msg = "Reconstructing the object failed."
        if validation == "raise":
            raise ValueError(msg)
        if validation == "warn":
            logger.warning(msg)
        if validation == "return_none":
            obj_params = None

    return obj_params
