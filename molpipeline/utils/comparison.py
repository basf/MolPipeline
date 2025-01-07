"""Functions for comparing pipelines."""

from typing import Any, TypeVar

from molpipeline import Pipeline
from molpipeline.utils.json_operations import recursive_to_json

_T = TypeVar("_T", list[Any], tuple[Any, ...], set[Any], dict[Any, Any], Any)


def remove_irrelevant_params(params: _T) -> _T:
    """Remove irrelevant parameters from a dictionary.

    Parameters
    ----------
    params : TypeVar
        Parameters to remove irrelevant parameters from.

    Returns
    -------
    TypeVar
        Parameters without irrelevant parameters.
    """
    if isinstance(params, list):
        return [remove_irrelevant_params(val) for val in params]
    if isinstance(params, tuple):
        return tuple(remove_irrelevant_params(val) for val in params)
    if isinstance(params, set):
        return {remove_irrelevant_params(val) for val in params}

    irrelevant_params = ["n_jobs", "uuid", "error_filter_id"]
    if isinstance(params, dict):
        params_new = {}
        for key, value in params.items():
            if not isinstance(key, str):
                continue
            if key.split("__")[-1] in irrelevant_params:
                continue
            params_new[key] = remove_irrelevant_params(value)
        return params_new
    return params


def compare_recursive(  # pylint: disable=too-many-return-statements
    value_a: Any, value_b: Any
) -> bool:
    """Compare two values recursively.

    Parameters
    ----------
    value_a : Any
        First value to compare.
    value_b : Any
        Second value to compare.

    Returns
    -------
    bool
        True if the values are the same, False otherwise.
    """
    if value_a.__class__ != value_b.__class__:
        return False

    if isinstance(value_a, dict):
        if set(value_a.keys()) != set(value_b.keys()):
            return False
        for key in value_a:
            if not compare_recursive(value_a[key], value_b[key]):
                return False
        return True

    if isinstance(value_a, (list, tuple)):
        if len(value_a) != len(value_b):
            return False
        for val_a, val_b in zip(value_a, value_b):
            if not compare_recursive(val_a, val_b):
                return False
        return True
    return value_a == value_b


def check_pipelines_equivalent(pipeline_a: Pipeline, pipeline_b: Pipeline) -> bool:
    """Check if two pipelines are the same.

    Parameters
    ----------
    pipeline_a : Pipeline
        Pipeline to compare.
    pipeline_b : Pipeline
        Pipeline to compare.

    Returns
    -------
    bool
        True if the pipelines are the same, False otherwise.
    """
    if not isinstance(pipeline_a, Pipeline) or not isinstance(pipeline_b, Pipeline):
        raise ValueError("Both inputs should be of type Pipeline.")
    pipeline_json_a = recursive_to_json(pipeline_a)
    pipeline_json_a = remove_irrelevant_params(pipeline_json_a)
    pipeline_json_b = recursive_to_json(pipeline_b)
    pipeline_json_b = remove_irrelevant_params(pipeline_json_b)
    return compare_recursive(pipeline_json_a, pipeline_json_b)
