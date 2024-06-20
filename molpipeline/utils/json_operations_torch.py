"""Functions for serializing and deserializing PyTorch models."""

from typing import TypeVar

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from typing import Any, Literal

_T = TypeVar("_T")

if TORCH_AVAILABLE:

    def tensor_to_json(
        obj: _T,
    ) -> tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]:
        """Recursively convert a PyTorch model to a JSON-serializable object.

        Parameters
        ----------
        obj : object
            The object to convert.

        Returns
        -------
        object
            The JSON-serializable object.
        """
        if isinstance(obj, torch.Tensor):
            object_dict: dict[str, Any] = {
                "__name__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "__init__": True,
            }
        else:
            return obj, False
        object_dict["data"] = obj.tolist()
        return object_dict, True

else:

    def tensor_to_json(
        obj: _T,
    ) -> tuple[dict[str, Any], Literal[True]] | tuple[_T, Literal[False]]:
        """Recursively convert a PyTorch model to a JSON-serializable object.

        Parameters
        ----------
        obj : object
            The object to convert.

        Returns
        -------
        object
            The JSON-serializable object.
        """
        return obj, False
