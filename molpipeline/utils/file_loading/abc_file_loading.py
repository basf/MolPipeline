"""Functionality for loading files from various sources."""

import abc
from typing import Any


class ABCFileLoader(abc.ABC):
    """Abstract base class for file loaders."""

    @abc.abstractmethod
    def load_file(self) -> bytes:
        """Load a file from the given path.

        Returns
        -------
        bytes
            The content of the file as bytes.

        """

    @abc.abstractmethod
    def get_params(self, deep: bool = False) -> dict[str, Any]:
        """Get the parameters of the file loader.

        Parameters
        ----------
        deep : bool, default=False
            Whether to return the parameters of sub-objects (if any).

        Returns
        -------
        dict[str, Any]
            The parameters of the file loader.

        """

    @abc.abstractmethod
    def set_params(self, **params: Any) -> None:
        """Set the parameters of the file loader.

        Parameters
        ----------
        **params : Any
            The parameters to set.

        """
