"""File loader for loading files from URLs."""

from typing import Any

import requests
from typing_extensions import override

from molpipeline.utils.file_loading.abc_file_loading import ABCFileLoader


class URLFileLoader(ABCFileLoader):
    """File loader for loading files from URLs."""

    url: str
    timeout: int

    def __init__(self, url: str, timeout: int = 60) -> None:
        """Initialize the URL file loader.

        Parameters
        ----------
        url : str
            The url to the file to load.
        timeout : int, default=60
            The timeout for the request in seconds.

        """
        self.url = url
        self.timeout = timeout

    def load_file(self) -> bytes:
        """Load a file from the given URL.

        Returns
        -------
        bytes
            The content of the file as bytes.

        """
        response = requests.get(self.url, timeout=self.timeout)
        response.raise_for_status()
        return response.content

    @override
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
        return {"url": self.url, "timeout": self.timeout}

    def set_params(self, **params: Any) -> None:
        """Set the parameters of the file loader.

        Parameters
        ----------
        **params : Any
            The parameters to set.

        """
        if "url" in params:
            self.url = params["url"]
        if "timeout" in params:
            self.timeout = params["timeout"]
