"""File loader for loading files from URLs."""

from typing import Any

import requests

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

    def get_params(self) -> dict[str, Any]:
        """Get the parameters of the file loader.

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
