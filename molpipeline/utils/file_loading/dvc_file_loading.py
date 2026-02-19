"""File loader to load files via DVC."""

from typing import Any

import dvc.api
from typing_extensions import override

from molpipeline.utils.file_loading.abc_file_loading import ABCFileLoader


class DVCFileLoader(ABCFileLoader):
    """File loader to load files via DVC."""

    file_path: str
    repo: str | None
    rev: str | None

    def __init__(
        self,
        file_path: str,
        repo: str | None = None,
        rev: str | None = None,
    ) -> None:
        """Initialize the DVC file loader.

        Parameters
        ----------
        file_path : str
            The path to the file in the DVC repository.
        repo : str | None, default None
            The path to the DVC repository. If None, the current directory is used.
        rev : str | None, default None
            The git revision (branch, tag, commit) to use.
            If None, the current revision is used.

        """
        self.file_path = file_path
        self.repo = repo
        self.rev = rev

    def load_file(self) -> bytes:
        """Load a file from the DVC repository.

        Returns
        -------
        bytes
            The content of the file as bytes.

        """
        with dvc.api.open(self.file_path, repo=self.repo, rev=self.rev, mode="rb") as f:
            return f.read()

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
        return {"file_path": self.file_path, "repo": self.repo, "rev": self.rev}

    def set_params(self, **params: Any) -> None:
        """Set the parameters of the file loader.

        Parameters
        ----------
        **params : Any
            The parameters to set.

        """
        if "file_path" in params:
            self.file_path = params["file_path"]
        if "repo" in params:
            self.repo = params["repo"]
        if "rev" in params:
            self.rev = params["rev"]
