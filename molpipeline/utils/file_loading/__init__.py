"""Initialize module for file loading."""

import importlib.util

from molpipeline.utils.file_loading.url_file_loading import URLFileLoader

__all__ = ["URLFileLoader"]


dvc_spec = importlib.util.find_spec("dvc")
if dvc_spec is not None:
    from molpipeline.utils.file_loading.dvc_file_loading import DVCFileLoader

    __all__ += ["DVCFileLoader"]
