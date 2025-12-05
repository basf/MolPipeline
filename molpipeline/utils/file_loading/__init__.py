"""Initialize module for file loading."""

import importlib.util

from molpipeline.utils.file_loading.dvc_file_loading import ABCFileLoader

__all__ = [
    "ABCFileLoader",
]

if importlib.util.find_spec("requests") is not None:
    from molpipeline.utils.file_loading.url_file_loading import URLFileLoader

    __all__ += ["URLFileLoader"]


if importlib.util.find_spec("dvc") is not None:
    from molpipeline.utils.file_loading.dvc_file_loading import DVCFileLoader

    __all__ += ["DVCFileLoader"]
