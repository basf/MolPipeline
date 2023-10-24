"""Code for backward compatibility with old naming for ErrorHandling."""
from typing import Any
import warnings

from molpipeline.pipeline_elements.error_handling import (
    ErrorReplacer,
    ErrorFilter,
)


# pylint: disable=too-few-public-methods
class NoneFiller(ErrorReplacer):
    """Backward compatibility with old naming for ErrorFiller."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize NoneFiller.

        Parameters
        ----------
        args: Any
            Arguments for ErrorFiller.
        kwargs: Any
            Keyword arguments for ErrorFiller.

        Returns
        -------
        None
        """
        warnings.warn("NoneFiller is deprecated, use ErrorReplacer instead")
        super().__init__(*args, **kwargs)

    @classmethod
    def from_none_filter(cls, *args, **kwargs) -> ErrorReplacer:
        """Backward compatibility with old naming for ErrorReplacer.from_error_filter."""
        return super().from_error_filter(*args, **kwargs)


# pylint: disable=too-few-public-methods
class NoneFilter(ErrorFilter):
    """Backward compatibility with old naming for ErrorFilter."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize NoneFilter.

        Parameters
        ----------
        args: Any
            Arguments for ErrorFilter.
            kwargs: Any
            Keyword arguments for ErrorFilter.

        Returns
        -------
        None
        """
        warnings.warn("NoneFilter is deprecated, use ErrorFilter instead")
        super().__init__(*args, **kwargs)
