"""Code for backward compatibility with old naming for ErrorHandling."""

import warnings
from typing import Any

from molpipeline.pipeline_elements.error_handling import ErrorFilter, FilterReinserter


# pylint: disable=too-few-public-methods
class NoneFiller(FilterReinserter):
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
    def from_none_filter(cls, *args: Any, **kwargs: Any) -> FilterReinserter:
        """Backward compatibility with old naming for ErrorReplacer.from_error_filter.

        Parameters
        ----------
        args: Any
            Arguments for ErrorReplacer.from_error_filter.
        kwargs: Any
            Keyword arguments for ErrorReplacer.from_error_filter.

        Returns
        -------
        FilterReinserter
            ErrorReplacer instance.
        """
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
