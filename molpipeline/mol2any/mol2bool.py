"""Pipeline elements for converting instances to bool."""

from typing import Any

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToAnyPipelineElement,
)


class MolToBool(MolToAnyPipelineElement):
    """Element to generate a bool array from input."""


    def pretransform_single(self, value: Any) -> bool:
        """Transform a value to a bool representation.

        Parameters
        ----------
        value: Any
            Value to be transformed to bool representation.

        Returns
        -------
        str
            Binary representation of molecule.
        """
        if isinstance(value, InvalidInstance):
            return False
        return True
