"""Pipeline elements for converting instances to bool."""

from typing import Any

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToAnyPipelineElement,
)


class MolToBool(MolToAnyPipelineElement):
    """
    Element to generate a bool array from input.

    Valid molecules are passed as True, InvalidInstances are passed as False.
    """

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

    def transform_single(self, value: Any) -> Any:
        """Transform a single molecule to a bool representation.

        Valid molecules are passed as True, InvalidInstances are passed as False.
        RemovedMolecule objects are passed without change, as no transformations are applicable.

        Parameters
        ----------
        value: Any
            Current representation of the molecule. (Eg. SMILES, RDKit Mol, ...)

        Returns
        -------
        Any
            Bool representation of the molecule.
        """
        pre_value = self.pretransform_single(value)
        return self.finalize_single(pre_value)
