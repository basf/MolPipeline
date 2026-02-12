"""Abstract classes for creating rdkit molecules from string representations."""

import abc

from molpipeline.abstract_pipeline_elements.core import (
    AnyToMolPipelineElement,
    InvalidInstance,
)
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


class StringToMolPipelineElement(AnyToMolPipelineElement, abc.ABC):
    """Abstract class for transforming strings to molecules."""

    _input_type = "str"
    _output_type = "RDKitMol"

    def transform(self, values: list[str]) -> list[OptionalMol]:
        """Transform the list of molecules to sparse matrix.

        Parameters
        ----------
        values: list[str]
            List of string representations of molecules.

        Returns
        -------
        list[OptionalMol]
            List of RDKit molecules.
            InvalidInstance if the representation was invalid.

        """
        return super().transform(values)

    @abc.abstractmethod
    def pretransform_single(self, value: str) -> OptionalMol:
        """Transform mol to a string.

        Parameters
        ----------
        value: str
            Representation transformed to a RDKit molecule.

        Returns
        -------
        OptionalMol
            RDKit molecule if representation was valid, else InvalidInstance.

        """


class SimpleStringToMolElement(StringToMolPipelineElement, abc.ABC):
    """Transforms string representation to RDKit Mol objects."""

    def pretransform_single(self, value: str) -> OptionalMol:
        """Transform string to molecule.

        Parameters
        ----------
        value: str
            String representation of the molecule.

        Returns
        -------
        OptionalMol
            Rdkit molecule if valid string representation, else None.

        """
        if not isinstance(value, str):
            return InvalidInstance(
                self.uuid,
                f"Not a string: {value}",
                self.name,
            )

        mol: RDKitMol = self.string_to_mol(value)

        if not mol:
            return InvalidInstance(
                self.uuid,
                f"Invalid representation: {value}",
                self.name,
            )
        mol.SetProp("identifier", value)
        return mol

    @abc.abstractmethod
    def string_to_mol(self, value: str) -> RDKitMol | None:
        """Transform string representation to molecule.

        Parameters
        ----------
        value: str
            String representation of the molecule.

        Returns
        -------
        RDKitMol | None
            Rdkit molecule if valid representation, else None.

        """
