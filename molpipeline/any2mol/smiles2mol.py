"""Classes ment to transform given input to a RDKit molecule."""

from __future__ import annotations

from typing import Any, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    SimpleStringToMolElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class SmilesToMol(SimpleStringToMolElement):
    """Transforms Smiles to RDKit Mol objects."""

    def __init__(
        self,
        remove_hydrogens: bool = True,
        name: str = "smiles2mol",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize SmilesToMol object.

        Parameters
        ----------
        remove_hydrogens: bool
            Whether to remove hydrogens from the molecule.
        name: str
            Name of the object.
        n_jobs: int
            Number of jobs to run in parallel.
        uuid: Optional[str]
            UUID of the object.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self._remove_hydrogens = remove_hydrogens

    def _get_parser_config(self) -> Chem.SmilesParserParams:
        """Get parser configuration.

        Returns
        -------
        dict[str, Any]
            Configuration for the parser.
        """
        # set up rdkit smiles parser parameters
        parser_params = Chem.SmilesParserParams()
        parser_params.removeHs = self._remove_hydrogens
        return parser_params

    def string_to_mol(self, value: str) -> RDKitMol:
        """Transform Smiles string to molecule.

        Parameters
        ----------
        value: str
            SMILES string.

        Returns
        -------
        RDKitMol
            Rdkit molecule if valid SMILES, else None.
        """
        return Chem.MolFromSmiles(value, self._get_parser_config())

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this object.

        Parameters
        ----------
        deep: bool
            If True, return a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameters.
        """
        parameters = super().get_params(deep)
        if deep:
            parameters["remove_hydrogens"] = bool(self._remove_hydrogens)

        else:
            parameters["remove_hydrogens"] = self._remove_hydrogens
        return parameters

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: Any
            Dictionary of parameter names and values.

        Returns
        -------
        Self
            SmilesToMol pipeline element with updated parameters.
        """
        parameter_copy = dict(parameters)
        remove_hydrogens = parameter_copy.pop("remove_hydrogens", None)
        if remove_hydrogens is not None:
            self._remove_hydrogens = remove_hydrogens
        super().set_params(**parameter_copy)
        return self
