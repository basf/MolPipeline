"""Class for Transforming SDF-strings to rdkit molecules."""

from __future__ import annotations

from typing import Any

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    StringToMolPipelineElement as _StringToMolPipelineElement,
)
from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.utils.molpipeline_types import OptionalMol


class SDFToMol(_StringToMolPipelineElement):
    """PipelineElement transforming a list of SDF strings to mol_objects."""

    identifier: str | None

    def __init__(
        self,
        identifier: str | None = "smiles",
        name: str = "SDF2Mol",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize SDFToMol.

        Parameters
        ----------
        identifier: str | None, default='enumerate'
            Method of assigning identifiers to molecules.
            At the moment molecules are counted.
        name: str, default='SDF2Mol'
            Name of PipelineElement
        n_jobs: int, default=1
            Number of cores used for processing.
        uuid: str | None, optional
            uuid of PipelineElement, by default None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.identifier = identifier

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all parameters defining the object.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters defining the object.
        """
        params = super().get_params(deep)
        if deep:
            params["identifier"] = copy.copy(self.identifier)
        else:
            params["identifier"] = self.identifier
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters of the object.

        Parameters
        ----------
        parameters: Any
            Dictionary containing all parameters defining the object.

        Returns
        -------
        Self
            SDFToMol with updated parameters.
        """
        super().set_params(**parameters)
        if "identifier" in parameters:
            self.identifier = parameters["identifier"]
        return self

    def pretransform_single(self, value: str) -> OptionalMol:
        """Transform an SDF-strings to a rdkit molecule.

        Parameters
        ----------
        value: str
            SDF-string to transform to a molecule.

        Returns
        -------
        OptionalMol
            Molecule if transformation was successful, else InvalidInstance.
        """
        if not isinstance(value, (str, bytes)):
            return InvalidInstance(
                self.uuid,
                "Invalid SDF string!",
                self.name,
            )
        mol = Chem.MolFromMolBlock(value)
        if mol is None:
            return InvalidInstance(
                self.uuid,
                "Invalid SDF string!",
                self.name,
            )
        if self.identifier == "smiles":
            mol.SetProp("identifier", Chem.MolToSmiles(mol))
        return mol
