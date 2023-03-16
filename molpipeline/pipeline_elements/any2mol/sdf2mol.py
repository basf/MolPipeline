"""Class for Transforming SDF-strings to rdkit molecules."""

from __future__ import annotations
from typing import Any

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.any2mol.string2mol import (
    StringToMolPipelineElement as _StringToMolPipelineElement,
)
from molpipeline.abstract_pipeline_elements.core import NONE_HANDLING_OPTIONS
from molpipeline.utils.molpipe_types import OptionalMol


class SDFToMolPipelineElement(_StringToMolPipelineElement):
    """PipelineElement transforming a list of SDF strings to mol_objects."""

    identifier: str
    mol_counter: int

    def __init__(
        self,
        identifier: str = "enumerate",
        none_handling: NONE_HANDLING_OPTIONS = "raise",
        fill_value: Any = None,
        name: str = "SDF2Mol",
        n_jobs: int = 1,
    ) -> None:
        """Initialize SDFToMolPipelineElement.

        Parameters
        ----------
        identifier: str
            Method of assigning identifiers to molecules. At the moment molecules are counted.
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores used for processing.
        """
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )
        self.identifier = identifier
        self.mol_counter = 0

    @property
    def params(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        params = super(SDFToMolPipelineElement, self).params
        params["identifier"] = self.identifier
        return params

    def copy(self) -> SDFToMolPipelineElement:
        """Create a copy of the object."""
        return SDFToMolPipelineElement(**self.params)

    def finish(self) -> None:
        """Reset the mol counter which assigns identifiers."""
        self.mol_counter = 0

    def _transform_single(self, value: str) -> OptionalMol:
        """Transform an SDF-strings to a rdkit molecule."""
        mol = Chem.MolFromMolBlock(value)
        if self.identifier == "smiles":
            mol.SetProp("identifier", self.mol_counter)
        self.mol_counter += 1
        return mol
