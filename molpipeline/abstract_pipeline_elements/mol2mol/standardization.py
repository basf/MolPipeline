"""Abstract pipeline elements for standardization of molecules."""
import abc

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
    RDKitMol,
    OptionalMol,
)

class EmptyMolCheckerPipelineElement(_MolToMolPipelineElement, abc.ABC):
    """EmptyMolCheckerPipelineElement which removes empty molecules."""

    def __init__(
        self,
        remove_empty: bool = True,
        name: str = "EmptyMolCheckerPipelineElement",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize EmptyMolCheckerPipelineElement.

        Parameters
        ----------
        remove_empty: bool, optional (default: True)
            If True, remove empty molecules.
        name: str, optional (default: "EmptyMolCheckerPipelineElement")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.remove_empty = remove_empty

    def finalize_single(self, value: RDKitMol) -> OptionalMol:
        """Check if molecule is empty.

        Parameters
        ----------
        value: RDKitMol
            Molecule to check.

        Returns
        -------
        OptionalMol
            Molecule if it is not empty, else InvalidInstance.
        """
        if not self.remove_empty:
            return value
        atoms = value.GetNumAtoms()
        if atoms == 0:
            return InvalidInstance(
                self.uuid, "Molecule contains no atoms.", self.name
            )
        return value
