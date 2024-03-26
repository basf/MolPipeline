"""Classes for standardizing molecules."""

from __future__ import annotations

from typing import Optional

from rdkit.Chem.Scaffolds import MurckoScaffold as RDKIT_MurckoScaffold

from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
)
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


class MurckoScaffold(_MolToMolPipelineElement):
    """MolToMol-PipelineElement which yields the Murcko-scaffold of a Molecule.

    The Murcko-scaffold is composed of all rings and the linker atoms between them.
    """

    def __init__(
        self,
        name: str = "MurckoScaffold",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MurckoScaffold.

        Parameters
        ----------
        name: str
            Name of pipeline element.
        n_jobs: int
            Number of jobs to use for parallelization.
        uuid: Optional[str]
            UUID of pipeline element.

        Returns
        -------
        None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Extract Murco-scaffold of molecule.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule object which is transformed.

        Returns
        -------
        OptionalMol
            Murco-scaffold of molecule if possible, else InvalidInstance.
        """
        return RDKIT_MurckoScaffold.GetScaffoldForMol(value)


class MakeScaffoldGeneric(_MolToMolPipelineElement):
    """MolToMol-PipelineElement which sets all atoms to carbon and all bonds to single bond.

    Done to make scaffolds less speciffic.
    """

    def __init__(
        self,
        name: str = "MakeScaffoldGeneric",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MakeScaffoldGeneric.

        Parameters
        ----------
        name: str
            Name of pipeline element.
        n_jobs: int
            Number of jobs to use for parallelization.
        uuid: Optional[str]
            UUID of pipeline element.

        Returns
        -------
        None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Set all atoms to carbon and all bonds to single bond and return mol object.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule object which is transformed.

        Returns
        -------
        OptionalMol
            Molecule where all atoms are carbon and all bonds are single bonds.
            If transformation failed, it returns InvalidInstance.
        """
        return RDKIT_MurckoScaffold.MakeScaffoldGeneric(value)
