"""Classes for standardizing molecules."""

from __future__ import annotations

from typing import Any, Optional

try:
    from typing import Self  # pylint: disable=no-name-in-module
except ImportError:
    from typing_extensions import Self

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold as RDKIT_MurckoScaffold

from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
)
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


class MurckoScaffold(_MolToMolPipelineElement):
    """MolToMol-PipelineElement which yields the Murcko-scaffold of a Molecule.

    The Murcko-scaffold is composed of all rings and the linker atoms between them.
    """

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
        generic_atoms: bool = False,
        generic_bonds: bool = False,
        name: str = "MakeScaffoldGeneric",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MakeScaffoldGeneric.

        Note
        ----
        Making atoms or bonds generic will generate SMARTS strings instead of SMILES strings.
        This can be useful to search for scaffolds and substructures in data sets.
        Per default, the scaffold is returned as SMILES string with all atoms set to carbon and all bonds are single bonds.

        Parameters
        ----------
        generic_atoms: bool
            If True, all atoms in the molecule are set to generic atoms (*).
        generic_bonds: bool
            If True, all bonds in the molecule are set to any bonds.
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
        self.generic_atoms = generic_atoms
        self.generic_bonds = generic_bonds
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
        scaffold = RDKIT_MurckoScaffold.MakeScaffoldGeneric(value)
        if self.generic_atoms:
            for atom in scaffold.GetAtoms():
                atom.SetAtomicNum(0)
        if self.generic_bonds:
            for bond in scaffold.GetBonds():
                bond.SetBondType(Chem.rdchem.BondType.UNSPECIFIED)
        return scaffold

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of pipeline element.

        Parameters
        ----------
        deep: bool
            If True, return the parameters of the pipeline element.

        Returns
        -------
        dict[str, Any]
            Parameters of the pipeline element.
        """
        parent_params = super().get_params()
        if deep:
            parent_params.update(
                {
                    "generic_atoms": bool(self.generic_atoms),
                    "generic_bonds": bool(self.generic_bonds),
                }
            )
        else:
            parent_params.update(
                {
                    "generic_atoms": self.generic_atoms,
                    "generic_bonds": self.generic_bonds,
                }
            )
        return parent_params

    def set_params(self, **parameters: dict[str, Any]) -> Self:
        """Set parameters of pipeline element.

        Parameters
        ----------
        parameters: dict[str, Any]
            Parameters to set.

        Returns
        -------
        Self
            Pipeline element with set parameters.
        """
        param_copy = parameters.copy()
        generic_atoms = param_copy.pop("generic_atoms", None)
        generic_bonds = param_copy.pop("generic_bonds", None)
        if generic_atoms is not None:
            if not isinstance(generic_atoms, bool):
                raise ValueError("generic_atoms must be a boolean.")
            self.generic_atoms = generic_atoms
        if generic_bonds is not None:
            if not isinstance(generic_bonds, bool):
                raise ValueError("generic_bonds must be a boolean.")
            self.generic_bonds = generic_bonds
        return self
