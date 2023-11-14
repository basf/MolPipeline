"""Classes for handling substructures.

This is only relevant for explainable AI, where atoms need to be mapped to features.
"""

from __future__ import annotations

from rdkit import Chem


# pylint: disable=R0903
class AtomEnvironment:
    """A Class to store environment-information for fingerprint features."""

    def __init__(self, environment_atoms: set[int]):
        """Initialize AtomEnvironment.

        Parameters
        ----------
        environment_atoms: set[int]
            Indices of atoms encoded by environment.
        """
        self.environment_atoms = environment_atoms  # set of all atoms within radius


# pylint: disable=R0903
class CircularAtomEnvironment(AtomEnvironment):
    """A Class to store environment-information for morgan-fingerprint features."""

    def __init__(self, central_atom: int, radius: int, environment_atoms: set[int]):
        """Initialize CircularAtomEnvironment.

        Parameters
        ----------
        central_atom: int
            Index of central atom in circular fingerprint.
        radius: int
            Radius of feature.
        environment_atoms: set[int]
            All indices of atoms within radius of central atom.
        """
        super().__init__(environment_atoms)
        self.central_atom = central_atom
        self.radius = radius

    @classmethod
    def from_mol(
        cls, mol: Chem.Mol, central_atom_index: int, radius: int
    ) -> CircularAtomEnvironment:
        """Generate class from mol, using location (central_atom_index) and the radius.

        Parameters
        ----------
        mol: Chem.Mol
            Molecule from which the environment is derived.
        central_atom_index: int
            Index of central atom in feature.
        radius: int
            Radius of feature.

        Returns
        -------
        CircularAtomEnvironment
            Encoded the atoms which are within the radius of the central atom and are part of the feature.
        """
        if radius == 0:
            return CircularAtomEnvironment(
                central_atom_index, radius, {central_atom_index}
            )

        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, central_atom_index)
        amap: dict[int, int] = {}
        _ = Chem.PathToSubmol(mol, env, atomMap=amap)
        env_atoms = amap.keys()
        return CircularAtomEnvironment(central_atom_index, radius, set(env_atoms))
